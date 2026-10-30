"""
backend/nebius_client.py
========================

Thin, defensive wrapper around the Nebius Studio (AI Studio) OpenAI-compatible
endpoint, used by every other module of the OMEGA x PHANTOM x PROMETHEUS stack.

    base_url : "https://integrate.api.nvidia.com/v1"
    model    : nvidia/Llama-3.1-Nemotron-70B-Instruct-HF   (default)
    auth     : NEBIUS_API_KEY environment variable (or the `api_key` argument)

Design goals
------------
* Never crash the demo: if the key or the `openai` package is missing the client
  reports `available == False` and callers fall back to deterministic logic.
* Structured output you can trust: `structured_completion()` asks for JSON mode,
  extracts JSON from messy model output (code fences, prose, trailing commas),
  validates required keys and re-prompts the model to repair malformed output.
* Observable: every call is timed so the dashboard can show Nebius latency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # The OpenAI SDK is only needed for live calls.
    from openai import OpenAI

    _OPENAI_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover - depends on the environment
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _exc

logger = logging.getLogger("omega.nebius")

DEFAULT_BASE_URL = "https://api.studio.nebius.ai/v1/"
DEFAULT_MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

JSON_ONLY_SUFFIX = (
    "\n\nRespond with a single valid JSON value and nothing else: no prose, "
    "no markdown, no code fences."
)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class NebiusError(RuntimeError):
    """Base error for every failure raised by this module."""


class NebiusUnavailableError(NebiusError):
    """Raised when no API key / SDK is available (offline mode)."""


class NebiusJSONError(NebiusError):
    """Raised when model output cannot be parsed or validated as JSON."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------- #
# JSON extraction helpers
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SMART_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}


def _balanced_blocks(text: str, max_blocks: int = 5, max_starts: int = 25) -> List[str]:
    """Return up to `max_blocks` balanced {...} / [...] substrings (string aware)."""
    blocks: List[str] = []
    starts = 0
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        starts += 1
        if starts > max_starts:
            break
        depth = 0
        in_str = False
        escaped = False
        for j in range(start, len(text)):
            c = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    blocks.append(text[start : j + 1])
                    break
        if len(blocks) >= max_blocks:
            break
    return blocks


def extract_json(text: Optional[str]) -> Any:
    """
    Parse JSON out of raw model output.

    Handles: plain JSON, ```json fences, prose before/after the payload,
    trailing commas and typographic quotes. Raises NebiusJSONError on failure.
    """
    if text is None or not str(text).strip():
        raise NebiusJSONError("Model returned an empty response.", str(text or ""))

    raw = str(text).strip()
    candidates: List[str] = [raw]
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(raw))
    for cand in list(candidates):
        candidates.extend(_balanced_blocks(cand))

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        smart = cand
        for bad, good in _SMART_QUOTES.items():
            smart = smart.replace(bad, good)
        for variant in (cand, _TRAILING_COMMA_RE.sub(r"\1", cand), _TRAILING_COMMA_RE.sub(r"\1", smart)):
            try:
                return json.loads(variant)
            except (json.JSONDecodeError, ValueError):
                continue

    raise NebiusJSONError("Could not locate valid JSON in model output.", raw)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class NebiusClient:
    """OpenAI-SDK wrapper pointed at Nebius Studio."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 45.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = (api_key or os.environ.get("NEBIUS_API_KEY") or "").strip()
        self.model = model or os.environ.get("NEBIUS_MODEL") or DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))

        self._client: Any = None
        self._json_mode_supported = True
        self._lock = threading.Lock()
        self._calls = 0
        self._failures = 0
        self._latencies_ms: List[float] = []
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self.unavailable_reason = ""

        if not self.api_key:
            self.unavailable_reason = "NEBIUS_API_KEY is not set."
        elif OpenAI is None:
            self.unavailable_reason = f"openai package not installed ({_OPENAI_IMPORT_ERROR})."
        else:
            # SDK retries are disabled: this class owns the retry/backoff policy.
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self._client is not None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            lat = list(self._latencies_ms)
            return {
                "model": self.model,
                "base_url": self.base_url,
                "available": self.available,
                "calls": self._calls,
                "failures": self._failures,
                "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
                "last_latency_ms": round(lat[-1], 1) if lat else 0.0,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
            }

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """Plain chat completion returning the assistant text."""
        if not self.available:
            raise NebiusUnavailableError(self.unavailable_reason or "Nebius client unavailable.")

        kwargs: Dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and self._json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
                elapsed = (time.perf_counter() - started) * 1000.0
                content = (response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                with self._lock:
                    self._calls += 1
                    self._latencies_ms.append(elapsed)
                    if usage is not None:
                        self._prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                        self._completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                return content
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                last_exc = exc
                status = getattr(exc, "status_code", None)
                # Endpoint rejected JSON mode: remember it and retry once without it.
                if "response_format" in kwargs and (status == 400 or "response_format" in str(exc).lower()):
                    self._json_mode_supported = False
                    kwargs.pop("response_format", None)
                    logger.info("Nebius JSON mode unsupported; retrying without response_format.")
                    continue
                with self._lock:
                    self._failures += 1
                # 4xx other than 429 will not succeed on retry.
                if status is not None and 400 <= int(status) < 500 and int(status) != 429:
                    break
                attempt += 1
                if attempt <= self.max_retries:
                    time.sleep(0.6 * (2 ** (attempt - 1)))

        raise NebiusError(f"Nebius chat completion failed: {last_exc}") from last_exc

    # ------------------------------------------------------------------ #
    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        required_keys: Optional[Iterable[str]] = None,
        retries: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> Any:
        """
        Ask the model for JSON and return the parsed value.

        If the output is not valid JSON (or lacks `required_keys`) the model is
        shown its own answer and asked to repair it, up to `retries` times.
        """
        keys = list(required_keys or [])
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt.rstrip() + JSON_ONLY_SUFFIX},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Optional[NebiusJSONError] = None

        for attempt in range(max(0, retries) + 1):
            text = self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
            try:
                data = extract_json(text)
                self._check_keys(data, keys, text)
                return data
            except NebiusJSONError as exc:
                last_error = exc
                logger.warning("Nebius JSON parse failed (attempt %d): %s", attempt + 1, exc)
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That was not usable: {exc}. Return ONLY corrected JSON"
                            + (f" containing the keys {keys}." if keys else ".")
                        ),
                    }
                )

        raise last_error or NebiusJSONError("Structured completion failed.")

    def try_structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        default: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Like structured_completion() but returns `default` instead of raising."""
        try:
            return self.structured_completion(system_prompt, user_prompt, **kwargs)
        except NebiusError as exc:
            logger.info("Nebius structured completion fell back to default: %s", exc)
            return default

    @staticmethod
    def _check_keys(data: Any, keys: List[str], raw: str) -> None:
        if not keys:
            return
        if not isinstance(data, dict):
            raise NebiusJSONError("Expected a JSON object.", raw)
        missing = [k for k in keys if k not in data]
        if missing:
            raise NebiusJSONError(f"Missing required keys: {missing}", raw)

    # ------------------------------------------------------------------ #
    def health_check(self) -> Dict[str, Any]:
        """Cheap round-trip used by the dashboard status chip."""
        if not self.available:
            return {"ok": False, "reason": self.unavailable_reason}
        started = time.perf_counter()
        try:
            self.chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=5, temperature=0)
            return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000.0, 1)}
        except NebiusError as exc:
            return {"ok": False, "reason": str(exc)}


__all__ = [
    "NebiusClient",
    "NebiusError",
    "NebiusUnavailableError",
    "NebiusJSONError",
    "extract_json",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
]
