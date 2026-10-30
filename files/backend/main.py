"""
OMEGA x PHANTOM  -  Autonomous Mobility Safety & Fleet Intelligence Backend
===========================================================================

OMEGA   = decision / control brain (AI actions, guardrails, fleet optimisation)
PHANTOM = perception / digital-twin layer (detections, twin-vs-reality gap)

Run
---
    pip install fastapi "uvicorn[standard]" pydantic
    pip install supabase            # OPTIONAL - only if you want persistence
    uvicorn main:app --reload --port 8000

Environment variables (all optional)
------------------------------------
    SUPABASE_URL / SUPABASE_SERVICE_KEY (or SUPABASE_KEY)   enable Supabase mirror
    CORS_ORIGINS            comma separated list, default "*"
    HUMAN_APPROVER_TOKEN    if set, /approve and /reject require header X-Approver-Token
    APPROVAL_TTL_S          seconds before a pending approval expires (default 120)
    OMEGA_HARD_SPEED_CAP_MPS  absolute speed ceiling for any AI action (default 38.9)
    OMEGA_SEED_DEMO         "0" disables demo vehicle seeding (default "1")

Storage model
-------------
The in-memory store is ALWAYS the working copy, so the API works with zero
configuration (demo mode). When Supabase credentials are present and the schema
below exists, every write is mirrored to Supabase and data is hydrated at start-up.
If Supabase is unreachable or misconfigured, the backend silently stays in (or falls
back to) in-memory mode - the API never breaks.

Supabase schema (run once in the SQL editor; one table per collection):

    create table if not exists omega_vehicles       (id text primary key, data jsonb not null, created_at timestamptz default now());
    create table if not exists omega_detections     (id text primary key, data jsonb not null, created_at timestamptz default now());
    create table if not exists omega_ai_actions     (id text primary key, data jsonb not null, created_at timestamptz default now());
    create table if not exists omega_approvals      (id text primary key, data jsonb not null, created_at timestamptz default now());
    create table if not exists omega_audit_log      (id text primary key, data jsonb not null, created_at timestamptz default now());
    create table if not exists omega_scenario_runs  (id text primary key, data jsonb not null, created_at timestamptz default now());
    create table if not exists omega_fleet_plans    (id text primary key, data jsonb not null, created_at timestamptz default now());
    create table if not exists omega_twin_readings  (id text primary key, data jsonb not null, created_at timestamptz default now());
"""

import hmac
import logging
import math
import os
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================================================================
# Configuration & constants
# =============================================================================

APP_NAME = "OMEGA x PHANTOM"
APP_VERSION = "2.0.0"
START_TIME = time.time()

G = 9.80665  # m/s^2

HARD_SPEED_CAP_MPS = float(os.getenv("OMEGA_HARD_SPEED_CAP_MPS", "38.9"))  # ~140 km/h
SYSTEM_REACTION_S = 0.5          # conservative AI actuation latency used in safety maths
MIN_STANDSTILL_GAP_M = 2.0       # bumper-to-bumper gap that must remain after stopping
DETECTION_TTL_S = 10.0           # detections older than this are ignored by guardrails
APPROVAL_TTL_S = int(os.getenv("APPROVAL_TTL_S", "120"))
MAX_ROWS_PER_TABLE = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("omega-phantom")


# =============================================================================
# Small helpers
# =============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rnd(value: Any, digits: int = 3) -> Any:
    """Round numbers, pass None / non-numbers through unchanged."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return round(value, digits)
    return value


def json_safe(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable (NaN/inf -> None)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return obj


def dump(model: BaseModel) -> Dict[str, Any]:
    """Pydantic v1/v2 compatible model -> dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # pragma: no cover - pydantic v1


def percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


# =============================================================================
# Storage: in-memory working copy + optional Supabase mirror with auto-fallback
# =============================================================================

class Store:
    TABLES = (
        "vehicles", "detections", "ai_actions", "approvals",
        "audit_log", "scenario_runs", "fleet_plans", "twin_readings",
    )
    MAX_FAILS = 3

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.mem: Dict[str, Dict[str, dict]] = {t: {} for t in self.TABLES}
        self.client: Any = None
        self.mode = "memory"
        self.fail_count = 0
        self.last_error: Optional[str] = None
        self._connect()

    @staticmethod
    def _tn(table: str) -> str:
        return f"omega_{table}"

    def _connect(self) -> None:
        url = os.getenv("SUPABASE_URL", "").strip()
        key = (
            os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
            or ""
        ).strip()
        if not url or not key:
            log.info("Supabase not configured -> running in IN-MEMORY DEMO mode")
            return
        try:
            from supabase import create_client  # optional dependency

            client = create_client(url, key)
            hydrated: Dict[str, Dict[str, dict]] = {}
            for table in self.TABLES:
                res = client.table(self._tn(table)).select("id,data").limit(1000).execute()
                hydrated[table] = {row["id"]: row["data"] for row in (res.data or [])}
            self.mem = hydrated
            self.client = client
            self.mode = "supabase"
            log.info("Supabase connected -> persistence enabled (in-memory cache hydrated)")
        except Exception as exc:  # ImportError, network, missing tables, bad key...
            self.client = None
            self.mode = "memory"
            self.last_error = str(exc)[:300]
            log.warning("Supabase unavailable (%s) -> falling back to IN-MEMORY mode", self.last_error)

    def _mirror(self, fn) -> None:
        if self.client is None:
            return
        try:
            fn(self.client)
            self.fail_count = 0
        except Exception as exc:
            self.fail_count += 1
            self.last_error = str(exc)[:300]
            log.warning("Supabase write failed (%d/%d): %s", self.fail_count, self.MAX_FAILS, self.last_error)
            if self.fail_count >= self.MAX_FAILS:
                self.client = None
                self.mode = "memory (supabase fallback)"
                log.error("Supabase disabled after repeated failures -> in-memory fallback active")

    def put(self, table: str, key: str, data: dict) -> dict:
        data = json_safe(data)
        with self.lock:
            tbl = self.mem[table]
            tbl[key] = data
            while len(tbl) > MAX_ROWS_PER_TABLE:
                tbl.pop(next(iter(tbl)))
        self._mirror(lambda c: c.table(self._tn(table)).upsert({"id": key, "data": data}).execute())
        return data

    def get(self, table: str, key: str) -> Optional[dict]:
        with self.lock:
            return self.mem[table].get(key)

    def values(self, table: str) -> List[dict]:
        with self.lock:
            return list(self.mem[table].values())

    def delete(self, table: str, key: str) -> bool:
        with self.lock:
            existed = self.mem[table].pop(key, None) is not None
        self._mirror(lambda c: c.table(self._tn(table)).delete().eq("id", key).execute())
        return existed

    def counts(self) -> Dict[str, int]:
        with self.lock:
            return {t: len(rows) for t, rows in self.mem.items()}


store = Store()


def audit(event: str, severity: str = "info", **details: Any) -> None:
    entry_id = new_id("aud")
    store.put("audit_log", entry_id, {
        "audit_id": entry_id,
        "event": event,
        "severity": severity,
        "details": details,
        "created_at": now_iso(),
    })


def newest_first(rows: List[dict], key: str = "created_at") -> List[dict]:
    return sorted(rows, key=lambda r_: r_.get(key, ""), reverse=True)


# =============================================================================
# Pydantic models
# =============================================================================

class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class GeoPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class VehicleCreate(BaseModel):
    vehicle_id: Optional[str] = Field(None, max_length=64)
    name: str = Field(..., min_length=1, max_length=100)
    vehicle_type: str = Field("van", max_length=32)
    mass_kg: float = Field(2200.0, gt=0, le=60000)
    battery_capacity_kwh: float = Field(75.0, gt=0, le=1000)
    battery_pct: float = Field(100.0, ge=0, le=100)
    max_speed_mps: float = Field(33.0, gt=0, le=70)
    speed_mps: float = Field(0.0, ge=0, le=100)
    heading_deg: float = 0.0
    position: Position = Field(default_factory=Position)
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    payload_capacity_kg: float = Field(1000.0, ge=0, le=100000)
    base_efficiency_kwh_per_km: float = Field(0.2, gt=0, le=5)
    road_friction: float = Field(0.8, ge=0.05, le=1.2)
    grade_pct: float = Field(0.0, ge=-30, le=30)
    ambient_temp_c: float = Field(20.0, ge=-50, le=60)


class TelemetryUpdate(BaseModel):
    speed_mps: Optional[float] = Field(None, ge=0, le=100)
    heading_deg: Optional[float] = None
    position: Optional[Position] = None
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    battery_pct: Optional[float] = Field(None, ge=0, le=100)
    road_friction: Optional[float] = Field(None, ge=0.05, le=1.2)
    grade_pct: Optional[float] = Field(None, ge=-30, le=30)
    ambient_temp_c: Optional[float] = Field(None, ge=-50, le=60)
    distance_km_delta: Optional[float] = Field(None, ge=0)
    energy_used_kwh_delta: Optional[float] = Field(None, ge=0)


class DetectionIn(BaseModel):
    vehicle_id: str
    object_type: str = Field("vehicle", max_length=32)
    object_id: Optional[str] = Field(None, max_length=64)
    distance_m: float = Field(..., ge=0, le=2000)
    # POSITIVE = object and ego are closing; NEGATIVE = opening
    relative_speed_mps: float = Field(0.0, ge=-100, le=100)
    bearing_deg: float = 0.0
    confidence: float = Field(0.9, ge=0, le=1)
    in_path: bool = True


class TTCRequest(BaseModel):
    distance_m: float = Field(..., ge=0, le=5000, description="Bumper-to-bumper gap")
    ego_speed_mps: float = Field(..., ge=0, le=100)
    target_speed_mps: float = Field(0.0, ge=0, le=100, description="Lead object speed")
    ego_accel_mps2: float = Field(0.0, ge=-20, le=20)
    target_accel_mps2: float = Field(0.0, ge=-20, le=20)


class StoppingDistanceRequest(BaseModel):
    speed_mps: float = Field(..., ge=0, le=100)
    reaction_time_s: float = Field(1.0, ge=0, le=5)
    friction_coefficient: float = Field(0.7, ge=0.05, le=1.2)
    grade_pct: float = Field(0.0, ge=-30, le=30, description="Positive = uphill")
    brake_efficiency: float = Field(1.0, ge=0.1, le=1.0)
    available_gap_m: Optional[float] = Field(None, ge=0, le=5000)


class SafetyAssessRequest(BaseModel):
    vehicle_id: Optional[str] = None
    distance_m: float = Field(..., ge=0, le=5000)
    ego_speed_mps: Optional[float] = Field(None, ge=0, le=100)
    target_speed_mps: float = Field(0.0, ge=0, le=100)
    reaction_time_s: float = Field(SYSTEM_REACTION_S, ge=0, le=5)
    friction_coefficient: Optional[float] = Field(None, ge=0.05, le=1.2)
    grade_pct: Optional[float] = Field(None, ge=-30, le=30)


class AIActionRequest(BaseModel):
    vehicle_id: str
    action_type: str = Field(..., min_length=1, max_length=64)
    target_speed_mps: Optional[float] = Field(None, ge=0, le=200)
    target_accel_mps2: Optional[float] = Field(None, ge=-100, le=100)
    lane_offset: Optional[int] = Field(None, ge=-3, le=3)
    lead_distance_m: Optional[float] = Field(None, ge=0, le=5000)
    lead_speed_mps: Optional[float] = Field(None, ge=0, le=100)
    confidence: float = Field(0.5, ge=0, le=1)
    source: str = Field("OMEGA", max_length=64)
    rationale: Optional[str] = Field(None, max_length=1000)
    params: Dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approval_id: str
    decided_by: str = Field(..., min_length=2, max_length=100)
    note: Optional[str] = Field(None, max_length=500)


class DecideRequest(BaseModel):
    vehicle_id: str
    cruise_speed_mps: Optional[float] = Field(None, gt=0, le=100)
    submit: bool = True


class ScenarioRunRequest(BaseModel):
    scenario_type: str = "lead_vehicle_brake"
    runs: int = Field(100, ge=1, le=1000)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    ego_speed_mps: Optional[float] = Field(None, gt=1, le=60)
    initial_gap_m: Optional[float] = Field(None, gt=1, le=500)
    friction: Optional[float] = Field(None, ge=0.1, le=1.2)
    jitter_pct: float = Field(15.0, ge=0, le=60)
    sensor_fault_prob: Optional[float] = Field(None, ge=0, le=1)
    include_details: bool = False


class FleetVehicleIn(BaseModel):
    vehicle_id: str = Field(..., min_length=1, max_length=64)
    start: GeoPoint
    battery_pct: float = Field(100.0, ge=0, le=100)
    battery_capacity_kwh: float = Field(75.0, gt=0, le=1000)
    payload_capacity_kg: float = Field(1000.0, ge=0, le=100000)
    curb_mass_kg: float = Field(2200.0, gt=0, le=60000)
    base_efficiency_kwh_per_km: float = Field(0.2, gt=0, le=5)
    avg_speed_kph: float = Field(40.0, ge=5, le=130)


class FleetStopIn(BaseModel):
    stop_id: str = Field(..., min_length=1, max_length=64)
    location: GeoPoint
    demand_kg: float = Field(0.0, ge=0, le=100000)
    service_min: float = Field(5.0, ge=0, le=600)
    priority: int = Field(3, ge=1, le=5)
    window_start_min: Optional[float] = Field(None, ge=0)
    window_end_min: Optional[float] = Field(None, ge=0)


class ChargingStationIn(BaseModel):
    station_id: str = Field(..., min_length=1, max_length=64)
    location: GeoPoint
    power_kw: float = Field(50.0, gt=0, le=1000)


class FleetOptimizeRequest(BaseModel):
    vehicles: Optional[List[FleetVehicleIn]] = None  # None -> use registered vehicles with lat/lon
    stops: List[FleetStopIn]
    charging_stations: List[ChargingStationIn] = Field(default_factory=list)
    objective: str = Field("balanced", description="energy | time | balanced")
    reserve_pct: float = Field(10.0, ge=0, le=50)
    charge_target_pct: float = Field(90.0, ge=20, le=100)
    ambient_temp_c: float = Field(20.0, ge=-50, le=60)
    traffic_factor: float = Field(1.0, ge=0.5, le=3.0)
    return_to_start: bool = True
    use_two_opt: bool = True
    time_limit_s: float = Field(3.0, ge=0.1, le=20)
    electricity_price_per_kwh: float = Field(0.15, ge=0)
    grid_co2_kg_per_kwh: float = Field(0.4, ge=0)


class EnergyEstimateRequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=5000)
    avg_speed_kph: float = Field(40.0, ge=5, le=160)
    payload_kg: float = Field(0.0, ge=0, le=100000)
    curb_mass_kg: float = Field(2200.0, gt=0, le=60000)
    base_efficiency_kwh_per_km: float = Field(0.2, gt=0, le=5)
    ambient_temp_c: float = Field(20.0, ge=-50, le=60)
    grade_pct: float = Field(0.0, ge=-30, le=30)
    traffic_factor: float = Field(1.0, ge=0.5, le=3.0)
    battery_capacity_kwh: float = Field(75.0, gt=0, le=1000)
    battery_pct: float = Field(100.0, ge=0, le=100)
    reserve_pct: float = Field(10.0, ge=0, le=50)
    electricity_price_per_kwh: float = Field(0.15, ge=0)
    grid_co2_kg_per_kwh: float = Field(0.4, ge=0)


class TwinState(BaseModel):
    position: Position = Field(default_factory=Position)
    speed_mps: float = Field(0.0, ge=0, le=100)
    heading_deg: float = 0.0
    battery_pct: float = Field(100.0, ge=0, le=100)
    temp_c: Optional[float] = Field(None, ge=-50, le=300)


class TwinReadingIn(BaseModel):
    vehicle_id: str
    twin: TwinState
    real: TwinState
    timestamp: Optional[str] = None


# =============================================================================
# 1. Physics: Time-to-Collision & Stopping Distance
# =============================================================================

def compute_ttc(distance_m: float, ego_speed: float, target_speed: float = 0.0,
                ego_accel: float = 0.0, target_accel: float = 0.0) -> Optional[float]:
    """
    Time to collision under a constant-acceleration model:
        distance = v_rel * t + 0.5 * a_rel * t^2
    where v_rel = ego - target (positive = closing) and a_rel = ego_a - target_a.
    Returns the smallest positive root, 0.0 if already touching, or None if
    the two objects never meet. (Model is valid until either body stops.)
    """
    if distance_m <= 0:
        return 0.0
    v = ego_speed - target_speed
    a = ego_accel - target_accel
    if abs(a) < 1e-9:
        return distance_m / v if v > 1e-9 else None
    disc = v * v + 2.0 * a * distance_m
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    roots = [(-v + sq) / a, (-v - sq) / a]
    positive = [t for t in roots if t > 1e-9]
    return min(positive) if positive else None


def ttc_risk(ttc: Optional[float]) -> str:
    if ttc is None:
        return "none"
    if ttc < 1.5:
        return "critical"
    if ttc < 3.0:
        return "high"
    if ttc < 5.0:
        return "medium"
    return "low"


def stopping_distance(speed: float, reaction_time: float = 1.0, friction: float = 0.7,
                      grade_pct: float = 0.0, brake_efficiency: float = 1.0) -> Dict[str, float]:
    """
    Total stopping distance = reaction distance + braking distance.
    Deceleration = g * (mu * eff * cos(theta) + sin(theta)); uphill grade helps braking.
    """
    theta = math.atan(grade_pct / 100.0)
    decel = G * (friction * brake_efficiency * math.cos(theta) + math.sin(theta))
    decel = max(decel, 0.5)  # physical floor to avoid division blow-ups on ice/downhill
    reaction_d = speed * reaction_time
    braking_d = speed * speed / (2.0 * decel)
    return {
        "reaction_distance_m": reaction_d,
        "braking_distance_m": braking_d,
        "total_stopping_distance_m": reaction_d + braking_d,
        "deceleration_mps2": decel,
        "time_to_stop_s": reaction_time + speed / decel,
    }


def format_stopping(res: Dict[str, float], speed: float) -> Dict[str, Any]:
    out = {k: rnd(v) for k, v in res.items()}
    out["speed_mps"] = rnd(speed)
    out["speed_kph"] = rnd(speed * 3.6, 1)
    return out


# =============================================================================
# Vehicles, telemetry, PHANTOM detections
# =============================================================================

def get_vehicle_or_404(vehicle_id: str) -> dict:
    v = store.get("vehicles", vehicle_id)
    if v is None:
        raise HTTPException(status_code=404, detail=f"Vehicle '{vehicle_id}' not found")
    return v


def make_vehicle(data: dict) -> dict:
    vid = data.get("vehicle_id") or new_id("veh")
    ts = now_iso()
    return {
        "vehicle_id": vid,
        "name": data["name"],
        "vehicle_type": data.get("vehicle_type", "van"),
        "mass_kg": data.get("mass_kg", 2200.0),
        "battery_capacity_kwh": data.get("battery_capacity_kwh", 75.0),
        "battery_pct": data.get("battery_pct", 100.0),
        "max_speed_mps": data.get("max_speed_mps", 33.0),
        "speed_mps": data.get("speed_mps", 0.0),
        "heading_deg": data.get("heading_deg", 0.0),
        "position": data.get("position") or {"x": 0.0, "y": 0.0},
        "lat": data.get("lat"),
        "lon": data.get("lon"),
        "payload_capacity_kg": data.get("payload_capacity_kg", 1000.0),
        "base_efficiency_kwh_per_km": data.get("base_efficiency_kwh_per_km", 0.2),
        "road_friction": data.get("road_friction", 0.8),
        "grade_pct": data.get("grade_pct", 0.0),
        "ambient_temp_c": data.get("ambient_temp_c", 20.0),
        "odometer_km": 0.0,
        "energy_used_kwh_total": 0.0,
        "commanded_speed_mps": None,
        "lane_offset": 0,
        "status": "ACTIVE",
        "last_action": None,
        "threat_level": "none",
        "created_at": ts,
        "updated_at": ts,
    }


def vehicle_safety_snapshot(v: dict) -> Dict[str, Any]:
    sd = stopping_distance(v["speed_mps"], SYSTEM_REACTION_S, v.get("road_friction", 0.8), v.get("grade_pct", 0.0))
    return {"stopping_distance": format_stopping(sd, v["speed_mps"])}


def recent_detections(vehicle_id: str) -> List[dict]:
    cutoff = time.time() - DETECTION_TTL_S
    return [d for d in store.values("detections")
            if d.get("vehicle_id") == vehicle_id and d.get("ts_epoch", 0) >= cutoff]


def build_threat_context(vehicle: dict, lead_distance: Optional[float] = None,
                         lead_speed: Optional[float] = None) -> Dict[str, Any]:
    """Merge PHANTOM detections and an optional caller-supplied lead object."""
    ego = float(vehicle.get("speed_mps", 0.0))
    threats: List[Dict[str, Any]] = []
    for d in recent_detections(vehicle["vehicle_id"]):
        if not d.get("in_path", True):
            continue
        threats.append({
            "source": "phantom",
            "object_type": d.get("object_type"),
            "distance_m": d["distance_m"],
            "closing_mps": d["relative_speed_mps"],
            "confidence": d.get("confidence", 1.0),
        })
    if lead_distance is not None:
        ls = lead_speed if lead_speed is not None else 0.0  # conservative: assume stationary
        threats.append({
            "source": "request",
            "object_type": "lead",
            "distance_m": lead_distance,
            "closing_mps": ego - ls,
            "confidence": 1.0,
        })
    min_ttc: Optional[float] = None
    for t in threats:
        t["ttc_s"] = compute_ttc(t["distance_m"], max(t["closing_mps"], 0.0), 0.0) if t["closing_mps"] > 0 else None
        if t["ttc_s"] is not None and (min_ttc is None or t["ttc_s"] < min_ttc):
            min_ttc = t["ttc_s"]
    nearest = min(threats, key=lambda t: t["distance_m"]) if threats else None
    return {"threat_count": len(threats), "threats": threats, "min_ttc": min_ttc, "nearest": nearest}


# =============================================================================
# 6 (support). Twin status lookup used by guardrails
# =============================================================================

def twin_status_for(vehicle_id: str) -> Optional[str]:
    latest = store.get("twin_readings", f"latest:{vehicle_id}")
    return latest.get("status") if latest else None


# =============================================================================
# 3. Backend-enforced safety guardrails
# =============================================================================

KNOWN_ACTIONS = {
    "accelerate", "brake", "maintain", "set_speed", "lane_change", "overtake",
    "emergency_stop", "route_change", "park", "charge",
}
ALWAYS_SAFE_ACTIONS = {"brake", "emergency_stop", "park"}
LANE_ACTIONS = {"lane_change", "overtake"}
FORBIDDEN_ACTIONS = {
    "disable_abs", "disable_safety", "disable_collision_avoidance", "disable_emergency_brake",
    "disable_guardrails", "disable_driver_monitoring", "override_geofence", "override_speed_limit",
    "disable_esc", "disable_airbags", "bypass_safety",
}
TAMPER_KEYS = {
    "bypass_guardrails", "disable_safety", "skip_safety_checks", "override_safety",
    "ignore_ttc", "disable_collision_avoidance", "force", "bypass_approval",
}

GUARDRAIL_CATALOG = [
    {"rule": "FORBIDDEN_ACTION", "severity": "block", "description": "Actions that disable safety systems are never executable."},
    {"rule": "GUARDRAIL_TAMPERING", "severity": "block", "description": "Parameters that try to bypass guardrails are rejected."},
    {"rule": "SPEED_HARD_CAP", "severity": "block", "description": "Projected speed may not exceed min(vehicle max speed, global hard cap)."},
    {"rule": "TTC_CRITICAL", "severity": "block", "description": "Non-braking actions blocked when time-to-collision is below 2.5 s (3.5 s for lane changes)."},
    {"rule": "TTC_LOW", "severity": "approval", "description": "Speed-increasing or lane actions with TTC below 5 s (7 s lane) need human approval."},
    {"rule": "STOPPING_DISTANCE_INSUFFICIENT", "severity": "block", "description": "RSS-style check: gap + lead stopping distance must cover ego stopping distance."},
    {"rule": "LOW_FRICTION_SPEED", "severity": "block", "description": "Speed increases beyond the friction-limited safe speed are blocked."},
    {"rule": "LOW_FRICTION_MANEUVER", "severity": "approval", "description": "Lane changes on low-friction surfaces need human approval."},
    {"rule": "BATTERY_CRITICAL", "severity": "block", "description": "Below 5% battery only braking, parking and charging are allowed."},
    {"rule": "BATTERY_LOW", "severity": "approval", "description": "Below 10% battery, energy-intensive actions need approval."},
    {"rule": "ACCEL_IMPLAUSIBLE", "severity": "block", "description": "Acceleration above 6 m/s^2 is rejected; above 3.5 m/s^2 needs approval."},
    {"rule": "LOW_CONFIDENCE", "severity": "approval", "description": "AI confidence below 0.4 needs approval for risky actions; below 0.15 is blocked."},
    {"rule": "HIGH_SPEED_LANE_CHANGE", "severity": "approval", "description": "Lane changes above 25 m/s need approval."},
    {"rule": "TWIN_DIVERGED", "severity": "approval", "description": "If the digital twin has diverged from reality, non-braking actions need approval."},
    {"rule": "PARK_WHILE_MOVING", "severity": "block", "description": "Park is rejected above 3 m/s; use brake / emergency_stop."},
    {"rule": "UNKNOWN_ACTION", "severity": "approval", "description": "Unrecognised actions always need human approval."},
]


def evaluate_guardrails(vehicle: dict, act: dict) -> Dict[str, Any]:
    """Pure evaluation (no side effects). Returns decision + violations + metrics."""
    violations: List[Dict[str, Any]] = []

    def add(rule: str, severity: str, message: str, value: Any = None, limit: Any = None) -> None:
        violations.append({"rule": rule, "severity": severity, "message": message,
                           "value": rnd(value), "limit": rnd(limit)})

    atype = str(act.get("action_type", "")).strip().lower()
    speed = float(vehicle.get("speed_mps", 0.0))
    mu = float(vehicle.get("road_friction", 0.8))
    grade = float(vehicle.get("grade_pct", 0.0))
    battery = float(vehicle.get("battery_pct", 100.0))
    max_speed = min(float(vehicle.get("max_speed_mps", HARD_SPEED_CAP_MPS)), HARD_SPEED_CAP_MPS)
    tgt = act.get("target_speed_mps")
    acc = act.get("target_accel_mps2")
    conf = float(act.get("confidence", 0.5))
    params = act.get("params") or {}

    lane_type = atype in LANE_ACTIONS
    always_safe = atype in ALWAYS_SAFE_ACTIONS
    is_decel = atype in ("brake", "emergency_stop") or (
        atype == "set_speed" and tgt is not None and tgt < speed - 0.5)
    is_increase = (atype in ("accelerate", "overtake")
                   or (tgt is not None and tgt > speed + 0.5)
                   or (acc is not None and acc > 0.2))

    if tgt is not None:
        projected = float(tgt)
    elif atype in ("accelerate", "overtake"):
        projected = speed + max(acc if acc is not None else 1.5, 0.5) * 2.0
    else:
        projected = speed
    if is_decel:
        projected = min(projected, speed)

    # --- Hard prohibitions -------------------------------------------------
    if atype in FORBIDDEN_ACTIONS:
        add("FORBIDDEN_ACTION", "block", f"Action '{atype}' disables a safety system and can never be executed.")
    elif atype not in KNOWN_ACTIONS:
        add("UNKNOWN_ACTION", "approval", f"Action '{atype}' is not a recognised OMEGA action.")

    tampered = [k for k, val in params.items() if str(k).lower() in TAMPER_KEYS and bool(val)]
    if tampered:
        add("GUARDRAIL_TAMPERING", "block", f"Parameters attempt to bypass safety checks: {sorted(tampered)}.")

    if atype in ("brake", "emergency_stop") and (
            (tgt is not None and tgt > speed + 0.5) or (acc is not None and acc > 0.2)):
        add("INCONSISTENT_BRAKE", "block", "A braking action may not request higher speed or positive acceleration.")

    if atype == "park" and speed > 3.0:
        add("PARK_WHILE_MOVING", "block", "Park requested while moving; use brake or emergency_stop first.", speed, 3.0)

    if acc is not None:
        if acc < -20.0:
            add("ACCEL_IMPLAUSIBLE", "block", "Requested deceleration is physically implausible.", acc, -20.0)
        elif acc < -(mu * G * 1.1):
            add("DECEL_CLAMPED", "warn",
                "Requested deceleration exceeds tyre-road friction; it will be clamped to the physical limit.",
                acc, -(mu * G))
        if acc > 6.0:
            add("ACCEL_IMPLAUSIBLE", "block", "Requested acceleration exceeds 6 m/s^2.", acc, 6.0)
        elif acc > 3.5 and not always_safe:
            add("ACCEL_HIGH", "approval", "Requested acceleration exceeds comfort/safety envelope (3.5 m/s^2).", acc, 3.5)

    # --- Situational metrics ----------------------------------------------
    ctx = build_threat_context(vehicle, act.get("lead_distance_m"), act.get("lead_speed_mps"))
    nearest = ctx["nearest"]
    ego_stop = stopping_distance(max(projected, 0.0), SYSTEM_REACTION_S, mu, grade)
    rss_margin: Optional[float] = None
    if nearest is not None:
        lead_speed = max(0.0, speed - nearest["closing_mps"])
        lead_stop = lead_speed ** 2 / (2.0 * max(2.0, mu * G))
        rss_margin = nearest["distance_m"] + lead_stop - ego_stop["total_stopping_distance_m"] - MIN_STANDSTILL_GAP_M
    twin_status = twin_status_for(vehicle["vehicle_id"])

    # --- Rules for any action that is neither braking nor slowing ----------
    if not always_safe and not is_decel:
        # Speed cap
        if projected > max_speed + 1e-6:
            add("SPEED_HARD_CAP", "block",
                f"Projected speed {projected:.1f} m/s exceeds the allowed maximum {max_speed:.1f} m/s.",
                projected, max_speed)

        # Collision risk
        ttc = ctx["min_ttc"]
        if ttc is not None:
            block_thr, appr_thr = (3.5, 7.0) if lane_type else (2.5, 5.0)
            if ttc < block_thr:
                add("TTC_CRITICAL", "block",
                    f"Time-to-collision {ttc:.2f}s is below {block_thr:.1f}s; only braking actions are permitted.",
                    ttc, block_thr)
            elif (is_increase or lane_type) and ttc < appr_thr:
                add("TTC_LOW", "approval",
                    f"Time-to-collision {ttc:.2f}s is below {appr_thr:.1f}s for a risk-increasing action.",
                    ttc, appr_thr)
        if rss_margin is not None and rss_margin < 0:
            add("STOPPING_DISTANCE_INSUFFICIENT", "block",
                f"Stopping distance {ego_stop['total_stopping_distance_m']:.1f} m cannot be satisfied by the "
                f"available gap (shortfall {-rss_margin:.1f} m). Only braking is permitted.",
                rss_margin, 0.0)

        # Friction
        v_friction = math.sqrt(160.0 * mu * G)  # speed that stops within ~80 m (no reaction time)
        if is_increase and projected > v_friction:
            add("LOW_FRICTION_SPEED", "block",
                f"Projected speed {projected:.1f} m/s exceeds the friction-limited safe speed {v_friction:.1f} m/s.",
                projected, v_friction)
        elif lane_type and mu < 0.5:
            add("LOW_FRICTION_MANEUVER", "approval", f"Lane manoeuvre on a low-friction surface (mu={mu:.2f}).", mu, 0.5)

        # Battery
        if battery < 5.0 and atype not in ("charge", "park"):
            add("BATTERY_CRITICAL", "block", f"Battery at {battery:.1f}%; only braking, parking and charging allowed.",
                battery, 5.0)
        elif battery < 10.0 and atype in ("accelerate", "overtake", "route_change", "lane_change"):
            add("BATTERY_LOW", "approval", f"Battery at {battery:.1f}%; energy-intensive action needs approval.",
                battery, 10.0)

        # Lane-change specifics
        if lane_type and speed > 25.0:
            add("HIGH_SPEED_LANE_CHANGE", "approval", f"Lane change at {speed:.1f} m/s needs approval.", speed, 25.0)

        # AI confidence
        if conf < 0.15:
            add("LOW_CONFIDENCE", "block", f"AI confidence {conf:.2f} is too low to act.", conf, 0.15)
        elif conf < 0.4 and (is_increase or lane_type):
            add("LOW_CONFIDENCE", "approval", f"AI confidence {conf:.2f} below 0.40 for a risky action.", conf, 0.4)

        # Digital twin trust
        if twin_status == "DIVERGED":
            add("TWIN_DIVERGED", "approval",
                "Digital twin has diverged from reality; AI planning is untrusted without human approval.")

    if any(v["severity"] == "block" for v in violations):
        decision = "BLOCKED"
    elif any(v["severity"] == "approval" for v in violations):
        decision = "PENDING_APPROVAL"
    else:
        decision = "APPROVED"

    return {
        "decision": decision,
        "violations": violations,
        "metrics": {
            "speed_mps": rnd(speed),
            "projected_speed_mps": rnd(projected),
            "min_ttc_s": rnd(ctx["min_ttc"]),
            "ttc_risk": ttc_risk(ctx["min_ttc"]),
            "stopping_distance_m": rnd(ego_stop["total_stopping_distance_m"]),
            "rss_margin_m": rnd(rss_margin),
            "threat_count": ctx["threat_count"],
            "road_friction": rnd(mu),
            "battery_pct": rnd(battery),
            "twin_status": twin_status,
        },
    }


def apply_action(vehicle: dict, act: dict) -> Dict[str, Any]:
    """Execute an approved action by updating the vehicle's commanded state."""
    atype = str(act.get("action_type", "")).strip().lower()
    speed = float(vehicle.get("speed_mps", 0.0))
    max_speed = min(float(vehicle.get("max_speed_mps", HARD_SPEED_CAP_MPS)), HARD_SPEED_CAP_MPS)
    mu = float(vehicle.get("road_friction", 0.8))
    tgt = act.get("target_speed_mps")
    effect: Dict[str, Any] = {"action": atype}

    if atype == "emergency_stop":
        vehicle["commanded_speed_mps"] = 0.0
        vehicle["status"] = "EMERGENCY_STOP"
        effect["decel_mps2"] = rnd(mu * G * 0.95)
    elif atype == "brake":
        vehicle["commanded_speed_mps"] = float(tgt) if tgt is not None else max(0.0, speed - 5.0)
        vehicle["status"] = "BRAKING"
    elif atype == "accelerate":
        vehicle["commanded_speed_mps"] = clamp(float(tgt) if tgt is not None else speed + 3.0, 0.0, max_speed)
        vehicle["status"] = "ACTIVE"
    elif atype == "set_speed":
        vehicle["commanded_speed_mps"] = clamp(float(tgt if tgt is not None else speed), 0.0, max_speed)
        vehicle["status"] = "ACTIVE"
    elif atype == "maintain":
        vehicle["commanded_speed_mps"] = speed
        vehicle["status"] = "ACTIVE"
    elif atype in LANE_ACTIONS:
        delta = act.get("lane_offset")
        delta = 1 if delta in (None, 0) else int(delta)
        vehicle["lane_offset"] = int(vehicle.get("lane_offset", 0)) + delta
        vehicle["status"] = "ACTIVE"
        effect["lane_offset"] = vehicle["lane_offset"]
        if atype == "overtake" and tgt is not None:
            vehicle["commanded_speed_mps"] = clamp(float(tgt), 0.0, max_speed)
    elif atype == "route_change":
        vehicle["route"] = (act.get("params") or {}).get("route")
        vehicle["status"] = "ACTIVE"
    elif atype == "park":
        vehicle["commanded_speed_mps"] = 0.0
        vehicle["status"] = "PARKED"
    elif atype == "charge":
        vehicle["commanded_speed_mps"] = 0.0
        vehicle["status"] = "CHARGING"

    vehicle["last_action"] = {"action_type": atype, "at": now_iso()}
    vehicle["updated_at"] = now_iso()
    store.put("vehicles", vehicle["vehicle_id"], vehicle)
    effect["commanded_speed_mps"] = rnd(vehicle.get("commanded_speed_mps"))
    effect["vehicle_status"] = vehicle["status"]
    return effect


def process_action(vehicle: dict, act: dict) -> Dict[str, Any]:
    """Run an AI action through the guardrails and persist the outcome."""
    action_id = new_id("act")
    ev = evaluate_guardrails(vehicle, act)
    record: Dict[str, Any] = {
        "action_id": action_id,
        "vehicle_id": vehicle["vehicle_id"],
        "action": act,
        "decision": ev["decision"],
        "violations": ev["violations"],
        "metrics": ev["metrics"],
        "approval_id": None,
        "effect": None,
        "created_at": now_iso(),
    }
    if ev["decision"] == "BLOCKED":
        record["status"] = "blocked"
        audit("GUARDRAIL_BLOCK", "warning", action_id=action_id, vehicle_id=vehicle["vehicle_id"],
              action_type=act.get("action_type"), source=act.get("source"),
              rules=[v["rule"] for v in ev["violations"] if v["severity"] == "block"])
    elif ev["decision"] == "PENDING_APPROVAL":
        approval_id = new_id("apr")
        expires = time.time() + APPROVAL_TTL_S
        store.put("approvals", approval_id, {
            "approval_id": approval_id,
            "action_id": action_id,
            "vehicle_id": vehicle["vehicle_id"],
            "status": "pending",
            "reasons": [v for v in ev["violations"] if v["severity"] == "approval"],
            "action_summary": {"action_type": act.get("action_type"),
                               "target_speed_mps": act.get("target_speed_mps"),
                               "source": act.get("source"), "rationale": act.get("rationale")},
            "requested_at": now_iso(),
            "created_at": now_iso(),
            "expires_epoch": expires,
            "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
            "decided_by": None,
            "decided_at": None,
            "note": None,
        })
        record["approval_id"] = approval_id
        record["status"] = "pending_approval"
        audit("APPROVAL_REQUESTED", "info", action_id=action_id, approval_id=approval_id,
              vehicle_id=vehicle["vehicle_id"], action_type=act.get("action_type"))
    else:
        record["effect"] = apply_action(vehicle, act)
        record["status"] = "executed"
        audit("ACTION_EXECUTED", "info", action_id=action_id, vehicle_id=vehicle["vehicle_id"],
              action_type=act.get("action_type"), auto=True)
    store.put("ai_actions", action_id, record)
    return record


# --- Human approval helpers --------------------------------------------------

AI_PRINCIPAL_PREFIXES = ("ai", "omega", "phantom", "bot", "system", "auto", "agent", "claude", "gpt")


def expire_stale_approvals() -> None:
    now = time.time()
    for ap in store.values("approvals"):
        if ap.get("status") == "pending" and now > ap.get("expires_epoch", now + 1):
            ap["status"] = "expired"
            ap["decided_at"] = now_iso()
            store.put("approvals", ap["approval_id"], ap)
            act = store.get("ai_actions", ap["action_id"])
            if act and act.get("status") == "pending_approval":
                act["status"] = "expired"
                store.put("ai_actions", act["action_id"], act)
            audit("APPROVAL_EXPIRED", "info", approval_id=ap["approval_id"], action_id=ap["action_id"])


def load_pending_approval(approval_id: str) -> dict:
    expire_stale_approvals()
    ap = store.get("approvals", approval_id)
    if ap is None:
        raise HTTPException(status_code=404, detail=f"Approval '{approval_id}' not found")
    if ap["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"Approval is already '{ap['status']}' and cannot be changed")
    return ap


def assert_human_approver(decided_by: str, token: Optional[str], action_source: Optional[str]) -> None:
    """AI / automated principals may never approve their own (or any) actions."""
    who = decided_by.strip().lower()
    if who.startswith(AI_PRINCIPAL_PREFIXES) or (action_source and who == str(action_source).strip().lower()):
        audit("APPROVAL_SELF_APPROVE_DENIED", "critical", decided_by=decided_by, action_source=action_source)
        raise HTTPException(status_code=403, detail="Automated or AI principals cannot approve safety actions")
    required = os.getenv("HUMAN_APPROVER_TOKEN")
    if required and not (token and hmac.compare_digest(token, required)):
        audit("APPROVAL_BAD_TOKEN", "critical", decided_by=decided_by)
        raise HTTPException(status_code=401, detail="Missing or invalid X-Approver-Token")


# =============================================================================
# OMEGA decision engine (core autonomous logic)
# =============================================================================

def omega_decide(vehicle: dict, cruise_speed: Optional[float]) -> Dict[str, Any]:
    speed = float(vehicle["speed_mps"])
    mu = float(vehicle.get("road_friction", 0.8))
    grade = float(vehicle.get("grade_pct", 0.0))
    max_speed = min(float(vehicle["max_speed_mps"]), HARD_SPEED_CAP_MPS)
    cruise = clamp(cruise_speed if cruise_speed is not None else min(max_speed, 22.0), 1.0, max_speed)
    ctx = build_threat_context(vehicle)
    stop = stopping_distance(speed, SYSTEM_REACTION_S, mu, grade)
    nearest = ctx["nearest"]
    ttc = ctx["min_ttc"]
    conf = min([t["confidence"] for t in ctx["threats"]], default=0.95)

    action: Dict[str, Any] = {"vehicle_id": vehicle["vehicle_id"], "source": "OMEGA", "confidence": conf, "params": {}}
    if nearest is not None and nearest["closing_mps"] > 0 and (
            (ttc is not None and ttc < 1.5) or nearest["distance_m"] <= stop["total_stopping_distance_m"] * 0.8):
        action.update(action_type="emergency_stop", target_speed_mps=0.0,
                      rationale=f"Imminent collision: TTC={rnd(ttc)}s, gap={rnd(nearest['distance_m'])}m, "
                                f"stopping distance={rnd(stop['total_stopping_distance_m'])}m")
    elif ttc is not None and ttc < 3.5:
        req_decel = nearest["closing_mps"] ** 2 / (2.0 * max(nearest["distance_m"], 0.1))
        action.update(action_type="brake", target_speed_mps=max(0.0, speed - nearest["closing_mps"]),
                      target_accel_mps2=-clamp(req_decel * 1.3, 2.0, mu * G * 0.9),
                      rationale=f"Closing on object, TTC={rnd(ttc)}s; braking to match lead speed")
    elif ttc is not None and ttc < 6.0:
        action.update(action_type="set_speed", target_speed_mps=max(0.0, speed - nearest["closing_mps"]),
                      rationale=f"Following distance shrinking, TTC={rnd(ttc)}s; matching lead speed")
    elif speed < cruise - 0.5:
        action.update(action_type="accelerate", target_speed_mps=cruise, target_accel_mps2=1.5,
                      rationale="Road clear; accelerating to cruise speed")
    else:
        action.update(action_type="maintain", rationale="Road clear; maintaining speed")
    return action


# =============================================================================
# 4. Scenario Lab: 100-run batch simulation engine
# =============================================================================

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "lead_vehicle_brake": dict(
        description="Lead vehicle at similar speed brakes suddenly.",
        ego_speed=25.0, gap=38.0, lead_ratio=1.0, lead_brake=(1.0, 4.0), lead_decel=(4.0, 8.0),
        friction=0.8, fault=0.03),
    "stationary_obstacle": dict(
        description="Stationary obstacle (stalled vehicle / debris) ahead on the road.",
        ego_speed=22.0, gap=85.0, lead_ratio=0.0, lead_brake=None, lead_decel=None,
        friction=0.8, fault=0.03),
    "cut_in": dict(
        description="Slower vehicle cuts in at close range and brakes lightly.",
        ego_speed=28.0, gap=20.0, lead_ratio=0.72, lead_brake=(0.5, 2.0), lead_decel=(2.0, 5.0),
        friction=0.8, fault=0.03),
    "pedestrian_crossing": dict(
        description="Pedestrian steps into the lane in an urban speed zone.",
        ego_speed=14.0, gap=32.0, lead_ratio=0.0, lead_brake=None, lead_decel=None,
        friction=0.8, fault=0.03),
    "low_friction_brake": dict(
        description="Wet / icy road with a braking lead vehicle.",
        ego_speed=20.0, gap=45.0, lead_ratio=1.0, lead_brake=(1.0, 3.0), lead_decel=(3.0, 6.0),
        friction=0.35, fault=0.03),
    "sensor_degradation": dict(
        description="Lead vehicle braking while perception suffers frequent dropouts.",
        ego_speed=24.0, gap=40.0, lead_ratio=1.0, lead_brake=(1.0, 3.0), lead_decel=(4.0, 7.0),
        friction=0.75, fault=0.25),
}

SIM_DT = 0.05
SIM_MAX_T = 25.0


def sample_scenario_params(rng: random.Random, cfg: Dict[str, Any], req: ScenarioRunRequest) -> Dict[str, Any]:
    j = req.jitter_pct

    def gauss(base: float, lo: float, hi: float) -> float:
        return clamp(rng.gauss(base, abs(base) * j / 200.0), lo, hi)

    ego = gauss(req.ego_speed_mps or cfg["ego_speed"], 2.0, 60.0)
    gap = gauss(req.initial_gap_m or cfg["gap"], 3.0, 600.0)
    mu = gauss(req.friction or cfg["friction"], 0.1, 1.2)
    lead_speed = min(ego * cfg["lead_ratio"] * clamp(rng.gauss(1.0, j / 200.0), 0.5, 1.5), ego)
    lead_brake_time: Optional[float] = None
    lead_decel = 0.0
    if cfg["lead_brake"] is not None:
        lead_brake_time = rng.uniform(*cfg["lead_brake"])
        lead_decel = min(rng.uniform(*cfg["lead_decel"]), mu * G * 0.95)
    fault = req.sensor_fault_prob if req.sensor_fault_prob is not None else cfg["fault"]
    return {
        "ego_speed": ego, "gap": gap, "friction": mu, "lead_speed": lead_speed,
        "lead_brake_time": lead_brake_time, "lead_decel": lead_decel, "sensor_fault_prob": fault,
    }


def simulate_run(p: Dict[str, Any], arm: str, rng: random.Random) -> Dict[str, Any]:
    """
    One braking-scenario simulation.
      arm == "baseline": human-like driver  (slow reaction, distraction, later trigger)
      arm == "omega"   : OMEGA with guardrails (fast reaction, TTC-triggered braking,
                         occasional sensor dropouts adding delay)
    """
    mu = p["friction"]
    if arm == "baseline":
        react = rng.uniform(0.9, 1.7)
        if rng.random() < 0.08:
            react += rng.uniform(0.8, 1.6)  # distraction
        eff, jerk, ttc_trigger = rng.uniform(0.65, 0.9), 8.0, 3.2
        sensor_fault = False
    else:
        react = rng.uniform(0.15, 0.40)
        sensor_fault = rng.random() < p["sensor_fault_prob"]
        if sensor_fault:
            react += rng.uniform(0.5, 1.2)
        eff, jerk, ttc_trigger = 0.95, 20.0, 4.5
    a_max = mu * G * eff

    v = p["ego_speed"]
    x = 0.0
    lead_v = p["lead_speed"]
    lead_x = p["gap"]
    t = 0.0
    brake_start: Optional[float] = None
    guardrail_triggered = False
    min_gap = float("inf")
    min_ttc = float("inf")
    collision = False
    impact = 0.0

    for _ in range(int(SIM_MAX_T / SIM_DT)):
        gap = lead_x - x
        closing = v - lead_v
        min_gap = min(min_gap, gap)
        if closing > 0.1 and gap > 0:
            min_ttc = min(min_ttc, gap / closing)
        if gap <= 0:
            collision = True
            impact = max(0.0, closing)
            min_gap = min(min_gap, 0.0)
            break

        lead_braking = p["lead_brake_time"] is not None and t >= p["lead_brake_time"]
        if brake_start is None:
            ttc = gap / closing if closing > 0.1 else float("inf")
            req_decel = (closing * closing) / (2.0 * max(gap, 0.1)) if closing > 0 else 0.0
            if arm == "baseline":
                trigger = (lead_braking and lead_v < v) or ttc < ttc_trigger
            else:
                trigger = ttc < ttc_trigger or req_decel > 0.35 * mu * G
                if trigger:
                    guardrail_triggered = True
            if trigger:
                brake_start = t + react

        if brake_start is not None and t >= brake_start:
            a = min(a_max, (t - brake_start) * jerk)
            v = max(0.0, v - a * SIM_DT)
        if lead_braking:
            lead_v = max(0.0, lead_v - p["lead_decel"] * SIM_DT)

        x += v * SIM_DT
        lead_x += lead_v * SIM_DT
        t += SIM_DT

        if brake_start is not None and t >= brake_start and v <= lead_v + 0.05:
            break
        if v <= 0.01 and lead_v <= 0.01:
            break

    final_gap = lead_x - x
    near_miss = (not collision) and (min_gap < 2.0 or min_ttc < 1.5)
    return {
        "collision": collision,
        "near_miss": near_miss,
        "impact_speed_mps": rnd(impact),
        "min_gap_m": rnd(min_gap),
        "min_ttc_s": rnd(min_ttc) if math.isfinite(min_ttc) else None,
        "final_gap_m": rnd(final_gap),
        "reaction_s": rnd(react),
        "guardrail_triggered": guardrail_triggered,
        "sensor_fault": sensor_fault,
    }


def summarize_arm(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    collisions = [r_ for r_ in results if r_["collision"]]
    near = [r_ for r_ in results if r_["near_miss"]]
    ttcs = sorted(r_["min_ttc_s"] for r_ in results if r_["min_ttc_s"] is not None)
    gaps = sorted(r_["min_gap_m"] for r_ in results if r_["min_gap_m"] is not None)
    return {
        "runs": n,
        "collisions": len(collisions),
        "collision_rate_pct": rnd(100.0 * len(collisions) / n if n else 0.0, 2),
        "near_misses": len(near),
        "near_miss_rate_pct": rnd(100.0 * len(near) / n if n else 0.0, 2),
        "mean_impact_speed_mps": rnd(mean([r_["impact_speed_mps"] for r_ in collisions])),
        "max_impact_speed_mps": rnd(max([r_["impact_speed_mps"] for r_ in collisions], default=None)),
        "min_ttc_s": {"mean": rnd(mean(ttcs)), "p5": rnd(percentile(ttcs, 0.05)), "p50": rnd(percentile(ttcs, 0.5))},
        "min_gap_m": {"mean": rnd(mean(gaps)), "p5": rnd(percentile(gaps, 0.05)), "p50": rnd(percentile(gaps, 0.5))},
        "mean_reaction_s": rnd(mean([r_["reaction_s"] for r_ in results])),
        "guardrail_activations": sum(1 for r_ in results if r_["guardrail_triggered"]),
        "sensor_faults": sum(1 for r_ in results if r_["sensor_fault"]),
    }


def run_scenario_batch(req: ScenarioRunRequest) -> Dict[str, Any]:
    cfg = SCENARIOS[req.scenario_type]
    seed = req.seed if req.seed is not None else random.randrange(1 << 30)
    baseline: List[Dict[str, Any]] = []
    omega: List[Dict[str, Any]] = []
    saved = regress = both_collided = both_safe = 0
    started = time.perf_counter()

    for i in range(req.runs):
        params = sample_scenario_params(random.Random(seed * 1_000_003 + i * 7919 + 1), cfg, req)
        b = simulate_run(params, "baseline", random.Random(seed * 1_000_003 + i * 7919 + 2))
        o = simulate_run(params, "omega", random.Random(seed * 1_000_003 + i * 7919 + 3))
        b["run"] = o["run"] = i + 1
        baseline.append(b)
        omega.append(o)
        if b["collision"] and not o["collision"]:
            saved += 1
        elif o["collision"] and not b["collision"]:
            regress += 1
        elif b["collision"] and o["collision"]:
            both_collided += 1
        else:
            both_safe += 1

    sb, so = summarize_arm(baseline), summarize_arm(omega)
    reduction = None
    if sb["collisions"] > 0:
        reduction = rnd(100.0 * (sb["collisions"] - so["collisions"]) / sb["collisions"], 1)
    if so["collisions"] < sb["collisions"]:
        verdict = "OMEGA reduced collisions versus the baseline driver."
    elif so["collisions"] == sb["collisions"]:
        verdict = "OMEGA and baseline produced the same number of collisions."
    else:
        verdict = "OMEGA produced MORE collisions than baseline - review guardrail thresholds."

    run_id = new_id("scn")
    result: Dict[str, Any] = {
        "run_id": run_id,
        "scenario_type": req.scenario_type,
        "description": cfg["description"],
        "runs": req.runs,
        "seed": seed,
        "created_at": now_iso(),
        "duration_ms": rnd((time.perf_counter() - started) * 1000.0, 1),
        "baseline": sb,
        "omega": so,
        "comparison": {
            "collisions_avoided": sb["collisions"] - so["collisions"],
            "collision_reduction_pct": reduction,
            "paired_outcomes": {
                "both_safe": both_safe, "omega_saved": saved,
                "omega_regressed": regress, "both_collided": both_collided,
            },
            "verdict": verdict,
        },
    }
    if req.include_details:
        result["details"] = {"baseline": baseline, "omega": omega}
    store.put("scenario_runs", run_id, result)
    audit("SCENARIO_BATCH_COMPLETE", "info", run_id=run_id, scenario=req.scenario_type, runs=req.runs, seed=seed)
    return result


# =============================================================================
# 5. Fleet Route Optimisation & Energy Intelligence
# =============================================================================

ROAD_FACTOR = 1.25  # road distance / great-circle distance


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    r_earth = 6371.0088
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dphi = p2 - p1
    dlmb = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r_earth * math.asin(min(1.0, math.sqrt(h)))


def leg_km(a: GeoPoint, b: GeoPoint) -> float:
    return haversine_km(a, b) * ROAD_FACTOR


def energy_factors(avg_speed_kph: float, temp_c: float, traffic: float,
                   curb_kg: float, payload_kg: float, grade_pct: float = 0.0) -> Dict[str, float]:
    mass_ratio = (curb_kg + payload_kg) / max(curb_kg, 1.0)
    mass_f = 1.0 + 0.35 * (mass_ratio - 1.0)
    speed_f = 0.75 + 0.25 * (avg_speed_kph / 50.0) ** 2
    if temp_c < 20.0:
        temp_f = 1.0 + 0.012 * (20.0 - temp_c)  # cabin heating + cold battery
    elif temp_c > 28.0:
        temp_f = 1.0 + 0.006 * (temp_c - 28.0)  # A/C load
    else:
        temp_f = 1.0
    traffic_f = 1.0 + 0.12 * (traffic - 1.0)
    if grade_pct >= 0:
        grade_f = 1.0 + 0.08 * grade_pct
    else:
        grade_f = max(0.3, 1.0 + 0.04 * grade_pct)  # regenerative braking recovers some energy
    return {"mass": mass_f, "speed": speed_f, "temperature": temp_f, "traffic": traffic_f, "grade": grade_f}


def leg_energy_kwh(dist_km: float, base_eff: float, avg_speed_kph: float, temp_c: float, traffic: float,
                   curb_kg: float, payload_kg: float, grade_pct: float = 0.0) -> float:
    f = energy_factors(avg_speed_kph, temp_c, traffic, curb_kg, payload_kg, grade_pct)
    total = dist_km * base_eff
    for value in f.values():
        total *= value
    return max(0.0, total)


def route_cost(veh: FleetVehicleIn, seq: List[FleetStopIn], ctx: Dict[str, Any], metric: str) -> float:
    """Fast cost used by insertion / 2-opt. metric: 'energy' or 'distance'. Includes soft time-window penalty."""
    payload = sum(s.demand_kg for s in seq)
    loc = veh.start
    speed = max(5.0, veh.avg_speed_kph / ctx["traffic_factor"])
    total = 0.0
    t = 0.0
    for s in seq:
        d = leg_km(loc, s.location)
        if metric == "energy":
            total += leg_energy_kwh(d, veh.base_efficiency_kwh_per_km, veh.avg_speed_kph, ctx["ambient_temp_c"],
                                    ctx["traffic_factor"], veh.curb_mass_kg, payload)
        else:
            total += d
        t += d / speed * 60.0
        if s.window_start_min is not None and t < s.window_start_min:
            t = s.window_start_min
        if s.window_end_min is not None and t > s.window_end_min:
            total += (t - s.window_end_min) * 0.02
        t += s.service_min
        payload = max(0.0, payload - s.demand_kg)
        loc = s.location
    if ctx["return_to_start"] and seq:
        d = leg_km(loc, veh.start)
        if metric == "energy":
            total += leg_energy_kwh(d, veh.base_efficiency_kwh_per_km, veh.avg_speed_kph, ctx["ambient_temp_c"],
                                    ctx["traffic_factor"], veh.curb_mass_kg, 0.0)
        else:
            total += d
    return total


def assign_stops(vehicles: List[FleetVehicleIn], stops: List[FleetStopIn], ctx: Dict[str, Any]):
    """Cheapest-insertion assignment with payload capacity and optional load balancing."""
    routes: Dict[str, List[FleetStopIn]] = {v.vehicle_id: [] for v in vehicles}
    load: Dict[str, float] = {v.vehicle_id: 0.0 for v in vehicles}
    rlen: Dict[str, float] = {v.vehicle_id: 0.0 for v in vehicles}
    unassigned: List[Dict[str, str]] = []
    bal = {"energy": 0.0, "balanced": 0.15, "time": 0.4}[ctx["objective"]]
    ordered = sorted(stops, key=lambda s: (-s.priority,
                                           s.window_end_min if s.window_end_min is not None else 1e9,
                                           -s.demand_kg))
    for s in ordered:
        best = None
        for v in vehicles:
            if load[v.vehicle_id] + s.demand_kg > v.payload_capacity_kg + 1e-9:
                continue
            seq = routes[v.vehicle_id]
            pts = [v.start] + [x.location for x in seq]
            if ctx["return_to_start"]:
                pts.append(v.start)
            for pos in range(len(seq) + 1):
                a = pts[pos]
                if pos + 1 < len(pts):
                    delta = leg_km(a, s.location) + leg_km(s.location, pts[pos + 1]) - leg_km(a, pts[pos + 1])
                else:
                    delta = leg_km(a, s.location)
                cost = delta + bal * rlen[v.vehicle_id]
                if best is None or cost < best[0]:
                    best = (cost, v, pos, delta)
        if best is None:
            unassigned.append({"stop_id": s.stop_id, "reason": "no vehicle with enough payload capacity"})
            continue
        _, v, pos, delta = best
        routes[v.vehicle_id].insert(pos, s)
        load[v.vehicle_id] += s.demand_kg
        rlen[v.vehicle_id] += delta
    return routes, unassigned


def two_opt(veh: FleetVehicleIn, seq: List[FleetStopIn], ctx: Dict[str, Any], metric: str, deadline: float):
    best = list(seq)
    if len(best) < 3:
        return best
    best_c = route_cost(veh, best, ctx, metric)
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                if time.perf_counter() > deadline:
                    return best
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = route_cost(veh, cand, ctx, metric)
                if c < best_c - 1e-9:
                    best, best_c, improved = cand, c, True
    return best


def simulate_route(veh: FleetVehicleIn, stops: List[FleetStopIn], stations: List[ChargingStationIn],
                   ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Full energy / time / charging simulation of one route."""
    cap = veh.battery_capacity_kwh
    reserve = cap * ctx["reserve_pct"] / 100.0
    batt = cap * veh.battery_pct / 100.0
    payload = float(sum(s.demand_kg for s in stops))
    loc = veh.start
    eff_speed = max(5.0, veh.avg_speed_kph / ctx["traffic_factor"])
    t = dist = energy = charge_kwh = charge_min = 0.0
    late = 0
    seq_no = 0
    events: List[Dict[str, Any]] = [{
        "seq": 0, "type": "start", "id": veh.vehicle_id, "lat": veh.start.lat, "lon": veh.start.lon,
        "arrival_min": 0.0, "depart_min": 0.0, "battery_kwh_after": rnd(batt),
        "battery_pct_after": rnd(100.0 * batt / cap, 1), "payload_kg_after": rnd(payload),
    }]
    served: List[str] = []
    unserved: List[str] = []
    return_ok = True

    def leg(a: GeoPoint, b: GeoPoint, pl: float):
        d = leg_km(a, b)
        e = leg_energy_kwh(d, veh.base_efficiency_kwh_per_km, veh.avg_speed_kph, ctx["ambient_temp_c"],
                           ctx["traffic_factor"], veh.curb_mass_kg, pl)
        return d, e

    def rest_estimate(from_loc: GeoPoint, remaining: List[FleetStopIn], pl: float) -> float:
        total, cur, p = 0.0, from_loc, pl
        for s in remaining:
            _, e = leg(cur, s.location, p)
            total += e
            p = max(0.0, p - s.demand_kg)
            cur = s.location
        if ctx["return_to_start"]:
            _, e = leg(cur, veh.start, 0.0)
            total += e
        return total

    def go(dest: GeoPoint, remaining: List[FleetStopIn]):
        """Travel to dest, inserting a charging stop if the reserve would be violated."""
        nonlocal batt, t, dist, energy, charge_kwh, charge_min, loc, seq_no
        d, e = leg(loc, dest, payload)
        if batt - e < reserve - 1e-9:
            best = None
            for st in stations:
                d1, e1 = leg(loc, st.location, payload)
                if batt - e1 < cap * 0.02:
                    continue
                d2, _ = leg(st.location, dest, payload)
                if best is None or d1 + d2 < best[0]:
                    best = (d1 + d2, st, d1, e1)
            if best is None:
                return False, 0.0, 0.0
            _, st, d1, e1 = best
            batt -= e1
            energy += e1
            dist += d1
            t += d1 / eff_speed * 60.0
            loc = st.location
            need = rest_estimate(loc, remaining, payload) + reserve + cap * 0.05
            target = min(cap * ctx["charge_target_pct"] / 100.0, max(need, batt))
            add_kwh = target - batt
            if add_kwh <= 1e-6:
                return False, 0.0, 0.0
            ct = add_kwh / st.power_kw * 60.0 / 0.92
            arrival_t = t
            batt = target
            charge_kwh += add_kwh
            charge_min += ct
            t += ct
            seq_no += 1
            events.append({
                "seq": seq_no, "type": "charge", "id": st.station_id, "lat": st.location.lat, "lon": st.location.lon,
                "arrival_min": rnd(arrival_t, 1), "depart_min": rnd(t, 1), "leg_km": rnd(d1), "leg_kwh": rnd(e1),
                "charged_kwh": rnd(add_kwh), "charge_min": rnd(ct, 1),
                "battery_kwh_after": rnd(batt), "battery_pct_after": rnd(100.0 * batt / cap, 1),
                "payload_kg_after": rnd(payload),
            })
            d, e = leg(loc, dest, payload)
            if batt - e < reserve - 1e-9:
                return False, 0.0, 0.0
        batt -= e
        energy += e
        dist += d
        t += d / eff_speed * 60.0
        loc = dest
        return True, d, e

    for idx, s in enumerate(stops):
        ok, d, e = go(s.location, stops[idx:])
        if not ok:
            unserved = [x.stop_id for x in stops[idx:]]
            break
        arrival = t
        if s.window_start_min is not None and t < s.window_start_min:
            t = s.window_start_min
        is_late = s.window_end_min is not None and arrival > s.window_end_min
        if is_late:
            late += 1
        t += s.service_min
        payload = max(0.0, payload - s.demand_kg)
        seq_no += 1
        served.append(s.stop_id)
        events.append({
            "seq": seq_no, "type": "stop", "id": s.stop_id, "lat": s.location.lat, "lon": s.location.lon,
            "arrival_min": rnd(arrival, 1), "depart_min": rnd(t, 1), "leg_km": rnd(d), "leg_kwh": rnd(e),
            "battery_kwh_after": rnd(batt), "battery_pct_after": rnd(100.0 * batt / cap, 1),
            "payload_kg_after": rnd(payload), "late": is_late,
        })

    if ctx["return_to_start"] and stops:
        ok, d, e = go(veh.start, [])
        if ok:
            seq_no += 1
            events.append({
                "seq": seq_no, "type": "return", "id": veh.vehicle_id, "lat": veh.start.lat, "lon": veh.start.lon,
                "arrival_min": rnd(t, 1), "depart_min": rnd(t, 1), "leg_km": rnd(d), "leg_kwh": rnd(e),
                "battery_kwh_after": rnd(batt), "battery_pct_after": rnd(100.0 * batt / cap, 1),
                "payload_kg_after": 0.0,
            })
        else:
            return_ok = False

    return {
        "vehicle_id": veh.vehicle_id,
        "feasible": (not unserved) and return_ok,
        "return_to_start_ok": return_ok,
        "stops_served": served,
        "stops_unserved": unserved,
        "late_stops": late,
        "charging_stops": sum(1 for ev in events if ev["type"] == "charge"),
        "total_distance_km": rnd(dist),
        "total_energy_kwh": rnd(energy),
        "energy_kwh_per_100km": rnd(100.0 * energy / dist) if dist > 0 else None,
        "charged_kwh": rnd(charge_kwh),
        "charge_time_min": rnd(charge_min, 1),
        "total_time_min": rnd(t, 1),
        "final_battery_pct": rnd(100.0 * batt / cap, 1),
        "events": events,
        "_raw": {"dist": dist, "energy": energy, "time": t, "late": late, "charges": events and sum(
            1 for ev in events if ev["type"] == "charge")},
    }


def fleet_totals(sims: List[Dict[str, Any]], req: FleetOptimizeRequest, unassigned_count: int) -> Dict[str, Any]:
    dist = sum(s["_raw"]["dist"] for s in sims)
    energy = sum(s["_raw"]["energy"] for s in sims)
    times = [s["_raw"]["time"] for s in sims if s["_raw"]["time"] > 0]
    unserved = sum(len(s["stops_unserved"]) for s in sims)
    return {
        "distance_km": rnd(dist),
        "energy_kwh": rnd(energy),
        "makespan_min": rnd(max(times, default=0.0), 1),
        "total_vehicle_time_min": rnd(sum(times), 1),
        "charging_stops": sum(s["_raw"]["charges"] for s in sims),
        "late_stops": sum(s["late_stops"] for s in sims),
        "stops_unserved": unserved + unassigned_count,
        "energy_cost": rnd(energy * req.electricity_price_per_kwh, 2),
        "co2_kg": rnd(energy * req.grid_co2_kg_per_kwh, 2),
        "feasible_routes": sum(1 for s in sims if s["feasible"]),
    }


def pct_saving(base: Optional[float], new: Optional[float]) -> Optional[float]:
    if base is None or new is None or base <= 0:
        return None
    return rnd(100.0 * (base - new) / base, 1)


def optimize_fleet(req: FleetOptimizeRequest, vehicles: List[FleetVehicleIn]) -> Dict[str, Any]:
    if req.objective not in ("energy", "time", "balanced"):
        raise HTTPException(status_code=422, detail="objective must be one of: energy, time, balanced")
    ctx = {
        "objective": req.objective, "reserve_pct": req.reserve_pct, "charge_target_pct": req.charge_target_pct,
        "ambient_temp_c": req.ambient_temp_c, "traffic_factor": req.traffic_factor,
        "return_to_start": req.return_to_start,
    }
    started = time.perf_counter()
    deadline = started + req.time_limit_s
    metric = "distance" if req.objective == "time" else "energy"

    # ---- Optimised plan ----------------------------------------------------
    routes, unassigned = assign_stops(vehicles, req.stops, ctx)
    if req.use_two_opt:
        active = [v for v in vehicles if len(routes[v.vehicle_id]) > 2]
        per_route_budget = max(0.05, req.time_limit_s / max(1, len(active)))
        for v in active:
            routes[v.vehicle_id] = two_opt(v, routes[v.vehicle_id], ctx, metric,
                                           min(deadline, time.perf_counter() + per_route_budget))
    sims = [simulate_route(v, routes[v.vehicle_id], req.charging_stations, ctx) for v in vehicles]
    totals = fleet_totals(sims, req, len(unassigned))

    # ---- Naive baseline: round-robin in input order, no optimisation --------
    base_routes: Dict[str, List[FleetStopIn]] = {v.vehicle_id: [] for v in vehicles}
    base_load = {v.vehicle_id: 0.0 for v in vehicles}
    base_unassigned = 0
    for i, s in enumerate(req.stops):
        placed = False
        for k in range(len(vehicles)):
            v = vehicles[(i + k) % len(vehicles)]
            if base_load[v.vehicle_id] + s.demand_kg <= v.payload_capacity_kg + 1e-9:
                base_routes[v.vehicle_id].append(s)
                base_load[v.vehicle_id] += s.demand_kg
                placed = True
                break
        if not placed:
            base_unassigned += 1
    base_sims = [simulate_route(v, base_routes[v.vehicle_id], req.charging_stations, ctx) for v in vehicles]
    base_totals = fleet_totals(base_sims, req, base_unassigned)

    savings = {
        "distance_pct": pct_saving(base_totals["distance_km"], totals["distance_km"]),
        "energy_pct": pct_saving(base_totals["energy_kwh"], totals["energy_kwh"]),
        "makespan_pct": pct_saving(base_totals["makespan_min"], totals["makespan_min"]),
        "energy_kwh_saved": rnd((base_totals["energy_kwh"] or 0) - (totals["energy_kwh"] or 0)),
        "cost_saved": rnd((base_totals["energy_cost"] or 0) - (totals["energy_cost"] or 0), 2),
        "co2_kg_saved": rnd((base_totals["co2_kg"] or 0) - (totals["co2_kg"] or 0), 2),
    }

    # ---- Energy intelligence insights -------------------------------------
    insights: List[str] = []
    energies = [s["_raw"]["energy"] for s in sims if s["_raw"]["energy"] > 0]
    if len(energies) > 1:
        avg_e = sum(energies) / len(energies)
        spread = (max(energies) - min(energies)) / avg_e if avg_e > 0 else 0
        if spread > 0.5:
            insights.append(f"Energy load is unbalanced across vehicles (spread {spread * 100:.0f}% of mean); "
                            "consider objective='balanced' or adding a vehicle.")
    for s, v in zip(sims, vehicles):
        if s["charging_stops"]:
            insights.append(f"{v.vehicle_id}: needs {s['charging_stops']} en-route charge(s) "
                            f"({s['charged_kwh']} kWh, {s['charge_time_min']} min). Pre-charging at the depot "
                            "would remove this downtime.")
        if not s["feasible"]:
            insights.append(f"{v.vehicle_id}: route infeasible with current battery and charger coverage; "
                            f"{len(s['stops_unserved'])} stop(s) unserved.")
        if v.battery_pct < 40 and s["stops_served"]:
            insights.append(f"{v.vehicle_id}: starts at only {v.battery_pct:.0f}% battery; charge before dispatch.")
    if req.ambient_temp_c < 5:
        insights.append(f"Cold ambient temperature ({req.ambient_temp_c:.0f}C) adds ~"
                        f"{(energy_factors(40, req.ambient_temp_c, 1, 1, 0)['temperature'] - 1) * 100:.0f}% energy use.")
    if req.traffic_factor > 1.5:
        insights.append("Heavy traffic factor increases energy via stop-and-go; shifting departures off-peak helps.")
    if (savings["energy_pct"] or 0) > 0:
        insights.append(f"Optimised plan saves {savings['energy_pct']}% energy versus the naive round-robin plan.")
    if unassigned:
        insights.append(f"{len(unassigned)} stop(s) could not be assigned (payload capacity).")

    plan_id = new_id("plan")
    for s in sims + base_sims:
        s.pop("_raw", None)
    plan = {
        "plan_id": plan_id,
        "created_at": now_iso(),
        "objective": req.objective,
        "algorithm": "cheapest-insertion + 2-opt (energy-aware, payload-aware, charger-aware)",
        "solve_time_ms": rnd((time.perf_counter() - started) * 1000.0, 1),
        "routes": sims,
        "unassigned_stops": unassigned,
        "fleet_totals": totals,
        "baseline_totals": base_totals,
        "savings": savings,
        "insights": insights,
    }
    store.put("fleet_plans", plan_id, plan)
    audit("FLEET_PLAN_CREATED", "info", plan_id=plan_id, vehicles=len(vehicles), stops=len(req.stops),
          objective=req.objective)
    return plan


def estimate_energy(req: EnergyEstimateRequest) -> Dict[str, Any]:
    factors = energy_factors(req.avg_speed_kph, req.ambient_temp_c, req.traffic_factor,
                             req.curb_mass_kg, req.payload_kg, req.grade_pct)
    energy = leg_energy_kwh(req.distance_km, req.base_efficiency_kwh_per_km, req.avg_speed_kph,
                            req.ambient_temp_c, req.traffic_factor, req.curb_mass_kg, req.payload_kg, req.grade_pct)
    per_km = energy / req.distance_km
    available = req.battery_capacity_kwh * req.battery_pct / 100.0
    reserve = req.battery_capacity_kwh * req.reserve_pct / 100.0
    usable = max(0.0, available - reserve)
    range_km = usable / per_km if per_km > 0 else None
    soc_used = 100.0 * energy / req.battery_capacity_kwh
    return {
        "energy_kwh": rnd(energy),
        "kwh_per_100km": rnd(per_km * 100.0),
        "soc_used_pct": rnd(soc_used, 2),
        "soc_after_pct": rnd(req.battery_pct - soc_used, 2),
        "usable_range_km_at_conditions": rnd(range_km, 1),
        "trip_feasible_without_charging": energy <= usable + 1e-9,
        "charging_needed_kwh": rnd(max(0.0, energy - usable)),
        "factors": {k: rnd(val, 3) for k, val in factors.items()},
        "cost": rnd(energy * req.electricity_price_per_kwh, 2),
        "co2_kg": rnd(energy * req.grid_co2_kg_per_kwh, 2),
    }


# =============================================================================
# 6. Digital Twin vs Reality gap monitoring
# =============================================================================

TWIN_TOLERANCES = {"position": 2.0, "speed": 0.5, "heading": 3.0, "battery": 1.5, "temp": 5.0}
TWIN_WEIGHTS = {"position": 0.30, "speed": 0.25, "heading": 0.10, "battery": 0.20, "temp": 0.15}
TWIN_EMA_ALPHA = 0.3


def wrap_angle(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def twin_status_from_index(idx: float) -> str:
    if idx < 1.0:
        return "SYNCED"
    if idx < 2.5:
        return "DRIFTING"
    return "DIVERGED"


def twin_history(vehicle_id: str) -> List[dict]:
    rows = [r_ for r_ in store.values("twin_readings")
            if r_.get("vehicle_id") == vehicle_id and not str(r_.get("reading_id", "")).startswith("latest:")]
    return sorted(rows, key=lambda r_: r_.get("ts_epoch", 0))


def process_twin_reading(body: TwinReadingIn) -> Dict[str, Any]:
    vid = body.vehicle_id
    latest = store.get("twin_readings", f"latest:{vid}") or {}
    bias = latest.get("bias") or {"dx": 0.0, "dy": 0.0, "speed": 0.0, "heading": 0.0, "battery": 0.0, "temp": 0.0}
    tw, re_ = body.twin, body.real

    raw = {
        "dx": tw.position.x - re_.position.x,
        "dy": tw.position.y - re_.position.y,
        "speed": tw.speed_mps - re_.speed_mps,
        "heading": wrap_angle(tw.heading_deg - re_.heading_deg),
        "battery": tw.battery_pct - re_.battery_pct,
        "temp": (tw.temp_c - re_.temp_c) if (tw.temp_c is not None and re_.temp_c is not None) else None,
    }
    corrected = {
        "position_m": math.hypot(raw["dx"] - bias.get("dx", 0.0), raw["dy"] - bias.get("dy", 0.0)),
        "speed_mps": abs(raw["speed"] - bias.get("speed", 0.0)),
        "heading_deg": abs(wrap_angle(raw["heading"] - bias.get("heading", 0.0))),
        "battery_pct": abs(raw["battery"] - bias.get("battery", 0.0)),
        "temp_c": abs(raw["temp"] - bias.get("temp", 0.0)) if raw["temp"] is not None else None,
    }
    normalized = {
        "position": corrected["position_m"] / TWIN_TOLERANCES["position"],
        "speed": corrected["speed_mps"] / TWIN_TOLERANCES["speed"],
        "heading": corrected["heading_deg"] / TWIN_TOLERANCES["heading"],
        "battery": corrected["battery_pct"] / TWIN_TOLERANCES["battery"],
    }
    if corrected["temp_c"] is not None:
        normalized["temp"] = corrected["temp_c"] / TWIN_TOLERANCES["temp"]
    wsum = sum(TWIN_WEIGHTS[k] for k in normalized)
    gap_index = math.sqrt(sum(TWIN_WEIGHTS[k] * normalized[k] ** 2 for k in normalized) / wsum)

    prev_ema = latest.get("ema_gap_index")
    ema = gap_index if prev_ema is None else TWIN_EMA_ALPHA * gap_index + (1 - TWIN_EMA_ALPHA) * prev_ema
    status = twin_status_from_index(ema)
    instant_status = twin_status_from_index(gap_index)
    prev_status = latest.get("status")

    # Physical consequence: stopping-distance prediction error caused by the gap
    real_vehicle = store.get("vehicles", vid) or {}
    mu = float(real_vehicle.get("road_friction", 0.8))
    grade = float(real_vehicle.get("grade_pct", 0.0))
    sd_twin = stopping_distance(tw.speed_mps, SYSTEM_REACTION_S, mu, grade)["total_stopping_distance_m"]
    sd_real = stopping_distance(re_.speed_mps, SYSTEM_REACTION_S, mu, grade)["total_stopping_distance_m"]

    ts_epoch = time.time()
    reading_id = new_id("twr")
    reading = {
        "reading_id": reading_id,
        "vehicle_id": vid,
        "timestamp": body.timestamp or now_iso(),
        "created_at": now_iso(),
        "ts_epoch": ts_epoch,
        "raw_errors": {k: rnd(val) for k, val in raw.items()},
        "corrected_errors": {k: rnd(val) for k, val in corrected.items()},
        "normalized_errors": {k: rnd(val) for k, val in normalized.items()},
        "gap_index": rnd(gap_index),
        "ema_gap_index": rnd(ema),
        "instant_status": instant_status,
        "status": status,
        "stopping_distance_twin_m": rnd(sd_twin),
        "stopping_distance_real_m": rnd(sd_real),
        "stopping_distance_error_m": rnd(sd_twin - sd_real),
    }
    store.put("twin_readings", reading_id, reading)

    history = twin_history(vid)
    idxs = [h["gap_index"] for h in history if h.get("gap_index") is not None]
    trend = "stable"
    if len(idxs) >= 6:
        recent = sum(idxs[-3:]) / 3.0
        earlier = sum(idxs[-6:-3]) / 3.0
        if recent > earlier * 1.15 + 0.05:
            trend = "worsening"
        elif recent < earlier * 0.85 - 0.05:
            trend = "improving"

    alerts: List[str] = []
    if status != prev_status and status in ("DRIFTING", "DIVERGED"):
        alerts.append(f"Digital twin for {vid} moved to {status} (gap index {rnd(ema, 2)}).")
        audit("TWIN_STATUS_CHANGE", "critical" if status == "DIVERGED" else "warning",
              vehicle_id=vid, previous=prev_status, status=status, gap_index=rnd(ema))
    if status == "DIVERGED":
        alerts.append("AI actions that increase risk now require human approval until the twin is recalibrated.")

    latest_row = {
        "reading_id": f"latest:{vid}",
        "vehicle_id": vid,
        "status": status,
        "gap_index": rnd(gap_index),
        "ema_gap_index": ema,
        "trend": trend,
        "bias": bias,
        "readings_count": int(latest.get("readings_count", 0)) + 1,
        "last_reading_id": reading_id,
        "last_updated": now_iso(),
        "created_at": latest.get("created_at", now_iso()),
        "recalibrated_at": latest.get("recalibrated_at"),
        "ts_epoch": 0,
    }
    store.put("twin_readings", f"latest:{vid}", latest_row)

    return {
        "vehicle_id": vid,
        "status": status,
        "instant_status": instant_status,
        "gap_index": rnd(gap_index),
        "ema_gap_index": rnd(ema),
        "trend": trend,
        "errors": reading["corrected_errors"],
        "raw_errors": reading["raw_errors"],
        "normalized_errors": reading["normalized_errors"],
        "tolerances": TWIN_TOLERANCES,
        "stopping_distance_error_m": reading["stopping_distance_error_m"],
        "ai_actions_restricted": status == "DIVERGED",
        "bias_correction_active": any(abs(val) > 1e-9 for val in bias.values()),
        "alerts": alerts,
        "recommendation": (
            "Twin is within tolerance." if status == "SYNCED" else
            "Twin is drifting; inspect sensor calibration and model parameters." if status == "DRIFTING" else
            "Twin has diverged; stop trusting twin-based predictions and recalibrate."),
        "reading_id": reading_id,
    }


# =============================================================================
# Demo seeding (in-memory fallback convenience)
# =============================================================================

def seed_demo(force: bool = False) -> Dict[str, Any]:
    if store.values("vehicles") and not force:
        return {"seeded": False, "reason": "vehicles already exist"}
    demo = [
        dict(vehicle_id="OMEGA-01", name="Aegis Sedan", vehicle_type="car", mass_kg=1800, battery_capacity_kwh=75,
             battery_pct=86, max_speed_mps=33, speed_mps=22.0, lat=37.7749, lon=-122.4194,
             base_efficiency_kwh_per_km=0.16, payload_capacity_kg=400),
        dict(vehicle_id="OMEGA-02", name="Courier Van", vehicle_type="van", mass_kg=2600, battery_capacity_kwh=90,
             battery_pct=64, max_speed_mps=28, speed_mps=15.0, lat=37.7849, lon=-122.4094,
             base_efficiency_kwh_per_km=0.22, payload_capacity_kg=1200),
        dict(vehicle_id="OMEGA-03", name="Freight Truck", vehicle_type="truck", mass_kg=7800, battery_capacity_kwh=220,
             battery_pct=41, max_speed_mps=25, speed_mps=0.0, lat=37.7649, lon=-122.4294,
             base_efficiency_kwh_per_km=0.55, payload_capacity_kg=4500),
    ]
    created = []
    for d in demo:
        v = make_vehicle(d)
        store.put("vehicles", v["vehicle_id"], v)
        created.append(v["vehicle_id"])
    audit("DEMO_SEEDED", "info", vehicles=created)
    return {"seeded": True, "vehicles": created}


# =============================================================================
# FastAPI app
# =============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("OMEGA_SEED_DEMO", "1") != "0":
        seed_demo()
    log.info("%s v%s ready - storage mode: %s", APP_NAME, APP_VERSION, store.mode)
    yield


app = FastAPI(
    title=f"{APP_NAME} API",
    version=APP_VERSION,
    description="Autonomous mobility safety guardrails, human-in-the-loop approvals, scenario lab, "
                "fleet energy optimisation and digital-twin gap monitoring.",
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=("*" not in _origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# -----------------------------------------------------------------------------
# Core: health / status / audit / demo
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {"name": APP_NAME, "version": APP_VERSION, "docs": "/docs", "storage_mode": store.mode}


@app.get("/api/health")
def health():
    return {"status": "ok", "time": now_iso(), "uptime_s": rnd(time.time() - START_TIME, 1),
            "storage_mode": store.mode}


@app.get("/api/status")
def status():
    expire_stale_approvals()
    pending = sum(1 for a in store.values("approvals") if a.get("status") == "pending")
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "storage": {
            "mode": store.mode,
            "supabase_connected": store.client is not None,
            "last_error": store.last_error,
            "rows": store.counts(),
        },
        "safety": {
            "hard_speed_cap_mps": HARD_SPEED_CAP_MPS,
            "system_reaction_s": SYSTEM_REACTION_S,
            "approval_ttl_s": APPROVAL_TTL_S,
            "approver_token_required": bool(os.getenv("HUMAN_APPROVER_TOKEN")),
            "pending_approvals": pending,
            "guardrail_rules": len(GUARDRAIL_CATALOG),
        },
        "features": ["ttc", "stopping_distance", "human_approval", "guardrails", "scenario_lab",
                     "fleet_optimization", "energy_intelligence", "digital_twin_gap", "supabase_optional"],
    }


@app.get("/api/audit")
def get_audit(limit: int = Query(100, ge=1, le=1000), event: Optional[str] = None,
              severity: Optional[str] = None):
    rows = store.values("audit_log")
    if event:
        rows = [r_ for r_ in rows if r_.get("event") == event]
    if severity:
        rows = [r_ for r_ in rows if r_.get("severity") == severity]
    return {"count": len(rows), "entries": newest_first(rows)[:limit]}


@app.post("/api/demo/seed")
def demo_seed(force: bool = False):
    return seed_demo(force=force)


# -----------------------------------------------------------------------------
# Vehicles & telemetry
# -----------------------------------------------------------------------------

@app.post("/api/vehicles", status_code=201)
def create_vehicle(body: VehicleCreate):
    data = dump(body)
    if data.get("vehicle_id") and store.get("vehicles", data["vehicle_id"]):
        raise HTTPException(status_code=409, detail=f"Vehicle '{data['vehicle_id']}' already exists")
    v = make_vehicle(data)
    store.put("vehicles", v["vehicle_id"], v)
    audit("VEHICLE_CREATED", "info", vehicle_id=v["vehicle_id"])
    return v


@app.get("/api/vehicles")
def list_vehicles():
    rows = newest_first(store.values("vehicles"))
    return {"count": len(rows), "vehicles": rows}


@app.get("/api/vehicles/{vehicle_id}")
def get_vehicle(vehicle_id: str):
    v = get_vehicle_or_404(vehicle_id)
    return {**v, **vehicle_safety_snapshot(v)}


@app.delete("/api/vehicles/{vehicle_id}")
def delete_vehicle(vehicle_id: str):
    get_vehicle_or_404(vehicle_id)
    store.delete("vehicles", vehicle_id)
    audit("VEHICLE_DELETED", "warning", vehicle_id=vehicle_id)
    return {"deleted": True, "vehicle_id": vehicle_id}


@app.post("/api/vehicles/{vehicle_id}/telemetry")
def update_telemetry(vehicle_id: str, body: TelemetryUpdate):
    v = get_vehicle_or_404(vehicle_id)
    data = dump(body)
    for key in ("speed_mps", "heading_deg", "battery_pct", "road_friction", "grade_pct",
                "ambient_temp_c", "lat", "lon"):
        if data.get(key) is not None:
            v[key] = data[key]
    if data.get("position") is not None:
        v["position"] = data["position"]
    if data.get("distance_km_delta"):
        v["odometer_km"] = v.get("odometer_km", 0.0) + data["distance_km_delta"]
    if data.get("energy_used_kwh_delta"):
        v["energy_used_kwh_total"] = v.get("energy_used_kwh_total", 0.0) + data["energy_used_kwh_delta"]
    v["updated_at"] = now_iso()
    store.put("vehicles", vehicle_id, v)
    return {**v, **vehicle_safety_snapshot(v)}


# -----------------------------------------------------------------------------
# PHANTOM: perception detections
# -----------------------------------------------------------------------------

@app.post("/api/phantom/detections", status_code=201)
def ingest_detection(body: DetectionIn):
    v = get_vehicle_or_404(body.vehicle_id)
    data = dump(body)
    closing = max(data["relative_speed_mps"], 0.0)
    ttc = compute_ttc(data["distance_m"], closing, 0.0) if closing > 0 else None
    sd = stopping_distance(v["speed_mps"], SYSTEM_REACTION_S, v.get("road_friction", 0.8), v.get("grade_pct", 0.0))
    det_id = new_id("det")
    row = {
        "detection_id": det_id,
        **data,
        "ttc_s": rnd(ttc),
        "ttc_risk": ttc_risk(ttc),
        "can_stop_in_gap": sd["total_stopping_distance_m"] <= data["distance_m"],
        "ts_epoch": time.time(),
        "created_at": now_iso(),
    }
    store.put("detections", det_id, row)
    if data["in_path"]:
        v["threat_level"] = ttc_risk(ttc) if ttc is not None else "none"
        v["updated_at"] = now_iso()
        store.put("vehicles", v["vehicle_id"], v)
    if ttc is not None and ttc < 3.0 and data["in_path"]:
        audit("PHANTOM_HIGH_THREAT", "warning", vehicle_id=v["vehicle_id"], detection_id=det_id, ttc_s=rnd(ttc))
    return {**row, "stopping_distance": format_stopping(sd, v["speed_mps"])}


@app.get("/api/phantom/detections/{vehicle_id}")
def list_detections(vehicle_id: str, active_only: bool = True, limit: int = Query(50, ge=1, le=500)):
    get_vehicle_or_404(vehicle_id)
    rows = recent_detections(vehicle_id) if active_only else [
        d for d in store.values("detections") if d.get("vehicle_id") == vehicle_id]
    rows = sorted(rows, key=lambda d: d.get("ts_epoch", 0), reverse=True)[:limit]
    return {"vehicle_id": vehicle_id, "count": len(rows), "detections": rows}


# -----------------------------------------------------------------------------
# 1. TTC & stopping-distance API
# -----------------------------------------------------------------------------

@app.post("/api/safety/ttc")
def api_ttc(body: TTCRequest):
    ttc = compute_ttc(body.distance_m, body.ego_speed_mps, body.target_speed_mps,
                      body.ego_accel_mps2, body.target_accel_mps2)
    closing = body.ego_speed_mps - body.target_speed_mps
    return {
        "ttc_s": rnd(ttc),
        "collision_course": ttc is not None,
        "risk_level": ttc_risk(ttc),
        "closing_speed_mps": rnd(closing),
        "relative_accel_mps2": rnd(body.ego_accel_mps2 - body.target_accel_mps2),
        "model": "constant-acceleration; valid until either body stops",
    }


@app.post("/api/safety/stopping-distance")
def api_stopping_distance(body: StoppingDistanceRequest):
    res = stopping_distance(body.speed_mps, body.reaction_time_s, body.friction_coefficient,
                            body.grade_pct, body.brake_efficiency)
    out = format_stopping(res, body.speed_mps)
    if body.available_gap_m is not None:
        margin = body.available_gap_m - res["total_stopping_distance_m"]
        out["available_gap_m"] = rnd(body.available_gap_m)
        out["can_stop"] = margin >= 0
        out["margin_m"] = rnd(margin)
    return out


@app.post("/api/safety/assess")
def api_assess(body: SafetyAssessRequest):
    mu = body.friction_coefficient
    grade = body.grade_pct
    ego = body.ego_speed_mps
    vehicle = None
    if body.vehicle_id:
        vehicle = get_vehicle_or_404(body.vehicle_id)
        ego = vehicle["speed_mps"] if ego is None else ego
        mu = vehicle.get("road_friction", 0.8) if mu is None else mu
        grade = vehicle.get("grade_pct", 0.0) if grade is None else grade
    if ego is None:
        raise HTTPException(status_code=422, detail="Provide ego_speed_mps or a known vehicle_id")
    mu = 0.7 if mu is None else mu
    grade = 0.0 if grade is None else grade

    ttc = compute_ttc(body.distance_m, ego, body.target_speed_mps)
    sd = stopping_distance(ego, body.reaction_time_s, mu, grade)
    lead_stop = body.target_speed_mps ** 2 / (2.0 * max(2.0, mu * G))
    margin = body.distance_m + lead_stop - sd["total_stopping_distance_m"] - MIN_STANDSTILL_GAP_M
    if margin < 0 or ttc_risk(ttc) == "critical":
        rec = "EMERGENCY_BRAKE"
    elif ttc_risk(ttc) in ("high", "medium"):
        rec = "BRAKE_OR_INCREASE_GAP"
    else:
        rec = "SAFE"
    return {
        "vehicle_id": body.vehicle_id,
        "ttc_s": rnd(ttc),
        "risk_level": ttc_risk(ttc),
        "stopping_distance": format_stopping(sd, ego),
        "rss_margin_m": rnd(margin),
        "can_stop_safely": margin >= 0,
        "recommendation": rec,
    }


# -----------------------------------------------------------------------------
# 3. Guardrails: catalogue, dry-run and AI action submission
# -----------------------------------------------------------------------------

@app.get("/api/safety/guardrails")
def list_guardrails():
    return {
        "constants": {
            "hard_speed_cap_mps": HARD_SPEED_CAP_MPS,
            "system_reaction_s": SYSTEM_REACTION_S,
            "min_standstill_gap_m": MIN_STANDSTILL_GAP_M,
            "detection_ttl_s": DETECTION_TTL_S,
            "forbidden_actions": sorted(FORBIDDEN_ACTIONS),
            "known_actions": sorted(KNOWN_ACTIONS),
        },
        "rules": GUARDRAIL_CATALOG,
    }


@app.post("/api/safety/evaluate")
def evaluate_only(body: AIActionRequest):
    """Dry-run: evaluate guardrails without executing or creating an approval."""
    v = get_vehicle_or_404(body.vehicle_id)
    return evaluate_guardrails(v, dump(body))


@app.post("/api/ai/action")
def submit_ai_action(body: AIActionRequest):
    """
    Every AI-proposed action passes through the backend guardrails:
      BLOCKED          -> HTTP 403, nothing executes
      PENDING_APPROVAL -> HTTP 202, a human must call /api/safety/approve
      APPROVED         -> HTTP 200, executed immediately
    """
    v = get_vehicle_or_404(body.vehicle_id)
    record = process_action(v, dump(body))
    code = {"BLOCKED": 403, "PENDING_APPROVAL": 202}.get(record["decision"], 200)
    return JSONResponse(status_code=code, content=json_safe(record))


@app.get("/api/ai/actions")
def list_ai_actions(vehicle_id: Optional[str] = None, status_filter: Optional[str] = Query(None, alias="status"),
                    limit: int = Query(100, ge=1, le=1000)):
    expire_stale_approvals()
    rows = store.values("ai_actions")
    if vehicle_id:
        rows = [r_ for r_ in rows if r_.get("vehicle_id") == vehicle_id]
    if status_filter:
        rows = [r_ for r_ in rows if r_.get("status") == status_filter]
    return {"count": len(rows), "actions": newest_first(rows)[:limit]}


@app.get("/api/ai/actions/{action_id}")
def get_ai_action(action_id: str):
    expire_stale_approvals()
    row = store.get("ai_actions", action_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Action not found")
    return row


@app.post("/api/omega/decide")
def omega_decision(body: DecideRequest):
    """OMEGA proposes an action from current perception; optionally submits it through the guardrails."""
    v = get_vehicle_or_404(body.vehicle_id)
    proposal = omega_decide(v, body.cruise_speed_mps)
    out: Dict[str, Any] = {"proposal": json_safe(proposal), "submitted": False}
    if body.submit:
        act = dict(proposal)
        act.setdefault("params", {})
        record = process_action(v, act)
        out["submitted"] = True
        out["result"] = json_safe(record)
    return out


# -----------------------------------------------------------------------------
# 2. Human approval workflow
# -----------------------------------------------------------------------------

@app.get("/api/safety/approvals")
def list_approvals(status_filter: Optional[str] = Query(None, alias="status"), vehicle_id: Optional[str] = None,
                   limit: int = Query(100, ge=1, le=1000)):
    expire_stale_approvals()
    rows = store.values("approvals")
    if status_filter:
        rows = [r_ for r_ in rows if r_.get("status") == status_filter]
    if vehicle_id:
        rows = [r_ for r_ in rows if r_.get("vehicle_id") == vehicle_id]
    return {"count": len(rows), "approvals": newest_first(rows)[:limit]}


@app.get("/api/safety/approvals/{approval_id}")
def get_approval(approval_id: str):
    expire_stale_approvals()
    ap = store.get("approvals", approval_id)
    if ap is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return ap


@app.post("/api/safety/approve")
def approve_action(body: ApprovalDecision, x_approver_token: Optional[str] = Header(default=None)):
    ap = load_pending_approval(body.approval_id)
    record = store.get("ai_actions", ap["action_id"])
    if record is None:
        raise HTTPException(status_code=404, detail="Underlying action not found")
    act = record["action"]
    assert_human_approver(body.decided_by, x_approver_token, act.get("source"))

    vehicle = get_vehicle_or_404(ap["vehicle_id"])
    # Re-run guardrails against CURRENT vehicle state: a human approval can never override a hard block.
    ev = evaluate_guardrails(vehicle, act)
    if ev["decision"] == "BLOCKED":
        ap.update(status="rejected", decided_by="guardrail-recheck", decided_at=now_iso(),
                  note="Auto-rejected: conditions changed and the action is now blocked by guardrails.")
        store.put("approvals", ap["approval_id"], ap)
        record.update(status="blocked", decision="BLOCKED", violations=ev["violations"], metrics=ev["metrics"])
        store.put("ai_actions", record["action_id"], record)
        audit("APPROVAL_OVERRIDDEN_BY_GUARDRAIL", "critical", approval_id=ap["approval_id"],
              action_id=record["action_id"], approver=body.decided_by,
              rules=[v_["rule"] for v_ in ev["violations"] if v_["severity"] == "block"])
        return JSONResponse(status_code=409, content=json_safe({
            "detail": "Approval refused: guardrails now block this action (hard blocks cannot be overridden).",
            "approval": ap, "violations": ev["violations"]}))

    effect = apply_action(vehicle, act)
    ap.update(status="approved", decided_by=body.decided_by, decided_at=now_iso(), note=body.note)
    store.put("approvals", ap["approval_id"], ap)
    record.update(status="executed", effect=effect, metrics=ev["metrics"], approved_by=body.decided_by)
    store.put("ai_actions", record["action_id"], record)
    audit("APPROVAL_GRANTED", "info", approval_id=ap["approval_id"], action_id=record["action_id"],
          approver=body.decided_by, action_type=act.get("action_type"))
    return {"approval": ap, "action": record, "effect": effect}


@app.post("/api/safety/reject")
def reject_action(body: ApprovalDecision, x_approver_token: Optional[str] = Header(default=None)):
    ap = load_pending_approval(body.approval_id)
    record = store.get("ai_actions", ap["action_id"])
    source = record["action"].get("source") if record else None
    assert_human_approver(body.decided_by, x_approver_token, source)
    ap.update(status="rejected", decided_by=body.decided_by, decided_at=now_iso(), note=body.note)
    store.put("approvals", ap["approval_id"], ap)
    if record:
        record.update(status="rejected", rejected_by=body.decided_by)
        store.put("ai_actions", record["action_id"], record)
    audit("APPROVAL_REJECTED", "info", approval_id=ap["approval_id"], action_id=ap["action_id"],
          approver=body.decided_by, note=body.note)
    return {"approval": ap, "action": record}


@app.post("/api/safety/cancel")
def cancel_action(body: ApprovalDecision):
    """Cancel a pending request (e.g. the situation resolved). Safe by design, so no approver auth is needed."""
    ap = load_pending_approval(body.approval_id)
    record = store.get("ai_actions", ap["action_id"])
    ap.update(status="cancelled", decided_by=body.decided_by, decided_at=now_iso(), note=body.note)
    store.put("approvals", ap["approval_id"], ap)
    if record:
        record.update(status="cancelled", cancelled_by=body.decided_by)
        store.put("ai_actions", record["action_id"], record)
    audit("APPROVAL_CANCELLED", "info", approval_id=ap["approval_id"], action_id=ap["action_id"],
          cancelled_by=body.decided_by, note=body.note)
    return {"approval": ap, "action": record}


# -----------------------------------------------------------------------------
# 4. Scenario Lab
# -----------------------------------------------------------------------------

@app.get("/api/scenarios/presets")
def scenario_presets():
    return {"default_runs": 100, "max_runs": 1000, "presets": {
        name: {
            "description": cfg["description"], "ego_speed_mps": cfg["ego_speed"], "initial_gap_m": cfg["gap"],
            "friction": cfg["friction"], "sensor_fault_prob": cfg["fault"],
        } for name, cfg in SCENARIOS.items()}}


@app.post("/api/scenarios/run")
def scenario_run(body: ScenarioRunRequest):
    """Run a batch (default 100 runs) of the scenario for both the baseline driver and OMEGA."""
    if body.scenario_type not in SCENARIOS:
        raise HTTPException(status_code=422,
                            detail=f"Unknown scenario_type. Choose one of: {sorted(SCENARIOS)}")
    return run_scenario_batch(body)


@app.get("/api/scenarios/runs")
def scenario_runs(limit: int = Query(25, ge=1, le=200)):
    rows = newest_first(store.values("scenario_runs"))[:limit]
    slim = [{k: r_.get(k) for k in ("run_id", "scenario_type", "runs", "seed", "created_at", "baseline", "omega",
                                    "comparison")} for r_ in rows]
    return {"count": len(slim), "runs": slim}


@app.get("/api/scenarios/runs/{run_id}")
def scenario_run_detail(run_id: str):
    row = store.get("scenario_runs", run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Scenario run not found")
    return row


# -----------------------------------------------------------------------------
# 5. Fleet route optimisation & energy intelligence
# -----------------------------------------------------------------------------

@app.post("/api/fleet/optimize")
def fleet_optimize(body: FleetOptimizeRequest):
    if not body.stops:
        raise HTTPException(status_code=422, detail="At least one stop is required")
    if len(body.stops) > 150:
        raise HTTPException(status_code=422, detail="A maximum of 150 stops is supported per request")
    if len({s.stop_id for s in body.stops}) != len(body.stops):
        raise HTTPException(status_code=422, detail="stop_id values must be unique")

    vehicles = body.vehicles
    if not vehicles:
        vehicles = []
        for v in store.values("vehicles"):
            if v.get("lat") is not None and v.get("lon") is not None:
                vehicles.append(FleetVehicleIn(
                    vehicle_id=v["vehicle_id"], start=GeoPoint(lat=v["lat"], lon=v["lon"]),
                    battery_pct=v.get("battery_pct", 100.0),
                    battery_capacity_kwh=v.get("battery_capacity_kwh", 75.0),
                    payload_capacity_kg=v.get("payload_capacity_kg", 1000.0),
                    curb_mass_kg=v.get("mass_kg", 2200.0),
                    base_efficiency_kwh_per_km=v.get("base_efficiency_kwh_per_km", 0.2)))
        if not vehicles:
            raise HTTPException(status_code=400,
                                detail="No vehicles supplied and no registered vehicles have lat/lon")
    if len(vehicles) > 25:
        raise HTTPException(status_code=422, detail="A maximum of 25 vehicles is supported per request")
    if len({v.vehicle_id for v in vehicles}) != len(vehicles):
        raise HTTPException(status_code=422, detail="vehicle_id values must be unique")
    return optimize_fleet(body, vehicles)


@app.get("/api/fleet/plans")
def fleet_plans(limit: int = Query(20, ge=1, le=100)):
    rows = newest_first(store.values("fleet_plans"))[:limit]
    return {"count": len(rows), "plans": [
        {k: p.get(k) for k in ("plan_id", "created_at", "objective", "solve_time_ms", "fleet_totals", "savings")}
        for p in rows]}


@app.get("/api/fleet/plans/{plan_id}")
def fleet_plan(plan_id: str):
    row = store.get("fleet_plans", plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    return row


@app.post("/api/fleet/energy/estimate")
def fleet_energy_estimate(body: EnergyEstimateRequest):
    return estimate_energy(body)


@app.get("/api/fleet/energy-intelligence")
def fleet_energy_intelligence(ambient_temp_c: Optional[float] = Query(None, ge=-50, le=60),
                              reserve_pct: float = Query(10.0, ge=0, le=50),
                              electricity_price_per_kwh: float = Query(0.15, ge=0)):
    vehicles = store.values("vehicles")
    rows: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    total_available = total_capacity = 0.0
    for v in vehicles:
        temp = ambient_temp_c if ambient_temp_c is not None else v.get("ambient_temp_c", 20.0)
        cap = v["battery_capacity_kwh"]
        available = cap * v["battery_pct"] / 100.0
        usable = max(0.0, available - cap * reserve_pct / 100.0)
        speed_kph = max(30.0, v.get("speed_mps", 0.0) * 3.6)
        per_km = leg_energy_kwh(1.0, v.get("base_efficiency_kwh_per_km", 0.2), speed_kph, temp, 1.0,
                                v.get("mass_kg", 2200.0), 0.0, v.get("grade_pct", 0.0))
        range_km = usable / per_km if per_km > 0 else None
        measured = None
        anomaly = False
        if v.get("odometer_km", 0.0) >= 5.0 and v.get("energy_used_kwh_total", 0.0) > 0:
            measured = v["energy_used_kwh_total"] / v["odometer_km"]
            anomaly = measured > v.get("base_efficiency_kwh_per_km", 0.2) * 1.25
        health = "critical" if v["battery_pct"] < 15 else "low" if v["battery_pct"] < 30 else "ok"
        if health != "ok":
            recommendations.append(f"{v['vehicle_id']}: battery {v['battery_pct']:.0f}% - schedule charging "
                                   f"({cap * (0.9 - v['battery_pct'] / 100.0):.0f} kWh to reach 90%).")
        if anomaly:
            recommendations.append(f"{v['vehicle_id']}: measured {measured:.3f} kWh/km is >25% above its rated "
                                   "efficiency; check tyre pressure, load and drivetrain.")
        total_available += available
        total_capacity += cap
        rows.append({
            "vehicle_id": v["vehicle_id"], "name": v.get("name"), "battery_pct": rnd(v["battery_pct"], 1),
            "stored_kwh": rnd(available), "usable_kwh": rnd(usable),
            "estimated_range_km": rnd(range_km, 1), "rated_kwh_per_km": rnd(v.get("base_efficiency_kwh_per_km")),
            "expected_kwh_per_km_now": rnd(per_km), "measured_kwh_per_km": rnd(measured),
            "efficiency_anomaly": anomaly, "battery_health": health,
            "cost_per_km": rnd(per_km * electricity_price_per_kwh, 3),
        })
    return {
        "vehicle_count": len(rows),
        "fleet": {
            "stored_kwh": rnd(total_available), "capacity_kwh": rnd(total_capacity),
            "fleet_soc_pct": rnd(100.0 * total_available / total_capacity, 1) if total_capacity else None,
            "low_battery_vehicles": sum(1 for r_ in rows if r_["battery_health"] != "ok"),
            "anomalies": sum(1 for r_ in rows if r_["efficiency_anomaly"]),
        },
        "vehicles": rows,
        "recommendations": recommendations,
    }


# -----------------------------------------------------------------------------
# 6. Digital Twin vs Reality gap
# -----------------------------------------------------------------------------

@app.post("/api/twin/readings", status_code=201)
def twin_ingest(body: TwinReadingIn):
    get_vehicle_or_404(body.vehicle_id)
    return process_twin_reading(body)


@app.get("/api/twin/gap")
def twin_gap_fleet():
    latest_rows = [r_ for r_ in store.values("twin_readings") if str(r_.get("reading_id", "")).startswith("latest:")]
    summary = {"SYNCED": 0, "DRIFTING": 0, "DIVERGED": 0}
    vehicles = []
    for r_ in latest_rows:
        summary[r_["status"]] = summary.get(r_["status"], 0) + 1
        vehicles.append({
            "vehicle_id": r_["vehicle_id"], "status": r_["status"], "gap_index": r_.get("gap_index"),
            "ema_gap_index": rnd(r_.get("ema_gap_index")), "trend": r_.get("trend"),
            "readings_count": r_.get("readings_count"), "last_updated": r_.get("last_updated"),
            "recalibrated_at": r_.get("recalibrated_at"),
        })
    vehicles.sort(key=lambda x: x.get("ema_gap_index") or 0, reverse=True)
    worst = vehicles[0]["vehicle_id"] if vehicles else None
    return {"monitored_vehicles": len(vehicles), "status_summary": summary, "worst_vehicle": worst,
            "vehicles": vehicles}


@app.get("/api/twin/gap/{vehicle_id}")
def twin_gap_vehicle(vehicle_id: str, limit: int = Query(50, ge=1, le=500)):
    get_vehicle_or_404(vehicle_id)
    latest = store.get("twin_readings", f"latest:{vehicle_id}")
    if latest is None:
        raise HTTPException(status_code=404, detail="No twin readings for this vehicle yet")
    history = twin_history(vehicle_id)[-limit:]
    idxs = [h["gap_index"] for h in history if h.get("gap_index") is not None]
    return {
        "vehicle_id": vehicle_id,
        "current": {**latest, "ema_gap_index": rnd(latest.get("ema_gap_index"))},
        "ai_actions_restricted": latest.get("status") == "DIVERGED",
        "stats": {
            "readings_returned": len(history),
            "mean_gap_index": rnd(mean(idxs)),
            "max_gap_index": rnd(max(idxs, default=None)),
            "pct_time_synced": rnd(100.0 * sum(1 for h in history if h.get("instant_status") == "SYNCED")
                                   / len(history), 1) if history else None,
        },
        "history": history,
    }


@app.post("/api/twin/recalibrate/{vehicle_id}")
def twin_recalibrate(vehicle_id: str, window: int = Query(20, ge=3, le=200)):
    """Estimate systematic bias from recent readings and compensate for it in future gap calculations."""
    get_vehicle_or_404(vehicle_id)
    latest = store.get("twin_readings", f"latest:{vehicle_id}")
    history = twin_history(vehicle_id)[-window:]
    if latest is None or len(history) < 3:
        raise HTTPException(status_code=409, detail="Need at least 3 readings before recalibrating")

    def avg(key: str) -> float:
        vals = [h["raw_errors"][key] for h in history if h.get("raw_errors", {}).get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    new_bias = {k: rnd(avg(k), 4) for k in ("dx", "dy", "speed", "heading", "battery", "temp")}
    latest["bias"] = new_bias
    latest["ema_gap_index"] = None
    latest["status"] = "SYNCED"
    latest["trend"] = "stable"
    latest["recalibrated_at"] = now_iso()
    store.put("twin_readings", f"latest:{vehicle_id}", latest)
    audit("TWIN_RECALIBRATED", "info", vehicle_id=vehicle_id, bias=new_bias, samples=len(history))
    return {"vehicle_id": vehicle_id, "recalibrated": True, "samples_used": len(history), "bias": new_bias,
            "note": "Bias compensation applied to subsequent readings; the gap EMA has been reset."}


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)