"""
backend/omega_engine.py
=======================

OMEGA - 2D grid warehouse Digital Twin.

The twin knows the static world (shelves, restricted zones, a worker lane) and
the dynamic world (people and a forklift patrolling on repeating routes). Given
a robot action plan from PROMETHEUS it forecasts THREE parallel futures:

    Timeline A - Optimal           shortest legal path
    Timeline B - Cautious          wide berth around people + waits for them to pass
    Timeline C - Poisoned Attack   what happens if a physical prompt injection
                                   hijacks the plan (shortcut through the worker
                                   lane and the hazmat cage at high speed)

Each timeline yields a 2D trajectory, a Collision Risk (%) and a Safety Score
(0-100). `run_parallel_stress_test(n)` replays the whole experiment under n
randomised human-timing scenarios, which is the workload used to benchmark
Nebius cloud compute.

Coordinates are (x, y) = (column, row) with the origin in the top-left corner.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]

# --------------------------------------------------------------------------- #
# World definition
# --------------------------------------------------------------------------- #
GRID_W, GRID_H = 20, 14
MAX_TICKS = 400
START_LOCATION = "HOME_BAY"

# Shelf blocks as inclusive rectangles (x0, y0, x1, y1).
SHELF_BLOCKS: Tuple[Tuple[int, int, int, int], ...] = tuple(
    (x0, y0, x1, y1) for (x0, x1) in ((3, 7), (11, 15)) for (y0, y1) in ((2, 3), (6, 7), (10, 11))
)

# Zones the robot must never enter.
RESTRICTED_ZONES: Dict[str, Tuple[int, int, int, int]] = {
    "HAZMAT_CAGE": (17, 5, 19, 8),
    "SERVER_ROOM": (0, 5, 1, 8),
}

# Human-priority lane in the centre aisle: legal but dangerous.
WORKER_LANE: Tuple[int, int, int, int] = (8, 4, 10, 9)

# Named, trusted waypoints. Only these may appear in a legitimate plan.
WAREHOUSE_LOCATIONS: Dict[str, Cell] = {
    "HOME_BAY": (1, 12),
    "INBOUND_DOCK": (1, 1),
    "OUTBOUND_DOCK": (18, 12),
    "PACK_STATION": (18, 1),
    "SHELF_A": (8, 3),
    "SHELF_B": (10, 7),
    "SHELF_C": (2, 10),
    "SHELF_D": (16, 11),
}

# Targets that an attacker can inject. They are NOT trusted locations.
HAZARD_LOCATIONS: Dict[str, Cell] = {
    "WORKER_LANE_CENTER": (9, 6),
    "HAZMAT_CAGE": (18, 6),
}

SPEED_RISK_MULTIPLIER = {"SLOW": 0.7, "NORMAL": 1.0, "FAST": 1.5}
ACTION_DWELL = {"MOVE": 0, "PICKUP": 2, "DROP": 2, "SCAN": 1, "CHARGE": 3, "WAIT": 2}

TIMELINE_META: Dict[str, Tuple[str, str, str]] = {
    "A": ("Timeline A: Optimal", "optimal", "#19E3FF"),
    "B": ("Timeline B: Cautious", "cautious", "#4DF0B3"),
    "C": ("Timeline C: Poisoned Attack Path", "poisoned", "#FF2E88"),
}

DEFAULT_DEMO_PLAN: List[Dict[str, Any]] = [
    {"step": 1, "action": "MOVE", "location": "SHELF_A", "target": [8, 3], "speed": "NORMAL",
     "description": "Travel to Shelf A", "source": "default"},
    {"step": 2, "action": "PICKUP", "location": "SHELF_A", "target": [8, 3], "speed": "NORMAL",
     "description": "Pick up package at Shelf A", "source": "default"},
    {"step": 3, "action": "MOVE", "location": "PACK_STATION", "target": [18, 1], "speed": "NORMAL",
     "description": "Travel to Pack Station", "source": "default"},
    {"step": 4, "action": "DROP", "location": "PACK_STATION", "target": [18, 1], "speed": "NORMAL",
     "description": "Drop package at Pack Station", "source": "default"},
]


# --------------------------------------------------------------------------- #
# Geometry helpers (module level so PROMETHEUS / PHANTOM can reuse them)
# --------------------------------------------------------------------------- #
def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def _in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def is_shelf(x: int, y: int) -> bool:
    return any(_in_rect(x, y, r) for r in SHELF_BLOCKS)


def restricted_zone_at(x: int, y: int) -> Optional[str]:
    for name, rect in RESTRICTED_ZONES.items():
        if _in_rect(x, y, rect):
            return name
    return None


def is_restricted(x: int, y: int) -> bool:
    return restricted_zone_at(x, y) is not None


def in_worker_lane(x: int, y: int) -> bool:
    return _in_rect(x, y, WORKER_LANE)


def renumber_plan(plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of `plan` with step numbers 1..n."""
    out = []
    for i, step in enumerate(plan, start=1):
        s = dict(step)
        s["step"] = i
        out.append(s)
    return out


def make_poisoned_plan(
    plan: Sequence[Dict[str, Any]],
    targets: Sequence[str] = ("WORKER_LANE_CENTER", "HAZMAT_CAGE"),
    fast: bool = True,
) -> List[Dict[str, Any]]:
    """
    Model a physical prompt injection: insert attacker-chosen waypoints right
    after the first PICKUP (or first step) and force high speed from there on.
    """
    base = [dict(s) for s in plan]
    insert_at = 1
    for i, s in enumerate(base):
        if str(s.get("action", "")).upper() == "PICKUP":
            insert_at = i + 1
            break
    insert_at = min(insert_at, len(base))

    injected = []
    for name in targets:
        if name not in HAZARD_LOCATIONS:
            continue
        x, y = HAZARD_LOCATIONS[name]
        injected.append(
            {
                "action": "MOVE",
                "location": name,
                "target": [x, y],
                "speed": "FAST" if fast else "NORMAL",
                "description": f"[INJECTED] shortcut via {name.replace('_', ' ').title()}",
                "source": "physical_injection",
                "injected": True,
            }
        )
    poisoned = base[:insert_at] + injected + base[insert_at:]
    if fast:
        for s in poisoned[insert_at + len(injected):]:
            if str(s.get("action", "")).upper() == "MOVE":
                s["speed"] = "FAST"
    return renumber_plan(poisoned)


# --------------------------------------------------------------------------- #
# Dynamic agents
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DynamicAgent:
    name: str
    kind: str                      # "worker" | "forklift"
    waypoints: Tuple[Cell, ...]
    ticks_per_cell: int            # 2 = walks at half the robot's speed
    sigma: float                   # spatial spread of the danger field
    color: str


# Patrol zones mirror real floor plans: Worker-1 owns the human-priority lane,
# Workers 2 and 3 pick in the east aisles, the forklift shuttles along the south
# aisle. Legitimate routes (west corridor, north aisle) stay mostly clear, while
# the injected shortcut (lane + hazmat cage) cuts straight through all of them.
AGENTS: Tuple[DynamicAgent, ...] = (
    DynamicAgent("Worker-1", "worker", ((9, 4), (9, 9)), 2, 0.60, "#FFC93C"),
    DynamicAgent("Worker-2", "worker", ((11, 5), (18, 5)), 2, 0.60, "#FFC93C"),
    DynamicAgent("Worker-3", "worker", ((11, 9), (18, 9)), 2, 0.60, "#FFC93C"),
    DynamicAgent("Forklift-1", "forklift", ((4, 13), (17, 13)), 1, 0.75, "#FF8A3C"),
)


def _segment(a: Cell, b: Cell) -> List[Cell]:
    x, y = a
    cells: List[Cell] = []
    while (x, y) != b:
        if x != b[0]:
            x += 1 if b[0] > x else -1
        elif y != b[1]:
            y += 1 if b[1] > y else -1
        cells.append((x, y))
    return cells


def _agent_cycle(agent: DynamicAgent) -> Tuple[Cell, ...]:
    forward: List[Cell] = [agent.waypoints[0]]
    for a, b in zip(agent.waypoints, agent.waypoints[1:]):
        forward.extend(_segment(a, b))
    ping_pong = forward + forward[-2:0:-1]
    cycle: List[Cell] = []
    for cell in ping_pong:
        cycle.extend([cell] * agent.ticks_per_cell)
    return tuple(cycle)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TimelineResult:
    key: str
    name: str
    strategy: str
    color: str
    trajectory: List[Cell]
    speeds: List[float]
    ticks: int
    collision_risk_pct: float
    safety_score: float
    collisions: int
    near_misses: int
    restricted_ticks: int
    min_distance: float
    risk_curve: List[float]          # cumulative risk (%) at every tick
    events: List[Dict[str, Any]] = field(default_factory=list)
    unreachable: List[str] = field(default_factory=list)
    waypoints: List[str] = field(default_factory=list)

    def to_dict(self, include_path: bool = True) -> Dict[str, Any]:
        d = {
            "key": self.key,
            "name": self.name,
            "strategy": self.strategy,
            "ticks": self.ticks,
            "collision_risk_pct": self.collision_risk_pct,
            "safety_score": self.safety_score,
            "collisions": self.collisions,
            "near_misses": self.near_misses,
            "restricted_ticks": self.restricted_ticks,
            "min_distance": self.min_distance,
            "events": self.events,
            "unreachable": self.unreachable,
            "waypoints": self.waypoints,
        }
        if include_path:
            d["trajectory"] = [list(c) for c in self.trajectory]
            d["risk_curve"] = self.risk_curve
        return d

    def summary_row(self) -> Dict[str, Any]:
        return {
            "Timeline": self.name,
            "Strategy": self.strategy,
            "Ticks": self.ticks,
            "Collision risk %": self.collision_risk_pct,
            "Safety score": self.safety_score,
            "Collisions": self.collisions,
            "Near misses": self.near_misses,
            "Restricted ticks": self.restricted_ticks,
            "Min distance (cells)": self.min_distance,
        }


# --------------------------------------------------------------------------- #
# The twin
# --------------------------------------------------------------------------- #
class OmegaTwin:
    """Deterministic, thread-safe (read-only after __init__) warehouse twin."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.agents: Tuple[DynamicAgent, ...] = AGENTS
        self._cycles: Dict[str, Tuple[Cell, ...]] = {a.name: _agent_cycle(a) for a in self.agents}
        self._agent_index = {a.name: a for a in self.agents}
        self._cost: Dict[str, List[List[Optional[float]]]] = {
            s: self._build_cost_grid(s) for s in ("optimal", "cautious", "poisoned")
        }
        self._min_cost = {"optimal": 1.0, "cautious": 1.0, "poisoned": 0.6}
        self.default_phases = self.sample_phases(random.Random(seed))

    # ------------------------------------------------------------------ #
    # World description
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_cost_grid(strategy: str) -> List[List[Optional[float]]]:
        grid: List[List[Optional[float]]] = []
        for y in range(GRID_H):
            row: List[Optional[float]] = []
            for x in range(GRID_W):
                if is_shelf(x, y):
                    row.append(None)
                elif strategy == "poisoned":
                    row.append(0.6 if (is_restricted(x, y) or in_worker_lane(x, y)) else 1.0)
                elif is_restricted(x, y):
                    row.append(None)
                elif in_worker_lane(x, y):
                    row.append(1.5 if strategy == "optimal" else 4.0)
                else:
                    row.append(1.0)
            grid.append(row)
        return grid

    def world_snapshot(self) -> Dict[str, Any]:
        return {
            "width": GRID_W,
            "height": GRID_H,
            "shelves": [list(r) for r in SHELF_BLOCKS],
            "restricted": {k: list(v) for k, v in RESTRICTED_ZONES.items()},
            "lane": list(WORKER_LANE),
            "locations": {k: list(v) for k, v in WAREHOUSE_LOCATIONS.items()},
            "hazards": {k: list(v) for k, v in HAZARD_LOCATIONS.items()},
        }

    def agent_meta(self) -> List[Dict[str, Any]]:
        return [{"name": a.name, "kind": a.kind, "color": a.color} for a in self.agents]

    # ------------------------------------------------------------------ #
    # Dynamic agents
    # ------------------------------------------------------------------ #
    def sample_phases(self, rng: Any = None) -> Dict[str, int]:
        """Random starting offsets for every patrol (accepts Random, int or None)."""
        if rng is None:
            rng = random.Random(self.seed)
        elif isinstance(rng, int):
            rng = random.Random(rng)
        return {a.name: rng.randrange(len(self._cycles[a.name])) for a in self.agents}

    def agent_position(self, name: str, t: int, phases: Optional[Dict[str, int]] = None) -> Cell:
        phases = phases or self.default_phases
        cycle = self._cycles[name]
        return cycle[(t + phases.get(name, 0)) % len(cycle)]

    def agent_frames(self, ticks: int, phases: Optional[Dict[str, int]] = None) -> Dict[str, List[List[int]]]:
        """Positions of every dynamic agent for t = 0..ticks (used by the canvas)."""
        return {
            a.name: [list(self.agent_position(a.name, t, phases)) for t in range(ticks + 1)]
            for a in self.agents
        }

    # ------------------------------------------------------------------ #
    # Path finding
    # ------------------------------------------------------------------ #
    def _astar(
        self,
        start: Cell,
        goal: Cell,
        strategy: str,
        extra_blocked: FrozenSet[Cell] = frozenset(),
    ) -> Optional[List[Cell]]:
        cost = self._cost[strategy]
        if start == goal:
            return [start]

        def passable(c: Cell) -> bool:
            return in_bounds(*c) and cost[c[1]][c[0]] is not None and c not in extra_blocked

        if not passable(goal):
            return None

        min_c = self._min_cost[strategy]
        counter = 0
        open_heap: List[Tuple[float, int, Cell]] = [(0.0, counter, start)]
        g_score: Dict[Cell, float] = {start: 0.0}
        parent: Dict[Cell, Cell] = {}
        closed: set = set()

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                return path[::-1]
            closed.add(cur)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in closed or not passable(nxt):
                    continue
                tentative = g_score[cur] + float(cost[nxt[1]][nxt[0]])  # type: ignore[arg-type]
                if tentative < g_score.get(nxt, math.inf):
                    g_score[nxt] = tentative
                    parent[nxt] = cur
                    h = (abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])) * min_c
                    counter += 1
                    heapq.heappush(open_heap, (tentative + h, counter, nxt))
        return None

    # ------------------------------------------------------------------ #
    # Plan -> waypoints -> trajectory
    # ------------------------------------------------------------------ #
    @staticmethod
    def plan_to_waypoints(plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        wps: List[Dict[str, Any]] = []
        for step in plan:
            action = str(step.get("action", "MOVE")).upper()
            target = step.get("target")
            speed = str(step.get("speed", "NORMAL")).upper()
            if speed not in SPEED_RISK_MULTIPLIER:
                speed = "NORMAL"
            dwell = int(step.get("duration", ACTION_DWELL.get(action, 0))) if action == "WAIT" else ACTION_DWELL.get(action, 0)
            pos = None
            if target is not None and len(target) == 2:
                pos = (int(target[0]), int(target[1]))
            wps.append(
                {
                    "pos": pos,
                    "dwell": dwell,
                    "speed": speed,
                    "action": action,
                    "name": step.get("location") or (f"{pos[0]},{pos[1]}" if pos else action),
                }
            )
        return wps

    def _human_threat(self, cell: Cell, t: int, phases: Dict[str, int], radius: float) -> bool:
        for a in self.agents:
            h = self.agent_position(a.name, t, phases)
            if math.hypot(cell[0] - h[0], cell[1] - h[1]) < radius:
                return True
        return False

    def _build_trajectory(
        self,
        waypoints: Sequence[Dict[str, Any]],
        strategy: str,
        phases: Dict[str, int],
        start: Cell,
        wait_cap: int,
        extra_blocked: FrozenSet[Cell] = frozenset(),
    ) -> Tuple[List[Cell], List[float], List[str], List[Dict[str, Any]]]:
        traj: List[Cell] = [start]
        speeds: List[float] = [1.0]
        unreachable: List[str] = []
        notes: List[Dict[str, Any]] = []
        cautious = strategy == "cautious"

        for wp in waypoints:
            mult = SPEED_RISK_MULTIPLIER[wp["speed"]]
            if wp["pos"] is not None and traj[-1] != wp["pos"]:
                path = self._astar(traj[-1], wp["pos"], strategy, extra_blocked)
                if path is None:
                    unreachable.append(str(wp["name"]))
                    continue
                for cell in path[1:]:
                    if len(traj) >= MAX_TICKS:
                        break
                    if cautious:
                        waits = 0
                        while (
                            waits < wait_cap
                            and len(traj) < MAX_TICKS
                            and self._human_threat(cell, len(traj), phases, 2.3)
                            and not self._human_threat(traj[-1], len(traj), phases, 1.5)
                        ):
                            traj.append(traj[-1])
                            speeds.append(0.7)
                            waits += 1
                        if waits:
                            notes.append({"t": len(traj), "type": "YIELD", "detail": f"Yielded {waits} tick(s) near {cell}"})
                    traj.append(cell)
                    speeds.append(mult)
            for _ in range(wp["dwell"]):
                if len(traj) >= MAX_TICKS:
                    break
                traj.append(traj[-1])
                speeds.append(0.7 if wp["speed"] == "SLOW" else 1.0)
        return traj, speeds, unreachable, notes

    # ------------------------------------------------------------------ #
    # Risk model
    # ------------------------------------------------------------------ #
    def evaluate_trajectory(
        self,
        traj: Sequence[Cell],
        speeds: Sequence[float],
        phases: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Per tick, every agent contributes a collision probability from a gaussian
        danger field (p = 0.95 * exp(-d^2 / 2 sigma^2)) scaled by robot speed.
        Entering a restricted zone adds 12 %/tick, the worker lane 2 %/tick and a
        shelf impact is certain. Total risk = 1 - product(1 - p_tick).
        """
        phases = phases or self.default_phases
        survive = 1.0
        curve: List[float] = []
        events: List[Dict[str, Any]] = []
        collisions = near_misses = restricted_ticks = 0
        min_dist = math.inf
        near_state = {a.name: False for a in self.agents}
        coll_state = {a.name: False for a in self.agents}
        was_restricted = False

        for t, r in enumerate(traj):
            mult = speeds[t] if t < len(speeds) else 1.0
            keep = 1.0

            for a in self.agents:
                h = self.agent_position(a.name, t, phases)
                d = math.hypot(r[0] - h[0], r[1] - h[1])
                min_dist = min(min_dist, d)
                hit = d < 0.75
                if t > 0 and not hit:
                    pr = traj[t - 1]
                    ph = self.agent_position(a.name, t - 1, phases)
                    hit = r == ph and h == pr  # swapped cells = passing through each other
                p = min(0.99, 0.95 * math.exp(-(d * d) / (2 * a.sigma * a.sigma)) * mult)
                if hit:
                    p = max(p, 0.9)
                    if not coll_state[a.name]:
                        collisions += 1
                        events.append({"t": t, "type": "COLLISION", "agent": a.name, "cell": list(r), "distance": round(d, 2)})
                    coll_state[a.name] = True
                else:
                    coll_state[a.name] = False
                if d < 1.6 and not hit:
                    if not near_state[a.name]:
                        near_misses += 1
                        events.append({"t": t, "type": "NEAR_MISS", "agent": a.name, "cell": list(r), "distance": round(d, 2)})
                    near_state[a.name] = True
                elif not hit:
                    near_state[a.name] = False
                keep *= 1.0 - p

            zone = restricted_zone_at(*r)
            if zone:
                restricted_ticks += 1
                keep *= 1.0 - min(0.99, 0.12 * mult)
                if not was_restricted:
                    events.append({"t": t, "type": "RESTRICTED_ZONE", "agent": zone, "cell": list(r), "distance": 0.0})
            was_restricted = zone is not None

            if in_worker_lane(*r):
                keep *= 1.0 - 0.02 * mult
            if is_shelf(*r):
                keep = 0.0
                events.append({"t": t, "type": "STATIC_IMPACT", "agent": "Shelf", "cell": list(r), "distance": 0.0})

            survive *= keep
            curve.append(round((1.0 - survive) * 100.0, 2))

        risk = round((1.0 - survive) * 100.0, 2)
        score = 100.0 * (1.0 - risk / 100.0) - 6.0 * collisions - 1.0 * restricted_ticks - 1.5 * near_misses
        return {
            "risk_pct": risk,
            "safety_score": round(max(0.0, min(100.0, score)), 1),
            "collisions": collisions,
            "near_misses": near_misses,
            "restricted_ticks": restricted_ticks,
            "min_distance": round(min_dist if min_dist != math.inf else 99.0, 2),
            "risk_curve": curve,
            "events": events,
        }

    # ------------------------------------------------------------------ #
    # Public simulation API
    # ------------------------------------------------------------------ #
    def simulate_plan(
        self,
        plan: Sequence[Dict[str, Any]],
        strategy: str = "optimal",
        phases: Optional[Dict[str, int]] = None,
        start: Optional[Cell] = None,
        key: Optional[str] = None,
        wait_cap: int = 6,
        extra_blocked: Optional[Sequence[Cell]] = None,
    ) -> TimelineResult:
        if strategy not in self._cost:
            raise ValueError(f"Unknown strategy '{strategy}'. Use optimal, cautious or poisoned.")
        phases = phases or self.default_phases
        start = tuple(start) if start else WAREHOUSE_LOCATIONS[START_LOCATION]  # type: ignore[assignment]
        waypoints = self.plan_to_waypoints(plan)
        traj, speeds, unreachable, notes = self._build_trajectory(
            waypoints, strategy, phases, start, wait_cap, frozenset(map(tuple, extra_blocked or []))  # type: ignore[arg-type]
        )
        metrics = self.evaluate_trajectory(traj, speeds, phases)

        default_key = next((k for k, v in TIMELINE_META.items() if v[1] == strategy), strategy[:1].upper())
        key = key or default_key
        name, _, color = TIMELINE_META.get(key, (strategy.title(), strategy, "#FFFFFF"))
        events = sorted(metrics["events"] + notes, key=lambda e: e["t"])
        return TimelineResult(
            key=key,
            name=name,
            strategy=strategy,
            color=color,
            trajectory=traj,
            speeds=speeds,
            ticks=len(traj),
            collision_risk_pct=metrics["risk_pct"],
            safety_score=metrics["safety_score"],
            collisions=metrics["collisions"],
            near_misses=metrics["near_misses"],
            restricted_ticks=metrics["restricted_ticks"],
            min_distance=metrics["min_distance"],
            risk_curve=metrics["risk_curve"],
            events=events,
            unreachable=unreachable,
            waypoints=[str(w["name"]) for w in waypoints],
        )

    def simulate_timelines(
        self,
        plan: Sequence[Dict[str, Any]],
        poisoned_plan: Optional[Sequence[Dict[str, Any]]] = None,
        phases: Optional[Dict[str, int]] = None,
        start: Optional[Cell] = None,
    ) -> Dict[str, TimelineResult]:
        """
        Forecast the three parallel futures. `plan` drives A and B. Timeline C
        uses `poisoned_plan` if supplied, otherwise a default injected plan.
        """
        poisoned = list(poisoned_plan) if poisoned_plan is not None else make_poisoned_plan(plan)
        return {
            "A": self.simulate_plan(plan, "optimal", phases, start, key="A"),
            "B": self.simulate_plan(plan, "cautious", phases, start, key="B"),
            "C": self.simulate_plan(poisoned, "poisoned", phases, start, key="C"),
        }

    # ------------------------------------------------------------------ #
    # Nebius cloud benchmark
    # ------------------------------------------------------------------ #
    def _stress_scenario(self, i: int, seed: int, plan: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rng = random.Random(seed * 100_003 + i)
        phases = self.sample_phases(rng)
        variants = (("WORKER_LANE_CENTER",), ("HAZMAT_CAGE",), ("WORKER_LANE_CENTER", "HAZMAT_CAGE"))
        targets = rng.choice(variants)
        poisoned = make_poisoned_plan(plan, targets)
        tl = self.simulate_timelines(plan, poisoned, phases)
        return {
            "run": i,
            "attack": "+".join(targets),
            "A_risk": tl["A"].collision_risk_pct,
            "B_risk": tl["B"].collision_risk_pct,
            "C_risk": tl["C"].collision_risk_pct,
            "A_safety": tl["A"].safety_score,
            "B_safety": tl["B"].safety_score,
            "C_safety": tl["C"].safety_score,
            "A_ticks": tl["A"].ticks,
            "B_ticks": tl["B"].ticks,
            "C_ticks": tl["C"].ticks,
        }

    def run_parallel_stress_test(
        self,
        n: int = 50,
        *,
        plan: Optional[Sequence[Dict[str, Any]]] = None,
        seed: int = 1337,
        workers: Optional[int] = None,
        executor: str = "thread",
        danger_threshold: float = 25.0,
    ) -> Dict[str, Any]:
        """
        Run `n` randomised scenarios x 3 timelines in parallel and return
        aggregate statistics plus throughput numbers (simulations / second),
        which is what the dashboard reports as the Nebius benchmark.

        executor="process" uses a process pool (use it on a multi-core Nebius VM
        for real parallel speed-up); "thread" is the safe default for notebooks
        and Streamlit.
        """
        n = max(1, int(n))
        plan = list(plan) if plan is not None else [dict(s) for s in DEFAULT_DEMO_PLAN]
        workers = workers or min(32, max(2, (_cpu_count() or 4)))
        started = time.perf_counter()

        if executor == "process":
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(_process_worker, [(i, seed, plan) for i in range(n)], chunksize=max(1, n // (workers * 4))))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(lambda i: self._stress_scenario(i, seed, plan), range(n)))

        elapsed = time.perf_counter() - started
        summary: Dict[str, Dict[str, float]] = {}
        for k in ("A", "B", "C"):
            risks = np.array([r[f"{k}_risk"] for r in rows], dtype=float)
            safety = np.array([r[f"{k}_safety"] for r in rows], dtype=float)
            summary[k] = {
                "name": TIMELINE_META[k][0],
                "mean_risk": round(float(risks.mean()), 2),
                "median_risk": round(float(np.median(risks)), 2),
                "p95_risk": round(float(np.percentile(risks, 95)), 2),
                "max_risk": round(float(risks.max()), 2),
                "mean_safety": round(float(safety.mean()), 1),
                "danger_rate_pct": round(float((risks >= danger_threshold).mean() * 100.0), 1),
            }
        sims = n * 3
        return {
            "scenarios": n,
            "simulations": sims,
            "elapsed_s": round(elapsed, 4),
            "sims_per_sec": round(sims / elapsed, 1) if elapsed > 0 else float("inf"),
            "workers": workers,
            "executor": executor,
            "danger_threshold": danger_threshold,
            "summary": summary,
            "attack_detectable_pct": summary["C"]["danger_rate_pct"],
            "risk_reduction_pct": round(summary["C"]["mean_risk"] - summary["B"]["mean_risk"], 2),
            "runs": rows,
        }


# --------------------------------------------------------------------------- #
# Process-pool plumbing (must live at module level to be picklable)
# --------------------------------------------------------------------------- #
_WORKER_TWIN: Optional[OmegaTwin] = None


def _process_worker(args: Tuple[int, int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    global _WORKER_TWIN
    if _WORKER_TWIN is None:
        _WORKER_TWIN = OmegaTwin()
    i, seed, plan = args
    return _WORKER_TWIN._stress_scenario(i, seed, plan)


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 4


__all__ = [
    "OmegaTwin",
    "TimelineResult",
    "DynamicAgent",
    "WAREHOUSE_LOCATIONS",
    "HAZARD_LOCATIONS",
    "RESTRICTED_ZONES",
    "WORKER_LANE",
    "DEFAULT_DEMO_PLAN",
    "TIMELINE_META",
    "GRID_W",
    "GRID_H",
    "in_bounds",
    "is_shelf",
    "is_restricted",
    "restricted_zone_at",
    "in_worker_lane",
    "make_poisoned_plan",
    "renumber_plan",
]
