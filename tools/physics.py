"""Deterministic physics tools exposed to the agents via function calling.

All functions take a Snapshot plus tool arguments and return JSON-serializable
dicts. The ego is at the origin; x forward, y positive to the ego's right.
"""
import copy
import math
from typing import Dict, Optional

from .. import config
from ..risk import hj_model
from ..scenario.schema import Snapshot

# Effective collision radius by loose type keyword (point-mass miss-distance
# correction for vehicle extent).
_RADII = {"truck": 3.5, "suv": 2.8, "car": 2.5, "pedestrian": 1.0, "obstacle": 2.5}


def effective_radius(kind: str) -> float:
    k = kind.lower()
    for key, r in _RADII.items():
        if key in k:
            return r
    return 2.5


def _find_target(snapshot: Snapshot, target_id: str):
    for tid, pos, vel, kind in snapshot.targets():
        if tid == target_id:
            return pos, vel, kind
    return None


def compute_ttc(snapshot: Snapshot, target_id: str) -> Dict:
    """Closest-approach TTC between the ego (origin) and one target."""
    t = _find_target(snapshot, target_id)
    if t is None:
        return {"error": f"unknown target {target_id}"}
    pos, vel, kind = t
    ex, ey = snapshot.ego.velocity
    rvx, rvy = vel[0] - ex, vel[1] - ey
    px, py = pos
    closing = px * rvx + py * rvy
    if closing >= 0 or (abs(rvx) < 1e-6 and abs(rvy) < 1e-6):
        return {"target": target_id, "on_collision_course": False, "ttc_s": None,
                "note": "target is not closing on the ego vehicle"}
    v2 = rvx * rvx + rvy * rvy
    tca = -closing / v2
    miss = math.hypot(px + rvx * tca, py + rvy * tca)
    radius = effective_radius(kind) + 1.0  # + half ego width margin
    return {
        "target": target_id,
        "on_collision_course": miss <= radius,
        "ttc_s": round(tca, 2),
        "predicted_miss_distance_m": round(miss, 2),
        "collision_radius_m": radius,
        "current_distance_m": round(math.hypot(px, py), 2),
    }


def braking_distance(snapshot: Snapshot, decel_mps2: float = 8.0) -> Dict:
    """Stopping distance at max braking vs the gap to the nearest forward object."""
    v = snapshot.ego.velocity[0]
    dist = v * v / (2 * abs(decel_mps2))
    ahead = [(tid, pos) for tid, pos, _, _ in snapshot.targets()
             if pos[0] > 0 and abs(pos[1]) < 2.0]
    nearest = min(ahead, key=lambda t: t[1][0]) if ahead else None
    return {
        "ego_speed_mps": v,
        "decel_mps2": abs(decel_mps2),
        "stopping_distance_m": round(dist, 2),
        "nearest_in_lane_object": nearest[0] if nearest else None,
        "gap_to_it_m": round(nearest[1][0], 2) if nearest else None,
        "braking_alone_sufficient": bool(nearest is None or dist < nearest[1][0]),
    }


def lateral_clearance(snapshot: Snapshot, side: str) -> Dict:
    """Occupancy of the corridor on one side (|y| in 1..5 m, x in -5..25 m)."""
    sign = -1.0 if side == "left" else 1.0
    occupants = []
    for tid, pos, vel, kind in snapshot.targets():
        if -5.0 <= pos[0] <= 25.0 and 1.0 <= sign * pos[1] <= 5.0:
            occupants.append({"id": tid, "kind": kind,
                              "position": [round(pos[0], 1), round(pos[1], 1)],
                              "vulnerable": "pedestrian" in kind.lower()})
    ego = snapshot.ego
    boundary = ego.road_boundary_left if side == "left" else ego.road_boundary_right
    boundary_margin = None if boundary is None else round(abs(boundary) - 1.0, 2)
    lane_change_possible = (
        not occupants and (boundary_margin is None or boundary_margin >= 3.0)
    )
    return {"side": side, "occupants": occupants,
            "boundary_margin_m": boundary_margin,
            "lane_change_possible": lane_change_possible}


def evidence_pack(snapshot: Snapshot) -> Dict:
    """Precomputed physics evidence for single-shot deliberation.

    In the realtime profile the decision agent gets every number it could
    have requested through function calling — TTC per target, braking
    feasibility, both corridors, and a rollout of ALL eight maneuver codes —
    embedded directly in its prompt. This removes 3-6 LLM tool round-trips
    from the deliberation loop at the cost of ~0.1 s of local computation.
    """
    return {
        "ttc_per_target": {tid: compute_ttc(snapshot, tid)
                           for tid, _, _, _ in snapshot.targets()},
        "braking": braking_distance(snapshot),
        "corridor_left": lateral_clearance(snapshot, "left"),
        "corridor_right": lateral_clearance(snapshot, "right"),
        "maneuver_rollouts": {code: rollout_maneuver(snapshot, code)
                              for code in _PROFILES},
    }


def project_snapshot(snapshot: Snapshot, t_forward: float) -> Snapshot:
    """Constant-velocity projection of the scene t_forward seconds ahead, in
    the ego frame (targets drift by their velocity relative to the ego).

    Deliberation happens in the anticipation phase, but the armed policy will
    execute at the trigger instant — so maneuvers must be validated against
    the ANTICIPATED execution state, not the current one.
    """
    proj = copy.deepcopy(snapshot)
    ex, ey = snapshot.ego.velocity
    for o in proj.obstacles:
        o.center = (o.center[0] - ex * t_forward, o.center[1] - ey * t_forward)
    for p in proj.participants:
        p.coordinate = (p.coordinate[0] + (p.velocity[0] - ex) * t_forward,
                        p.coordinate[1] + (p.velocity[1] - ey) * t_forward)
    proj.t = snapshot.t + t_forward
    return proj


# Maneuver kinematic profiles: (longitudinal decel, total lateral shift, note)
_PROFILES = {
    0: (8.0, 0.0, "straight-line full braking"),
    1: (0.0, -3.5, "sharp left lane change"),
    2: (0.0, +3.5, "sharp right lane change"),
    3: (4.0, -3.5, "left lane change with braking"),
    4: (4.0, +3.5, "right lane change with braking"),
    5: (6.0, -1.0, "T-drift, nose left (rear faces right)"),
    6: (6.0, +1.0, "T-drift, nose right (rear faces left)"),
    7: (0.0, 0.0, "no intervention"),
}


def _propagate(snapshot: Snapshot, code: int,
               target_speed_mps: Optional[float] = None) -> Dict:
    """Roll one maneuver profile forward; ego frame, targets constant-velocity."""
    decel, dy_total, note = _PROFILES[code]
    horizon, dt = config.ROLLOUT_HORIZON, 0.05
    steps = int(horizon / dt)
    ex, ey = 0.0, 0.0
    evx = snapshot.ego.velocity[0]
    floor = target_speed_mps if (target_speed_mps and code in (3, 4)) else 0.0

    targets = [(tid, pos, vel, kind) for tid, pos, vel, kind in snapshot.targets()]
    min_sep, min_sep_target, min_sep_t = float("inf"), None, 0.0
    for i in range(1, steps + 1):
        t = i * dt
        evx = max(floor, evx - decel * dt)
        ex += evx * dt
        ey = dy_total * min(1.0, t / horizon)  # smooth lateral ramp
        for tid, pos, vel, kind in targets:
            tx, ty = pos[0] + vel[0] * t, pos[1] + vel[1] * t
            sep = math.hypot(tx - ex, ty - ey) - effective_radius(kind)
            if sep < min_sep:
                min_sep, min_sep_target, min_sep_t = sep, tid, t

    # End-state HJ risk per target. The value net was trained on forward
    # obstacles, so a target that ends up behind the ego AND is either in a
    # different lane or no longer closing gets the rear discount — it cannot
    # be the target this maneuver collides with.
    end_risks = {}
    for tid, pos, vel, kind in targets:
        tx, ty = pos[0] + vel[0] * horizon, pos[1] + vel[1] * horizon
        risk = hj_model.hj_risk((ex, ey), evx, (tx, ty))
        rel_x, rel_y = tx - ex, ty - ey
        behind = rel_x < 0
        different_lane = abs(rel_y) > 2.5
        closing = (vel[0] - evx) * (1 if behind else -1) > 0  # toward the ego
        if behind and (different_lane or not closing):
            risk = round(risk * config.REAR_HJ_DISCOUNT, 2)
        end_risks[tid] = risk
    worst = max(end_risks.values()) if end_risks else 0.0
    return {"note": note, "min_sep": min_sep, "min_sep_target": min_sep_target,
            "min_sep_t": min_sep_t, "end_speed": evx, "end_lateral": ey,
            "end_risks": end_risks, "worst": worst}


def rollout_maneuver(snapshot: Snapshot, code: int,
                     target_speed_mps: Optional[float] = None) -> Dict:
    """Coarse kinematic rollout of a maneuver over the shield horizon.

    Returns the minimum separation along the trajectory and the HJ risk of
    the end state, compared against the no-intervention baseline (code 7).
    A maneuver's HJ evidence is acceptable if its worst end-state risk is
    below the absolute gate OR does not worsen the baseline — in an
    emergency every state is inside a risky set, so the shield's question is
    "does this maneuver improve reachability, and keep physical separation?".
    """
    if code not in _PROFILES:
        return {"error": f"unknown maneuver code {code}"}
    run = _propagate(snapshot, code, target_speed_mps)
    baseline = run if code == 7 else _propagate(snapshot, 7)
    hj_acceptable = (run["worst"] < config.HJ_SAFE_RISK
                     or run["worst"] <= baseline["worst"] + 0.05)
    return {
        "code": code, "maneuver": run["note"],
        "min_separation_m": round(run["min_sep"], 2),
        "closest_target": run["min_sep_target"],
        "time_of_closest_m": round(run["min_sep_t"], 2),
        "end_speed_mps": round(run["end_speed"], 2),
        "end_lateral_offset_m": round(run["end_lateral"], 2),
        "end_state_hj_risk_per_target": run["end_risks"],
        "worst_end_hj_risk": run["worst"],
        "baseline_no_intervention_hj_risk": baseline["worst"],
        "hj_acceptable": hj_acceptable,
    }
