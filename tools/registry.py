"""Builds the per-deliberation ToolRegistry bound to a snapshot + case store."""
from ..llm import ToolRegistry
from ..memory.case_store import CaseStore
from ..risk import hj_model
from ..scenario.schema import Snapshot
from . import physics


def build_registry(snapshot: Snapshot, case_store: CaseStore) -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        "compute_ttc",
        "Time-to-closest-approach, predicted miss distance and collision-course "
        "flag between the ego vehicle and one target.",
        {"type": "object",
         "properties": {"target_id": {"type": "string"}},
         "required": ["target_id"]},
        lambda target_id: physics.compute_ttc(snapshot, target_id),
    )
    reg.register(
        "braking_distance",
        "Stopping distance at a given deceleration vs the gap to the nearest "
        "in-lane forward object; says whether braking alone avoids collision.",
        {"type": "object",
         "properties": {"decel_mps2": {"type": "number",
                                       "description": "braking deceleration, default 8"}},
         "required": []},
        lambda decel_mps2=8.0: physics.braking_distance(snapshot, decel_mps2),
    )
    reg.register(
        "lateral_clearance",
        "Whether the corridor on one side of the ego is free for a lane change: "
        "occupants (incl. pedestrians) and road-boundary margin.",
        {"type": "object",
         "properties": {"side": {"type": "string", "enum": ["left", "right"]}},
         "required": ["side"]},
        lambda side: physics.lateral_clearance(snapshot, side),
    )
    reg.register(
        "rollout_maneuver",
        "Kinematically roll out a maneuver code for 1 s and report minimum "
        "separation, end speed, and HJ-reachability risk of the end state. Use "
        "this to compare candidate maneuvers with physical evidence.",
        {"type": "object",
         "properties": {"code": {"type": "integer", "minimum": 0, "maximum": 7},
                        "target_speed_mps": {"type": "number"}},
         "required": ["code"]},
        lambda code, target_speed_mps=None:
            physics.rollout_maneuver(snapshot, code, target_speed_mps),
    )
    reg.register(
        "query_hj_risk",
        "Current HJ-reachability risk [0..1] of the ego state w.r.t. one target, "
        "plus the target's vulnerability weight.",
        {"type": "object",
         "properties": {"target_id": {"type": "string"}},
         "required": ["target_id"]},
        lambda target_id: _query_hj(snapshot, target_id),
    )
    reg.register(
        "retrieve_similar_cases",
        "Retrieve past collision-avoidance cases similar to the current scene, "
        "with their decisions, outcomes and distilled lessons.",
        {"type": "object",
         "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 5}},
         "required": []},
        lambda k=2: {"cases": case_store.retrieve(snapshot.compact_text(), k)},
    )
    return reg


def _query_hj(snapshot: Snapshot, target_id: str) -> dict:
    for tid, pos, vel, kind in snapshot.targets():
        if tid == target_id:
            return {
                "target": tid,
                "hj_risk": hj_model.hj_risk((0.0, 0.0), snapshot.ego.velocity[0], pos),
                "vulnerability_weight": hj_model.participant_type_risk(kind),
            }
    return {"error": f"unknown target {target_id}"}
