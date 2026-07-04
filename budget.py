"""Reflex-loop scene assessment and TTC-driven cognition budgeting.

`assess_scene` is the pure-computation risk monitor (HJ value network + TTC
urgency); `budget_from` maps the remaining time window onto a cognition
budget — which model tier deliberates, whether vision and debate are
affordable, and the hard deadline for the deliberation loop. This is the
"anytime" knob of the architecture: less time buys less cognition, never
less safety (the reflex fallback is unconditional).
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional

from . import config
from .risk import hj_model
from .scenario.schema import Snapshot
from .tools.physics import effective_radius

W_HJ, W_URGENCY = 0.6, 0.4


@dataclass
class SceneAssessment:
    scene_risk: float
    min_ttc: Optional[float]          # None = nothing on a collision course
    primary_threat: Optional[str]
    per_target: Dict[str, dict]

    def band(self) -> str:
        if self.scene_risk < config.RISK_LOW:
            return "low"
        if self.scene_risk < config.RISK_TRIGGER:
            return "elevated"
        return "critical"


def assess_scene(snapshot: Snapshot) -> SceneAssessment:
    ex, ey = snapshot.ego.velocity
    per_target: Dict[str, dict] = {}
    scene_risk, min_ttc, primary = 0.0, None, None

    for tid, pos, vel, kind in snapshot.targets():
        hj = hj_model.hj_risk((0.0, 0.0), ex, pos)
        px, py = pos
        rvx, rvy = vel[0] - ex, vel[1] - ey
        ttc, miss, on_course = None, None, False
        closing = px * rvx + py * rvy
        v2 = rvx * rvx + rvy * rvy
        if closing < 0 and v2 > 1e-9:
            tca = -closing / v2
            miss = math.hypot(px + rvx * tca, py + rvy * tca)
            on_course = miss <= effective_radius(kind) + 1.0
            if on_course:
                ttc = tca
        urgency = 0.0
        if ttc is not None:
            urgency = max(0.0, min(1.0, 1.0 - ttc / config.TTC_URGENCY_HORIZON))
        # The HJ net was trained on forward obstacles; a target already behind
        # the ego and not on a collision course cannot be the binding threat.
        hj_weight = W_HJ * (config.REAR_HJ_DISCOUNT if (px < 0 and not on_course) else 1.0)
        combined = round(hj_weight * hj + W_URGENCY * urgency, 3)
        per_target[tid] = {"hj_risk": hj, "ttc_s": None if ttc is None else round(ttc, 2),
                           "miss_m": None if miss is None else round(miss, 2),
                           "on_collision_course": on_course,
                           "urgency": round(urgency, 3), "combined": combined}
        if combined > scene_risk:
            scene_risk, primary = combined, tid
        if ttc is not None and (min_ttc is None or ttc < min_ttc):
            min_ttc = ttc

    return SceneAssessment(scene_risk=round(scene_risk, 3), min_ttc=min_ttc,
                           primary_threat=primary, per_target=per_target)


@dataclass
class CognitionBudget:
    deadline_s: float
    decision_role: str      # "decision" (flagship) or "decision_fast" (mini)
    allow_vision: bool
    allow_debate: bool
    max_tool_iters: int
    tier: str


def budget_from(assessment: SceneAssessment) -> CognitionBudget:
    if assessment.min_ttc is None:
        window = 8.0  # nothing on a collision course yet: generous anticipation window
    else:
        window = max(0.0, assessment.min_ttc - config.T_ACTUATION - config.DEADLINE_MARGIN)

    if window >= 1.5:
        return CognitionBudget(window, "decision", True, True, 6, "deliberate")
    if window >= 0.6:
        return CognitionBudget(window, "decision_fast", False, False, 3, "fast")
    return CognitionBudget(window, "decision_fast", False, False, 0, "reflex_only")
