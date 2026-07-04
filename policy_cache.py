"""Armed contingency-policy cache — the bridge between the deliberation loop
(seconds) and the reflex loop (milliseconds).

A deliberation produces a contingency policy tree whose branches are labelled
with a closed set of guard conditions (config.BRANCH_CONDITIONS). At the
trigger instant the reflex loop classifies the observed scene evolution from
two snapshots by pure kinematics and selects the matching branch — no LLM,
no free-text parsing, wall-clock in the sub-millisecond range.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .budget import assess_scene
from .scenario.schema import Snapshot


@dataclass
class PolicyBranch:
    condition_type: str
    condition: str
    code: int
    target_speed_mps: float
    rationale: str
    confidence: float

    @staticmethod
    def from_dict(d: dict) -> "PolicyBranch":
        return PolicyBranch(
            condition_type=d["condition_type"], condition=d.get("condition", ""),
            code=int(d["code"]), target_speed_mps=float(d.get("target_speed_mps", 0.0)),
            rationale=d.get("rationale", ""), confidence=float(d.get("confidence", 0.5)),
        )


@dataclass
class ArmedPolicy:
    scenario_name: str
    branches: List[PolicyBranch]
    source: str                    # "deliberation" | "precomputed"
    primary_threat: Optional[str]
    deliberation_s: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "source": self.source,
            "primary_threat": self.primary_threat,
            "deliberation_s": round(self.deliberation_s, 3),
            "branches": [vars(b) for b in self.branches],
        }


def classify_evolution(prev: Snapshot, now: Snapshot,
                       primary_threat: Optional[str]) -> str:
    """Kinematic classification of how the scene evolved since deliberation."""
    now_assess = assess_scene(now)
    if primary_threat and now_assess.primary_threat and \
            now_assess.primary_threat != primary_threat:
        return "secondary_threat_activates"

    prev_t = {tid: (pos, vel) for tid, pos, vel, _ in prev.targets()}
    now_t = {tid: (pos, vel) for tid, pos, vel, _ in now.targets()}
    if primary_threat not in now_t:
        return "primary_threat_yields"
    per = now_assess.per_target.get(primary_threat, {})
    if not per.get("on_collision_course", False):
        return "primary_threat_yields"
    if primary_threat in prev_t:
        (_, pv), (_, nv) = prev_t[primary_threat], now_t[primary_threat]
        prev_speed = (pv[0] ** 2 + pv[1] ** 2) ** 0.5
        now_speed = (nv[0] ** 2 + nv[1] ** 2) ** 0.5
        if now_speed > prev_speed * 1.1 + 0.5:
            return "primary_threat_accelerates"
        if now_speed < prev_speed * 0.7 - 0.5:
            return "primary_threat_yields"
    return "primary_threat_maintains"


class PolicyCache:
    def __init__(self):
        self._armed: Optional[ArmedPolicy] = None

    @property
    def armed(self) -> Optional[ArmedPolicy]:
        return self._armed

    def arm(self, policy: ArmedPolicy):
        self._armed = policy

    def invalidate(self):
        self._armed = None

    def match(self, prev: Snapshot, now: Snapshot) -> Tuple[Optional[PolicyBranch], str, float]:
        """Returns (branch, observed_condition, elapsed_ms). Reflex hot path."""
        start = time.perf_counter()
        if self._armed is None:
            return None, "no_armed_policy", (time.perf_counter() - start) * 1000
        observed = classify_evolution(prev, now, self._armed.primary_threat)
        branch = next((b for b in self._armed.branches if b.condition_type == observed), None)
        if branch is None:
            branch = next((b for b in self._armed.branches if b.condition_type == "default"), None)
        return branch, observed, (time.perf_counter() - start) * 1000
