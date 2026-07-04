"""SafetyAgent — the physics-grounded safety shield.

Every policy branch is checked deterministically BEFORE any LLM judgement:
a kinematic rollout of the maneuver is scored against the HJ-reachability
value network (end state must stay out of the high-risk set), plus hard
rules (never steer toward a pedestrian-occupied side, respect road
boundaries, T-drift preconditions). The LLM then reviews the evidence and
can only tighten, never loosen, the verdict: a branch fails if either the
deterministic shield or the LLM rejects it. Code 0 (full braking) is the
minimal-risk maneuver and is always admissible as a fallback.
"""
import json

from .. import config
from ..tools import physics
from .base import Agent, strict_schema

_SCHEMA = strict_schema("safety_review", {
    "approved": {"type": "boolean"},
    "veto_reason": {"type": "string"},
    "notes": {"type": "string"},
})

_SYSTEM = """\
You are the safety-verification agent (runtime shield) of a multi-agent
collision-avoidance system. You receive a proposed contingency policy tree
together with deterministic physics evidence per branch: a kinematic rollout
scored by an HJ-reachability value network, and hard-rule checks. Approve the
policy only if every branch is physically defensible. You may reject a policy
the deterministic checks passed (if you see a hazard they missed), but you
must NEVER approve a policy any deterministic check failed.
"""


def _hard_rule_check(bb, branch: dict) -> list:
    """Returns a list of violated-rule strings (empty = pass).

    All geometry is evaluated at the ANTICIPATED EXECUTION STATE
    (bb.exec_snapshot): the branch fires at the trigger instant, not now.
    """
    code = int(branch["code"])
    snap = bb.exec_snapshot
    violations = []
    if code == 0:
        return violations  # minimal-risk maneuver: always admissible
    if code == 7:
        if bb.assessment.band() != "low":
            violations.append("code 7 (no intervention) in a non-low-risk scene")
        return violations

    side = "left" if code in config.LEFTWARD_CODES else "right"
    corridor = physics.lateral_clearance(snap, side)
    if any(o["vulnerable"] for o in corridor["occupants"]):
        violations.append(f"steers toward the {side} side occupied by a vulnerable road user")
    if code in (1, 2, 3, 4):
        if not corridor["lane_change_possible"]:
            violations.append(
                f"{side} lane change infeasible (occupants={[o['id'] for o in corridor['occupants']]}, "
                f"boundary_margin={corridor['boundary_margin_m']})")
    if code in (5, 6):
        on_course_imminent = any(
            v.get("on_collision_course") and v.get("ttc_s") is not None
            and v["ttc_s"] <= 1.3
            for v in bb.exec_assessment.per_target.values()
        )
        lateral_fast = any(
            abs(vel[1]) > 4.0 and pos[0] > 0 and abs(pos[1]) < 8.0
            for _, pos, vel, _ in snap.targets()
        )
        if not (on_course_imminent and lateral_fast):
            violations.append(
                "T-drift preconditions unmet at the anticipated execution state "
                "(needs a laterally approaching target ahead, |vy|>4 m/s, "
                "lateral distance <8 m, TTC<=1.3 s)")
    return violations


class SafetyAgent(Agent):
    name = "safety"
    role = "safety"

    async def run(self, bb, round_idx: int = 1) -> dict:
        branch_verdicts = []
        deterministic_ok = True
        for branch in bb.policy.get("branches", []):
            code = int(branch["code"])
            rollout = physics.rollout_maneuver(bb.exec_snapshot, code,
                                               branch.get("target_speed_mps") or None)
            violations = _hard_rule_check(bb, branch)
            # Codes 5/6 are last-resort impact-posture maneuvers: their end
            # state is EXPECTED to be near the target (the point is choosing
            # which face of the car takes the hit), so the HJ end-state and
            # separation gates apply only to avoidance maneuvers 1-4.
            if code in (1, 2, 3, 4) and not rollout.get("hj_acceptable", True):
                violations.append(
                    f"rollout end state worsens HJ reachability risk "
                    f"({rollout['worst_end_hj_risk']:.2f} vs no-intervention baseline "
                    f"{rollout['baseline_no_intervention_hj_risk']:.2f})")
            if code in (1, 2, 3, 4) and rollout.get("min_separation_m", 99) < 0.3:
                violations.append(
                    f"rollout predicts near-contact (min separation "
                    f"{rollout['min_separation_m']} m with {rollout['closest_target']})")
            ok = not violations
            deterministic_ok &= ok
            branch_verdicts.append({
                "condition_type": branch["condition_type"], "code": code,
                "approved": ok, "violations": violations, "rollout": rollout,
            })

        review = await self.llm.chat(
            self.role,
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content":
                 f"DETERMINISTIC_CHECKS_PASSED: {'true' if deterministic_ok else 'false'}\n\n"
                 "Policy under review:\n" + json.dumps(bb.policy, indent=1)
                 + "\n\nPer-branch evidence:\n" + json.dumps(branch_verdicts, indent=1)}],
            response_schema=_SCHEMA, max_tokens=400,
        )
        approved = deterministic_ok and bool(review.parsed["approved"])
        veto_reason = review.parsed["veto_reason"]
        if not deterministic_ok:
            failed = [v for v in branch_verdicts if not v["approved"]]
            veto_reason = "; ".join(
                f"branch[{v['condition_type']}] code {v['code']}: {'; '.join(v['violations'])}"
                for v in failed) + (f" | LLM: {veto_reason}" if veto_reason else "")
        verdict = {"round": round_idx, "approved": approved,
                   "veto_reason": veto_reason,
                   "branch_verdicts": branch_verdicts,
                   "llm_notes": review.parsed["notes"]}
        bb.safety_verdicts.append(verdict)
        self.record(bb, "deliberation",
                    {k: verdict[k] for k in ("round", "approved", "veto_reason")},
                    review, extra={"branch_verdicts": [
                        {k: v[k] for k in ("condition_type", "code", "approved", "violations")}
                        for v in branch_verdicts]})
        return verdict

    @staticmethod
    def harden_policy(bb, verdict: dict):
        """Final fallback after SAFETY_MAX_ROUNDS: replace still-failing
        branches with the minimal-risk maneuver (code 0)."""
        failing = {v["condition_type"] for v in verdict["branch_verdicts"]
                   if not v["approved"]}
        for branch in bb.policy.get("branches", []):
            if branch["condition_type"] in failing:
                branch["code"] = 0
                branch["target_speed_mps"] = 0.0
                branch["rationale"] = ("Replaced by the minimal-risk maneuver after "
                                       "repeated safety vetoes: " + branch["rationale"])
