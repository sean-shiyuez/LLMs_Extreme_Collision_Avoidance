"""RiskAgent — deterministic risk metrics plus a short LLM synthesis.

The numbers (HJ risk, TTC, braking feasibility, corridor occupancy) come from
the physics tools, not the LLM; the LLM only fuses them into a prioritized
risk statement so downstream agents get one authoritative reading.
"""
import json

from .. import config
from ..tools import physics
from .base import Agent, strict_schema

_SCHEMA = strict_schema("risk_report", {
    "summary": {"type": "string", "description": "<=4 sentences fusing the metrics"},
    "primary_threat": {"type": "string"},
})

_SYSTEM = """\
You are the risk-assessment agent of a multi-agent collision-avoidance
system. You receive deterministic metrics computed by physics models:
HJ-reachability risk per target (1.0 = certain collision set), TTC and
predicted miss distances, braking feasibility, and lateral corridor
occupancy. Fuse them into a short prioritized risk statement and name the
binding threat (primary_threat, by target id). Do not invent numbers.
"""


class RiskAgent(Agent):
    name = "risk"
    role = "risk"

    async def run(self, bb):
        snap = bb.snapshot
        exec_snap = bb.exec_snapshot or snap
        metrics = {
            "per_target": bb.assessment.per_target,
            "scene_risk": bb.assessment.scene_risk,
            "min_ttc_s": bb.assessment.min_ttc,
            # maneuver feasibility is judged at the anticipated execution state
            "braking_at_execution": physics.braking_distance(exec_snap),
            "corridor_left_at_execution": physics.lateral_clearance(exec_snap, "left"),
            "corridor_right_at_execution": physics.lateral_clearance(exec_snap, "right"),
        }
        if config.SKIP_RISK_LLM:
            # Realtime profile: the summary is a deterministic template — the
            # numbers already say everything the decision agent needs.
            report = self._template_summary(bb, metrics)
            bb.risk_report = dict(metrics)
            bb.risk_report.update(report)
            self.record(bb, "deliberation", report,
                        extra={"note": "deterministic template (realtime profile)"})
            return
        result = await self.llm.chat(
            self.role,
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": "Deterministic metrics:\n"
              + json.dumps(metrics, indent=1)
              + "\n\nScene:\n" + snap.compact_text()}],
            response_schema=_SCHEMA, max_tokens=400,
        )
        bb.risk_report = dict(metrics)
        bb.risk_report.update(result.parsed)
        self.record(bb, "deliberation", result.parsed, result)

    @staticmethod
    def _template_summary(bb, metrics) -> dict:
        primary = bb.assessment.primary_threat or "none"
        per = bb.assessment.per_target.get(primary, {})
        braking = metrics["braking_at_execution"]
        left = metrics["corridor_left_at_execution"]
        right = metrics["corridor_right_at_execution"]

        def corridor_desc(c):
            if c["occupants"]:
                return "occupied by " + ",".join(o["id"] for o in c["occupants"])
            if not c["lane_change_possible"]:
                return f"blocked (boundary margin {c['boundary_margin_m']} m)"
            return "free"

        summary = (
            f"Primary threat {primary}: HJ risk {per.get('hj_risk')}, "
            f"TTC {per.get('ttc_s')} s, on collision course: "
            f"{per.get('on_collision_course')}. Braking alone "
            f"{'IS' if braking['braking_alone_sufficient'] else 'is NOT'} sufficient "
            f"(stopping {braking['stopping_distance_m']} m vs gap "
            f"{braking['gap_to_it_m']} m). Left corridor {corridor_desc(left)}; "
            f"right corridor {corridor_desc(right)}."
        )
        return {"summary": summary, "primary_threat": primary}
