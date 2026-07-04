"""DecisionAgent — tool-augmented deliberation producing a contingency
policy tree (not a single action).

Each branch is guarded by one of the closed BRANCH_CONDITIONS so the reflex
loop can match the observed scene evolution in milliseconds at the trigger
instant. The agent is expected to ground its choices in tool evidence
(rollouts, TTC, corridors) before committing.
"""
import json

from .. import config
from .base import Agent, strict_schema

POLICY_SCHEMA = strict_schema("contingency_policy", {
    "branches": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "condition_type": {"type": "string", "enum": config.BRANCH_CONDITIONS},
            "condition": {"type": "string",
                          "description": "one-line human-readable guard"},
            "code": {"type": "integer", "minimum": 0, "maximum": 7},
            "target_speed_mps": {"type": "number",
                                 "description": "required for codes 3/4, else 0"},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["condition_type", "condition", "code", "target_speed_mps",
                     "rationale", "confidence"],
        "additionalProperties": False}},
    "overall_rationale": {"type": "string"},
})


def _codes_text() -> str:
    return "\n".join(f"  {k} = {v}" for k, v in config.DECISION_CODES.items())


_PREAMBLE = f"""\
You are the decision agent of a multi-agent collision-avoidance system for an
autonomous vehicle in an extreme, safety-critical situation. Your output is a
CONTINGENCY POLICY TREE that will be armed in a millisecond-latency cache and
executed later without any further LLM call, so it must cover how the scene
may evolve.

Maneuver catalogue:
{_codes_text()}

Branch guard conditions (use each at most once; always include "default"):
  primary_threat_maintains / primary_threat_yields / primary_threat_accelerates
  / secondary_threat_activates / default

Domain knowledge:
{config.DOMAIN_KNOWLEDGE}
"""

_METHOD_TOOLS = """\
Method — follow strictly:
1. Use the tools to gather evidence BEFORE deciding: braking feasibility,
   corridor occupancy on both sides, TTC of the primary threat, and
   rollout_maneuver for every candidate code you seriously consider.
2. Weigh safety first (minimize worst-case harm, protect vulnerable road
   users), then legality, then efficiency/comfort.
3. For codes 3/4 set target_speed_mps and justify it; otherwise set it to 0.
4. Learn from the retrieved historical cases and their lessons.
5. If a safety veto from a previous round is quoted, treat the vetoed
   branches as forbidden and propose physically different alternatives.
"""

_METHOD_EVIDENCE = """\
Method — follow strictly:
1. ALL physics evidence is precomputed in `physics_evidence`: TTC per target,
   braking feasibility, corridor occupancy on both sides, and a kinematic
   rollout of every maneuver code (min separation, end-state HJ risk vs the
   no-intervention baseline). Ground every branch in these numbers; do not
   invent physics.
2. Weigh safety first (minimize worst-case harm, protect vulnerable road
   users), then legality, then efficiency/comfort.
3. For codes 3/4 set target_speed_mps and justify it; otherwise set it to 0.
4. Learn from the retrieved historical cases and their lessons.
5. If a safety veto from a previous round is quoted, treat the vetoed
   branches as forbidden and propose physically different alternatives.
Answer with the JSON policy directly.
"""

_SYSTEM = _PREAMBLE + _METHOD_TOOLS  # kept for the distillation export


class DecisionAgent(Agent):
    name = "decision"

    async def run(self, bb, veto_feedback: str = "", round_idx: int = 1):
        role = bb.budget.decision_role
        context = {
            "scene_now": bb.snapshot.compact_text(
                {tid: v["hj_risk"] for tid, v in bb.assessment.per_target.items()}),
            "scene_at_anticipated_execution": bb.exec_snapshot.compact_text(
                {tid: v["hj_risk"] for tid, v in bb.exec_assessment.per_target.items()})
                if bb.exec_snapshot else None,
            "perception_report": bb.perception_report,
            "risk_report": {k: bb.risk_report.get(k)
                            for k in ("summary", "primary_threat",
                                      "braking_at_execution",
                                      "corridor_left_at_execution",
                                      "corridor_right_at_execution")},
            "historical_cases": bb.memory_report.get("cases", []),
            "cognition_budget": {"tier": bb.budget.tier,
                                 "deadline_s": round(bb.budget.deadline_s, 2)},
        }
        if config.EVIDENCE_UPFRONT:
            from ..tools import physics
            context["physics_evidence"] = physics.evidence_pack(bb.exec_snapshot)
        user = "Deliberation context:\n" + json.dumps(context, indent=1)
        if veto_feedback:
            user += f"\n\nSAFETY VETO FROM PREVIOUS ROUND (round {round_idx - 1}):\n{veto_feedback}"

        if config.EVIDENCE_UPFRONT:
            # Realtime profile: one structured call, zero tool round-trips.
            result = await self.llm.chat(
                role,
                [{"role": "system", "content": _PREAMBLE + _METHOD_EVIDENCE},
                 {"role": "user", "content": user}],
                response_schema=POLICY_SCHEMA, max_tokens=1200,
            )
        else:
            result = await self.llm.tool_loop(
                role,
                [{"role": "system", "content": _PREAMBLE + _METHOD_TOOLS},
                 {"role": "user", "content": user}],
                bb.registry,
                response_schema=POLICY_SCHEMA,
                max_iters=max(1, bb.budget.max_tool_iters),
            )
        bb.policy = result.parsed
        self.record(bb, "deliberation", result.parsed, result,
                    extra={"decision_round": round_idx, "role_used": role})
