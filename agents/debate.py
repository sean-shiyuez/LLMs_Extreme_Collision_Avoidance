"""Consistency-gated debate.

Full debate on every decision would double latency for nothing when the
semantic (perception) and numeric (risk) channels already agree. Debate is
therefore triggered only on disagreement or low decision confidence (or
forced via --debate on): a safety advocate and an efficiency advocate argue
over the proposed policy in parallel, and an arbiter issues binding branch
revisions, which are re-verified by the safety shield afterwards.
"""
import asyncio
import json

from .. import config
from .base import Agent, strict_schema

_ADVOCATE_SCHEMA = strict_schema("advocate_position", {
    "argument": {"type": "string", "description": "<=4 sentences"},
})

_ARBITER_SCHEMA = strict_schema("arbiter_resolution", {
    "resolution": {"type": "string"},
    "revised_codes": {"type": "array", "items": {
        "type": "object",
        "properties": {"condition_type": {"type": "string",
                                          "enum": config.BRANCH_CONDITIONS},
                       "code": {"type": "integer", "minimum": 0, "maximum": 7}},
        "required": ["condition_type", "code"], "additionalProperties": False}},
})

_SAFETY_SYSTEM = """\
You are the SAFETY ADVOCATE in a collision-avoidance debate. Argue for the
policy revision that minimizes worst-case harm, giving absolute priority to
vulnerable road users, even at the cost of vehicle damage or traffic flow.
"""

_EFFICIENCY_SYSTEM = """\
You are the EFFICIENCY ADVOCATE in a collision-avoidance debate. Argue
against overreaction: unnecessary emergency maneuvers cause secondary
accidents, rear-end collisions and false triggering. Prefer the mildest
maneuver that the physics evidence still supports.
"""

_ARBITER_SYSTEM = """\
You are the ARBITER of a collision-avoidance debate. You receive the proposed
contingency policy, the physics evidence, and the two advocates' arguments.
Issue a short resolution. If a branch's maneuver code should change, list it
in revised_codes; otherwise return an empty list. Safety outranks efficiency
whenever the two genuinely conflict.
"""


def debate_needed(bb, mode: str) -> bool:
    if mode == "off" or not bb.budget.allow_debate:
        return False
    if mode == "on":
        return True
    perception_threat = (bb.perception_report or {}).get("primary_threat", "")
    risk_threat = (bb.risk_report or {}).get("primary_threat", "")
    disagree = bool(perception_threat and risk_threat
                    and perception_threat != risk_threat)
    confidences = [b.get("confidence", 1.0) for b in bb.policy.get("branches", [])]
    low_conf = bool(confidences) and min(confidences) < 0.6
    return disagree or low_conf


class DebatePanel(Agent):
    name = "debate"
    role = "arbiter"

    async def run(self, bb):
        brief = ("Scene:\n" + bb.snapshot.compact_text()
                 + "\n\nProposed policy:\n" + json.dumps(bb.policy, indent=1)
                 + "\n\nRisk summary: " + str(bb.risk_report.get("summary", "")))
        safety_pos, efficiency_pos = await asyncio.gather(
            self.llm.chat("advocate",
                          [{"role": "system", "content": _SAFETY_SYSTEM},
                           {"role": "user", "content": brief}],
                          response_schema=_ADVOCATE_SCHEMA, max_tokens=300),
            self.llm.chat("advocate",
                          [{"role": "system", "content": _EFFICIENCY_SYSTEM},
                           {"role": "user", "content": brief}],
                          response_schema=_ADVOCATE_SCHEMA, max_tokens=300),
        )
        arbiter = await self.llm.chat(
            "arbiter",
            [{"role": "system", "content": _ARBITER_SYSTEM},
             {"role": "user", "content": brief
              + "\n\nSAFETY ADVOCATE: " + safety_pos.parsed["argument"]
              + "\n\nEFFICIENCY ADVOCATE: " + efficiency_pos.parsed["argument"]}],
            response_schema=_ARBITER_SCHEMA, max_tokens=400,
        )
        revisions = {r["condition_type"]: r["code"]
                     for r in arbiter.parsed["revised_codes"]}
        for branch in bb.policy.get("branches", []):
            if branch["condition_type"] in revisions:
                branch["code"] = revisions[branch["condition_type"]]
                branch["rationale"] += " [revised by debate arbiter]"
        bb.debate_report = {
            "safety_argument": safety_pos.parsed["argument"],
            "efficiency_argument": efficiency_pos.parsed["argument"],
            "resolution": arbiter.parsed["resolution"],
            "revisions": revisions,
        }
        self.record(bb, "deliberation", bb.debate_report, arbiter)
