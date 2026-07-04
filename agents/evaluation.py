"""EvaluationAgent — consolidation-loop reflection.

Turns each executed episode into (a) a structured lesson stored with the case
in episodic memory and (b) a fine-tuning sample appended to the distillation
export, so the slow deliberative system gradually teaches the fast one.
"""
import json
import time

from .. import config
from .base import Agent, strict_schema
from .decision import _SYSTEM as DECISION_SYSTEM

_SCHEMA = strict_schema("evaluation_report", {
    "assessment": {"type": "string", "description": "<=3 sentences on effectiveness"},
    "lesson": {"type": "string",
               "description": "one transferable, scenario-generalizable lesson"},
    "better_alternative": {"type": "string",
                           "description": "a better maneuver if one existed, else 'None'"},
})

_SYSTEM = """\
You are the evaluation agent of a multi-agent collision-avoidance system.
Given the scenario, the executed maneuver and its (simulated) outcome, write
a blunt effectiveness assessment, one transferable lesson for future similar
scenes, and a better alternative if one plausibly existed. The lesson will be
retrieved verbatim in future emergencies — make it operational, not vague.
"""


class EvaluationAgent(Agent):
    name = "evaluation"
    role = "evaluation"

    async def run(self, bb, executed_code: int, executed_branch, outcome: str) -> dict:
        result = await self.llm.chat(
            self.role,
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content":
                 "Scenario:\n" + bb.snapshot.compact_text()
                 + f"\n\nExecuted maneuver: code {executed_code} "
                 f"({config.DECISION_CODES.get(executed_code, 'unknown')})"
                 + (f"\nBranch rationale: {executed_branch['rationale']}"
                    if executed_branch else "")
                 + f"\n\nOutcome:\n{outcome}"}],
            response_schema=_SCHEMA, max_tokens=350,
        )
        bb.evaluation_report = result.parsed
        self.record(bb, "consolidation", result.parsed, result)
        return result.parsed

    def export_distillation_sample(self, bb, executed_code: int):
        """Append this episode as a fine-tuning sample (same messages format as
        fine-tuning/data_fine_shiyue_exported.jsonl) — the data outlet of the
        slow-teaches-fast loop."""
        sample = {"messages": [
            {"role": "system",
             "content": DECISION_SYSTEM.split("Method — follow strictly")[0].strip()},
            {"role": "user",
             "content": "CURRENT SCENARIO:\n" + bb.snapshot.compact_text()},
            {"role": "assistant",
             "content": json.dumps({"policy": bb.policy,
                                    "executed_code": executed_code})},
        ]}
        config.DISTILL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.DISTILL_EXPORT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.record(bb, "consolidation",
                    {"distill_sample_appended": str(config.DISTILL_EXPORT_PATH),
                     "at": time.strftime("%H:%M:%S")})
