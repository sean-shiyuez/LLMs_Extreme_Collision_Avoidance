"""PerceptionAgent — VLM scene understanding over the rendered BEV image.

Runs only in the anticipation phase (never on the trigger critical path).
Complements the numeric channel with semantics: which side offers an escape
corridor, which object is the binding threat, what the image suggests that
coordinates alone do not (occlusion, clustering).
"""
from .base import Agent, strict_schema

_SCHEMA = strict_schema("scene_report", {
    "summary": {"type": "string",
                "description": "3-4 sentence semantic description of the scene"},
    "hazards": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "kind": {"type": "string"},
                       "note": {"type": "string"}},
        "required": ["id", "kind", "note"], "additionalProperties": False}},
    "free_corridor_left": {"type": "boolean"},
    "free_corridor_right": {"type": "boolean"},
    "occlusions": {"type": "string"},
    "primary_threat": {"type": "string"},
})

_SYSTEM = """\
You are the perception agent of a multi-agent collision-avoidance system for
an autonomous vehicle. You receive a bird's-eye-view (BEV) rendering of the
scene (ego vehicle in blue at the origin, driving direction is +x to the
right; the TOP of the image is the ego's LEFT side) together with the same
scene in structured text. Identify the hazards, judge which lateral corridor
(left/right of the ego) is free enough for an evasive maneuver, name the
single most dangerous object (primary_threat, by its id), and note anything
the geometry implies (occlusion, closing patterns). Be terse and factual —
your report is consumed by a decision agent, not a human.
"""


class PerceptionAgent(Agent):
    name = "perception"
    role = "perception"

    async def run(self, bb):
        if bb.bev_b64 is None:
            bb.perception_report = {"summary": "Vision disabled for this run.",
                                    "hazards": [], "free_corridor_left": True,
                                    "free_corridor_right": True, "occlusions": "unknown",
                                    "primary_threat": bb.assessment.primary_threat or ""}
            self.record(bb, "deliberation", bb.perception_report,
                        extra={"note": "vision skipped"})
            return
        user_content = [
            {"type": "text",
             "text": "Structured scene:\n" + bb.snapshot.compact_text(
                 {tid: v["hj_risk"] for tid, v in bb.assessment.per_target.items()})
             + "\n\nAnalyze the BEV image and report."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{bb.bev_b64}"}},
        ]
        result = await self.llm.chat(
            self.role,
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": user_content}],
            response_schema=_SCHEMA, max_tokens=600,
        )
        bb.perception_report = result.parsed
        self.record(bb, "deliberation", result.parsed, result)
