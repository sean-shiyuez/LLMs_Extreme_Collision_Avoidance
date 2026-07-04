"""Agent base class: timing, token accounting and blackboard trace plumbing."""
import time
from typing import Optional

from ..llm import LLMResult


class Agent:
    name = "agent"
    role = "decision"

    def __init__(self, llm):
        self.llm = llm

    def record(self, blackboard, phase: str, output,
               result: Optional[LLMResult] = None, extra: Optional[dict] = None):
        event = {
            "phase": phase,
            "agent": self.name,
            "t_wall": time.time(),
            "output": output,
        }
        if result is not None:
            event.update({
                "model": result.model,
                "elapsed_s": round(result.elapsed, 3),
                "usage": result.usage,
                "tool_calls": result.tool_trace,
            })
        if extra:
            event.update(extra)
        blackboard.trace.append(event)


def strict_schema(name: str, properties: dict, required=None) -> dict:
    """Helper for OpenAI strict structured outputs (all fields required,
    additionalProperties disallowed)."""
    return {
        "name": name,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
            "additionalProperties": False,
        },
    }
