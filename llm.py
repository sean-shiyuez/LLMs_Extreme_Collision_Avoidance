"""LLM client layer.

`OpenAIClient` wraps AsyncOpenAI with the three call shapes MACA needs:
plain chat, strict-JSON structured output, and a function-calling tool loop.
Every call is timed and token-counted so the orchestrator can enforce
deadlines and produce the latency breakdown used in the paper experiments.

`MockLLM` implements the same interface with deterministic, role-aware
responses (and real tool executions), so the full three-loop pipeline runs
offline via `--mock`.
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import config


@dataclass
class LLMResult:
    content: str
    parsed: Optional[dict] = None
    usage: Dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0
    tool_trace: List[dict] = field(default_factory=list)
    model: str = ""


class ToolRegistry:
    """Maps tool names to (OpenAI function schema, callable)."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: dict, fn: Callable[..., dict]):
        self._tools[name] = {
            "schema": {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            },
            "fn": fn,
        }

    def schemas(self) -> List[dict]:
        return [t["schema"] for t in self._tools.values()]

    def call(self, name: str, arguments: dict) -> dict:
        if name not in self._tools:
            return {"error": f"unknown tool {name}"}
        try:
            return self._tools[name]["fn"](**arguments)
        except Exception as exc:  # tool errors go back to the model, not up the stack
            return {"error": f"{type(exc).__name__}: {exc}"}


def _token_kwargs(model: str, max_tokens: int) -> dict:
    # gpt-5.x / o-series accept only max_completion_tokens and fixed
    # temperature. Reasoning tokens draw from the same budget: with reasoning
    # disabled ("none") almost no headroom is needed; otherwise reserve room
    # so thinking cannot starve the visible output.
    if model.startswith(("gpt-5", "o")):
        effort = config.REASONING_EFFORT
        headroom = 200 if effort == "none" else 4000
        kwargs = {"max_completion_tokens": max_tokens + headroom}
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs
    return {"max_tokens": max_tokens, "temperature": 0}


async def _create_with_fallback(client, kwargs: dict):
    """Call chat.completions.create; if the model rejects reasoning_effort
    (older snapshots), retry once without it."""
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as exc:
        if "reasoning_effort" in kwargs and "reasoning" in str(exc).lower():
            kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
            return await client.chat.completions.create(**kwargs)
        raise


class OpenAIClient:
    def __init__(self):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def chat(
        self,
        role: str,
        messages: List[dict],
        response_schema: Optional[dict] = None,
        max_tokens: int = 900,
        timeout_s: float = 120.0,
    ) -> LLMResult:
        model = config.model_for(role)
        kwargs: Dict[str, Any] = dict(model=model, messages=messages, **_token_kwargs(model, max_tokens))
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_schema["name"], "strict": True,
                                "schema": response_schema["schema"]},
            }
        start = time.perf_counter()
        resp = await asyncio.wait_for(
            _create_with_fallback(self._client, kwargs), timeout=timeout_s
        )
        elapsed = time.perf_counter() - start
        content = (resp.choices[0].message.content or "").strip()
        parsed = None
        if response_schema is not None:
            parsed = json.loads(content)
        return LLMResult(
            content=content,
            parsed=parsed,
            usage={"prompt_tokens": resp.usage.prompt_tokens,
                   "completion_tokens": resp.usage.completion_tokens},
            elapsed=elapsed,
            model=model,
        )

    async def tool_loop(
        self,
        role: str,
        messages: List[dict],
        registry: ToolRegistry,
        response_schema: dict,
        max_iters: int = 6,
        max_tokens: int = 1200,
        timeout_s: float = 300.0,
    ) -> LLMResult:
        """Function-calling loop, then one structured finalization call."""
        model = config.model_for(role)
        messages = list(messages)
        trace: List[dict] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        start = time.perf_counter()
        deadline = start + timeout_s

        for _ in range(max_iters):
            resp = await asyncio.wait_for(
                _create_with_fallback(self._client, dict(
                    model=model, messages=messages, tools=registry.schemas(),
                    **_token_kwargs(model, max_tokens),
                )),
                timeout=max(5.0, deadline - time.perf_counter()),
            )
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens
            msg = resp.choices[0].message
            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or ""})
                break
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                t0 = time.perf_counter()
                result = registry.call(tc.function.name, args)
                trace.append({"tool": tc.function.name, "arguments": args,
                              "result": result, "elapsed": time.perf_counter() - t0})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})

        messages.append({
            "role": "user",
            "content": "Based on your analysis above, output the final answer now "
                       "in the required JSON format.",
        })
        final = await self.chat(role, messages, response_schema=response_schema,
                                max_tokens=max_tokens,
                                timeout_s=max(5.0, deadline - time.perf_counter()))
        usage["prompt_tokens"] += final.usage["prompt_tokens"]
        usage["completion_tokens"] += final.usage["completion_tokens"]
        return LLMResult(content=final.content, parsed=final.parsed, usage=usage,
                         elapsed=time.perf_counter() - start, tool_trace=trace, model=model)


class MockLLM:
    """Deterministic offline stand-in. Role-aware canned responses; the
    decision role really exercises registered tools so traces stay honest,
    and its first proposal is deliberately unsafe to exercise the veto loop."""

    def __init__(self):
        self.decision_round = 0

    @staticmethod
    def _text_of(messages: List[dict]) -> str:
        parts = []
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                parts.extend(x.get("text", "") for x in c if isinstance(x, dict))
            else:
                parts.append(str(c))
        return "\n".join(parts)

    async def chat(self, role, messages, response_schema=None, max_tokens=900,
                   timeout_s=60.0) -> LLMResult:
        text = self._text_of(messages)
        payload = self._respond(role, text)
        content = json.dumps(payload) if isinstance(payload, dict) else payload
        return LLMResult(content=content,
                         parsed=payload if isinstance(payload, dict) else None,
                         usage={"prompt_tokens": 0, "completion_tokens": 0},
                         elapsed=0.001, model=f"mock-{role}")

    async def tool_loop(self, role, messages, registry, response_schema,
                        max_iters=6, max_tokens=1200, timeout_s=90.0) -> LLMResult:
        start = time.perf_counter()
        trace = []
        # Demonstrate genuine tool use: probe every registered tool that takes
        # a target_id, using the first participant mentioned in the prompt.
        text = self._text_of(messages)
        target = None
        m = re.search(r"(?:TRUCK|SUV|PEDESTRIAN|SMALL CAR)\s+(\S+):", text)
        if m:
            target = m.group(1)
        if target:
            for name in ("compute_ttc", "query_hj_risk", "braking_distance"):
                if name in registry._tools:
                    args = {"target_id": target} if name != "braking_distance" else {}
                    t0 = time.perf_counter()
                    result = registry.call(name, args)
                    trace.append({"tool": name, "arguments": args, "result": result,
                                  "elapsed": time.perf_counter() - t0})
        final = await self.chat(role, messages, response_schema=response_schema)
        final.tool_trace = trace
        final.elapsed = time.perf_counter() - start
        return final

    # ------------------------------------------------------------------
    def _respond(self, role: str, text: str):
        if role == "perception":
            return self._perception(text)
        if role == "risk":
            return {"summary": "Deterministic metrics indicate an imminent-collision "
                               "geometry dominated by the primary threat; braking distance "
                               "exceeds the available gap so steering-based maneuvers must "
                               "be considered.",
                    "primary_threat": self._primary_threat(text)}
        if role == "decision" or role == "decision_fast":
            return self._decision(text)
        if role == "safety":
            ok = "DETERMINISTIC_CHECKS_PASSED: true" in text
            return {"approved": ok,
                    "veto_reason": "" if ok else
                    "Rejected branches violate hard safety rules (steering toward a "
                    "pedestrian-occupied side or exceeding HJ safe-set risk).",
                    "notes": "Mock review mirrors the deterministic shield verdict."}
        if role in ("advocate", "arbiter"):
            if role == "arbiter":
                return {"resolution": "Adopt the safety advocate's branch ordering; "
                                      "efficiency concerns are secondary under an "
                                      "imminent-collision TTC.", "revised_codes": []}
            return {"argument": "Prioritize minimizing worst-case harm; accept maneuver "
                                "cost to protect the vulnerable participant."}
        if role == "evaluation":
            return {"assessment": "The executed maneuver matched the armed branch and "
                                  "produced the least-harm outcome available in this "
                                  "scenario.",
                    "lesson": "For laterally approaching heavy vehicles with TTC below "
                              "1.3 s, arm a rear-facing T-drift branch early and keep "
                              "full braking as the default branch.",
                    "better_alternative": "None identified given the geometry."}
        return {"note": f"mock response for role {role}"}

    @staticmethod
    def _primary_threat(text: str) -> str:
        m = re.search(r"(?:TRUCK|SUV)\s+(\S+):", text)
        return m.group(1) if m else "unknown"

    def _perception(self, text: str):
        hazards = []
        for m in re.finditer(r"(TRUCK|SUV|PEDESTRIAN|SMALL CAR)\s+(\S+): ([^\n]+)", text):
            hazards.append({"id": m.group(2), "kind": m.group(1).lower(),
                            "note": m.group(3)[:120]})
        left_blocked = bool(re.search(r"m left", text))
        right_blocked = bool(re.search(r"m right", text))
        return {
            "summary": "BEV shows the ego lane constrained by nearby traffic; see hazards.",
            "hazards": hazards,
            "free_corridor_left": not left_blocked,
            "free_corridor_right": not right_blocked,
            "occlusions": "None visible in the rendered BEV.",
            "primary_threat": self._primary_threat(text),
        }

    def _decision(self, text: str):
        self.decision_round += 1
        pedestrian_left = bool(re.search(r"PEDESTRIAN \S+: [\d.]+m ahead, [\d.]+m left", text))
        intersection = "Intersection" in text
        if self.decision_round == 1:
            # Deliberately propose steering into the constrained side once, so
            # the safety shield's veto -> re-decide loop is exercised offline.
            unsafe_code = 3 if pedestrian_left else 3
            branches = [
                {"condition_type": "primary_threat_maintains",
                 "condition": "Primary threat keeps closing on the current course",
                 "code": unsafe_code, "target_speed_mps": 6.0,
                 "rationale": "Swerve left with braking to open lateral distance.",
                 "confidence": 0.55},
                {"condition_type": "default", "condition": "Any other evolution",
                 "code": 0, "target_speed_mps": 0.0,
                 "rationale": "Full braking minimizes kinetic energy.", "confidence": 0.8},
            ]
        else:
            main_code = 6 if intersection else 4
            branches = [
                {"condition_type": "primary_threat_maintains",
                 "condition": "Primary threat keeps closing on the current course",
                 "code": main_code,
                 "target_speed_mps": 0.0 if main_code == 6 else 8.0,
                 "rationale": ("Rear-facing T-drift toward the safer side takes the "
                               "impact on the energy-absorbing rear structure."
                               if main_code == 6 else
                               "Lane change to the free side with braking clears the "
                               "blocked lane at a controllable speed."),
                 "confidence": 0.85},
                {"condition_type": "primary_threat_yields",
                 "condition": "Primary threat brakes or turns away",
                 "code": 0, "target_speed_mps": 0.0,
                 "rationale": "Straight-line braking is sufficient once the threat yields.",
                 "confidence": 0.9},
                {"condition_type": "default", "condition": "Any other evolution",
                 "code": 0, "target_speed_mps": 0.0,
                 "rationale": "Full braking is the minimal-risk fallback.", "confidence": 0.9},
            ]
        return {"branches": branches,
                "overall_rationale": "Contingency tree covering threat persistence, "
                                     "yielding, and a braking fallback.",
                "requested_by": f"mock-round-{self.decision_round}"}
