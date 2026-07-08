"""LLM 客户端层 —— 仅供离线工厂使用（在线平面绝不 import 本模块）。

`OpenAIClient.chat` 是唯一的调用形态：严格 JSON 结构化输出（strict
json_schema），模型与思考强度按智能体角色从 config.FACTORY_ROLES 解析
—— 离线不吝啬思考，在线零思考。

`MockLLM` 以确定性方式实现同一接口，使整条工厂流水线可通过 --mock
完全离线跑通：生成器返回预置场景族（轮换），归纳器回落到无 LLM 的
归纳基线（induction_baseline）。
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config

# 各思考强度下，在可见输出 token 之外预留的推理 token 余量
_EFFORT_HEADROOM = {"none": 200, "low": 1500, "medium": 4000, "high": 8000}


@dataclass
class LLMResult:
    """一次 LLM 调用的结果（含计时与 token 统计，供工厂报告汇总）。"""
    content: str
    parsed: Optional[dict] = None
    usage: Dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0
    model: str = ""


def _token_kwargs(model: str, max_tokens: int, effort: str) -> dict:
    """按模型代际组装 token/思考参数。

    gpt-5.x / o 系列：只接受 max_completion_tokens（推理 token 与可见输出
    共享该预算，故按思考强度追加余量），并通过 reasoning_effort 控制思考；
    旧模型：常规 max_tokens + temperature=0。
    """
    if model.startswith(("gpt-5", "o")):
        kwargs = {"max_completion_tokens": max_tokens + _EFFORT_HEADROOM.get(effort, 4000)}
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs
    return {"max_tokens": max_tokens, "temperature": 0}


async def _create_with_fallback(client, kwargs: dict):
    """若模型快照不支持 reasoning_effort（旧版本），去掉该参数重试一次。"""
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as exc:
        if "reasoning_effort" in kwargs and "reasoning" in str(exc).lower():
            kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
            return await client.chat.completions.create(**kwargs)
        raise


# 判定为"瞬时可重试"的错误类型名（连接中断、服务端断开、超时等）
_TRANSIENT = ("APIConnectionError", "APITimeoutError", "InternalServerError",
              "RateLimitError", "RemoteProtocolError", "TimeoutError")


async def _with_network_retry(coro_factory, max_retries: int = 4):
    """对瞬时网络错误做指数退避重试（1s, 2s, 4s, 8s）；非瞬时错误直接抛出。
    coro_factory 是"每次调用返回一个新协程"的工厂（协程不可重复 await）。"""
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            name = type(exc).__name__
            transient = name in _TRANSIENT or "connection" in str(exc).lower()
            if not transient or attempt == max_retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2


class OpenAIClient:
    """真实 API 客户端（异步）。"""

    def __init__(self):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def chat(self, role: str, messages: List[dict],
                   response_schema: dict, max_tokens: int = 1200,
                   timeout_s: float = 300.0, max_retries: int = 4) -> LLMResult:
        """按角色发起一次严格结构化输出的对话调用。

        对瞬时网络错误（连接中断/服务端断开）做指数退避重试，避免一次抖动
        炸掉整个工厂批次、丢失已完成场景族的工作。
        """
        model = config.model_for(role)
        effort = config.effort_for(role)
        kwargs = dict(
            model=model, messages=messages,
            response_format={"type": "json_schema",
                             "json_schema": {"name": response_schema["name"],
                                             "strict": True,
                                             "schema": response_schema["schema"]}},
            **_token_kwargs(model, max_tokens, effort),
        )
        start = time.perf_counter()
        resp = await _with_network_retry(
            lambda: asyncio.wait_for(
                _create_with_fallback(self._client, kwargs), timeout=timeout_s),
            max_retries=max_retries)
        content = (resp.choices[0].message.content or "").strip()
        return LLMResult(
            content=content, parsed=json.loads(content),
            usage={"prompt_tokens": resp.usage.prompt_tokens,
                   "completion_tokens": resp.usage.completion_tokens},
            elapsed=time.perf_counter() - start, model=f"{model}({effort})")


# --------------------------------------------------------------------------
# --mock 模式的预置场景族（生成器按调用次数轮换，保证离线批次的多样性）
# --------------------------------------------------------------------------
_MOCK_FAMILIES = [
    {
        "family_id": "mock-lateral-crossing-left",
        "description": "Mock family: truck crossing from the left at an "
                       "intersection, no pedestrian.",
        "pattern": "lateral_crossing",
        "template": {"road_topology": "Intersection",
                     "threat_type": "Large Truck", "from_side": "left",
                     "pedestrian": {"present": False, "side": "left"},
                     "road_boundary_left": None, "road_boundary_right": None,
                     "follower": False, "blocked_side": None},
        "parameters": [
            {"name": "ego_speed", "min": 12, "max": 18, "steps": 2},
            {"name": "ttc", "min": 0.8, "max": 1.6, "steps": 3},
            {"name": "lateral_speed", "min": 5, "max": 9, "steps": 2},
        ],
        "hypothesis": "T-drift should dominate at short TTC / fast lateral "
                      "closing; braking or no-op should win at long TTC.",
    },
    {
        "family_id": "mock-crossing-escape-window",
        "description": "Mock family: left-crossing truck with more lateral room "
                       "— probing the escape window inside the seed T-drift "
                       "rule's claimed region.",
        "pattern": "lateral_crossing",
        "template": {"road_topology": "Intersection",
                     "threat_type": "Large Truck", "from_side": "left",
                     "pedestrian": {"present": False, "side": "left"},
                     "road_boundary_left": None, "road_boundary_right": None,
                     "follower": False, "blocked_side": None},
        "parameters": [
            {"name": "ego_speed", "min": 14, "max": 16, "steps": 2},
            {"name": "ttc", "min": 1.1, "max": 1.7, "steps": 3},
            {"name": "lateral_speed", "min": 7, "max": 9, "steps": 2},
        ],
        "hypothesis": "With enough lateral room a sharp far-side lane change "
                      "escapes cleanly, so the T-drift seed rule overclaims here.",
    },
    {
        "family_id": "mock-cut-in-right",
        "description": "Mock family: a slower SUV cuts in from the right lane.",
        "pattern": "cut_in",
        "template": {"road_topology": "Normal Road",
                     "threat_type": "SUV", "from_side": "right",
                     "pedestrian": {"present": False, "side": "right"},
                     "road_boundary_left": -1.5, "road_boundary_right": 10,
                     "follower": False, "blocked_side": None},
        "parameters": [
            {"name": "ego_speed", "min": 18, "max": 22, "steps": 2},
            {"name": "gap", "min": 8, "max": 16, "steps": 3},
            {"name": "speed_diff", "min": 3, "max": 6, "steps": 2},
        ],
        "hypothesis": "Braking dominates close cut-ins; no intervention "
                      "suffices at long gaps.",
    },
]


class MockLLM:
    """--mock 模式的确定性替身：同接口、零网络，供离线全流程自测。"""

    def __init__(self):
        self._generator_calls = 0   # 生成器角色的调用计数（场景族轮换用）

    async def chat(self, role: str, messages: List[dict],
                   response_schema: dict, max_tokens: int = 1200,
                   timeout_s: float = 300.0) -> LLMResult:
        text = "\n".join(str(m.get("content", "")) for m in messages)
        payload = self._respond(role, text)
        return LLMResult(content=json.dumps(payload), parsed=payload,
                         usage={"prompt_tokens": 0, "completion_tokens": 0},
                         elapsed=0.001, model=f"mock-{role}")

    def _respond(self, role: str, text: str) -> dict:
        """按角色返回确定性的合理响应（部分角色会解析提示词中的标记）。"""
        if role == "generator":
            # 轮换返回预置场景族，保证 --mock 批次的多样性
            family = _MOCK_FAMILIES[self._generator_calls % len(_MOCK_FAMILIES)]
            self._generator_calls += 1
            return json.loads(json.dumps(family))  # 深拷贝，防原型被就地修改
        if role == "consolidator":
            # 从提示词中提取规律摘要，交给无 LLM 的归纳基线
            m = re.search(r"LAWFUL_SUMMARY_JSON: (\{.*?\})\n", text, re.S)
            proposals = []
            if m:
                from .factory.induction_baseline import induce
                proposals = induce(json.loads(m.group(1)), "mock-family")
            return {"proposals": [{k: p[k] for k in ("guards", "policy", "rationale")}
                                  for p in proposals],
                    "amendments": [],
                    "analysis": "Mock consolidation via the no-LLM induction baseline."}
        if role == "decision":
            # 委员会提示词中带有 SIM_BEST 标记（仿真最优机动），mock 直接采纳
            m = re.search(r"SIM_BEST: (\d+)", text)
            best = int(m.group(1)) if m else 0
            return {"branches": [
                {"condition_type": "primary_threat_maintains",
                 "condition": "threat keeps closing", "code": best,
                 "target_speed_mps": 0.0,
                 "rationale": "Follow the simulator's least-harm ranking.",
                 "confidence": 0.8},
                {"condition_type": "primary_threat_yields",
                 "condition": "threat brakes or turns away", "code": 0,
                 "target_speed_mps": 0.0,
                 "rationale": "Braking suffices once the threat yields.",
                 "confidence": 0.85},
                {"condition_type": "default", "condition": "any other evolution",
                 "code": 0, "target_speed_mps": 0.0,
                 "rationale": "Minimal-risk fallback.", "confidence": 0.85}],
                "overall_rationale": "Mock committee decision grounded in the "
                                     "simulated severity table."}
        if role in ("advocate",):
            return {"argument": "Prioritize minimum worst-case harm; accept "
                                "maneuver cost to protect the most exposed party."}
        if role == "arbiter":
            return {"resolution": "Keep the lower-severity option; severity "
                                  "evidence outranks style preferences.",
                    "revised_codes": []}
        if role == "evaluation":
            return {"assessment": "The chosen maneuver matched the least-harm "
                                  "option available at this grid point.",
                    "decision_was_effective": True,
                    "lesson": "In this feature region, prefer the maneuver with "
                              "the lowest simulated severity; braking is not "
                              "automatically the safest choice."}
        return {"note": f"mock response for role {role}"}
