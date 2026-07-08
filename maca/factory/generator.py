"""场景族生成器 —— 高思考 LLM 设计参数化实验（从不产出原始坐标：
实例化由 instantiate.py 确定性完成，物理合理性由构造保证）。

生成后经 _sanitize 做确定性护栏（补齐必需参数、把范围裁进合理区间、
限制网格步数），防止 LLM 设计出越界或退化的实验。
"""
import json
from typing import Dict, List, Optional

from . import schemas

_SANE_RANGES = {
    "ego_speed": (5.0, 30.0), "ttc": (0.4, 2.4), "lateral_speed": (2.0, 12.0),
    "gap": (3.0, 60.0), "speed_diff": (0.0, 12.0), "lateral_offset": (3.0, 6.0),
}
_REQUIRED_PARAMS = {
    "lateral_crossing": ["ego_speed", "ttc", "lateral_speed"],
    "lead_braking": ["ego_speed", "gap", "speed_diff"],
    "cut_in": ["ego_speed", "gap", "speed_diff"],
    "static_blockage": ["ego_speed", "gap"],
}

_SYSTEM = """\
You are the scenario-generation agent of a collision-avoidance law factory.
Design ONE parameterized scenario family — an experiment whose swept grid is
expected to reveal a decision law (which maneuver dominates which region).

You choose: the pattern, the context template (road topology, threat type,
approach side, pedestrian presence/side, road boundaries, a blocked corridor,
a follower behind the ego), the swept parameters with ranges and step counts,
and a one-line hypothesis. Instantiation is deterministic: threats are placed
on a collision course by construction.

Design principles:
1. Target the UNCOVERED or CONTESTED region described in the briefing — do
   not redesign what active rules already cover; probe their boundaries or
   their blind spots (e.g. same geometry but with a pedestrian on the escape
   side, or with the escape corridor blocked).
2. Sweep the parameters the decision boundary should depend on (TTC, closing
   speed, gap, ego speed), 2-4 steps each, <=36 points total.
3. Prefer the DECISION-RICH regime, not only the doomed one. At extremely
   short TTC (~0.4 s) every maneuver collides, so the only lesson is "brake
   and take it on the front" — low information. The interesting laws live in
   the ACTIONABLE window where different maneuvers yield different severities:
   sweep TTC roughly over 0.6–1.6 s (you may include a 0.4 s lower edge, but
   do NOT pin the whole family there), so braking, lane changes and T-drift
   genuinely compete and a real decision boundary emerges.
"""


async def design_family(llm, briefing: str,
                        queue_item: Optional[dict] = None) -> Dict:
    """让生成器设计一个场景族。briefing 描述当前覆盖状态；若给定
    queue_item（运行时缺口/冲突），则要求围绕其特征复现并泛化。"""
    user = briefing
    if queue_item is not None:
        user += ("\n\nThis family must reproduce and generalize the following "
                 "runtime queue item (design the sweep around its features):\n"
                 + json.dumps({k: queue_item[k] for k in ("kind", "features", "details")
                               if k in queue_item}))
    result = await llm.chat(
        "generator",
        [{"role": "system", "content": _SYSTEM},
         {"role": "user", "content": user}],
        response_schema=schemas.FAMILY_SCHEMA, max_tokens=1200)
    family = result.parsed
    family["_generator_meta"] = {"model": result.model,
                                 "elapsed_s": round(result.elapsed, 2),
                                 "usage": result.usage}
    return _sanitize(family)


def _sanitize(family: Dict) -> Dict:
    """对 LLM 的设计做确定性护栏：补齐该模式必需的参数、把每个参数范围
    裁进 _SANE_RANGES、限制网格步数在 [2,4]。"""
    pattern = family["pattern"]
    required = _REQUIRED_PARAMS[pattern]
    params = {p["name"]: p for p in family.get("parameters", [])}
    for name in required:
        if name not in params:
            lo, hi = _SANE_RANGES[name]
            params[name] = {"name": name, "min": lo, "max": hi, "steps": 3}
    clean: List[dict] = []
    for name, p in params.items():
        if name not in _SANE_RANGES:
            continue
        lo, hi = _SANE_RANGES[name]
        pmin = max(lo, min(float(p["min"]), hi))
        pmax = max(pmin + 0.1, min(float(p["max"]), hi))
        clean.append({"name": name, "min": round(pmin, 2), "max": round(pmax, 2),
                      "steps": max(2, min(int(p.get("steps", 3)), 4))})
    family["parameters"] = clean
    return family


def coverage_briefing(rules: List[dict], stats: Dict) -> str:
    """组装给生成器的"覆盖简报"：库统计 + 现役规则覆盖区域 + 各场景模式
    的规则计数，引导它去探测无规则覆盖的模式/特征空间，或压测既有规则的
    边界条件。"""
    active = [{"rule_id": r["rule_id"], "guards": r["guards"]}
              for r in rules if r.get("status") == "active"]
    # 统计每种模式已有多少条现役规则，明确点出尚未探索的模式
    all_patterns = ["lateral_crossing", "lead_braking", "cut_in", "static_blockage"]
    per_pattern = {p: 0 for p in all_patterns}
    for r in rules:
        if r.get("status") != "active":
            continue
        fam = r.get("provenance", {}).get("family", {})
        pat = fam.get("pattern")
        if pat in per_pattern:
            per_pattern[pat] += 1
    unexplored = [p for p, n in per_pattern.items() if n == 0]
    return (
        "LIBRARY STATE:\n" + json.dumps(stats)
        + "\n\nRULES PER PATTERN:\n" + json.dumps(per_pattern)
        + (f"\n\nUNEXPLORED PATTERNS (no active rule yet): {unexplored}. "
           "STRONGLY PREFER designing a family in one of these patterns to widen "
           "coverage." if unexplored else "")
        + "\n\nACTIVE RULE REGIONS:\n" + json.dumps(active)
        + "\n\nDesign a family probing feature-space regions no active rule "
          "covers, or an unexplored pattern above, or stress-testing an active "
          "rule's boundary conditions."
    )
