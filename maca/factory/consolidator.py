"""规则归纳 —— 思考型 LLM 把已标注的扫描网格提炼成通用规则。

每条提案在成为候选之前，都必须逐点对照网格做数值复核：规则守卫覆盖
的每个规律格点，其 maintains 分支的机动码都必须在该点的最优机动之列，
否则打回重提（最多 CONSOLIDATOR_MAX_ROUNDS 轮）。LLM 的幻觉边界
在这一步被数值交叉验证拦下 —— 这是"用仿真真值约束 LLM"的关键环节。
"""
import json
from typing import Dict, List

from .. import config
from ..runtime.features import features_from_dict
from ..runtime.rule_engine import Rule
from . import schemas, sweep

_SYSTEM = f"""\
You are the rule-induction agent of a collision-avoidance law factory. You
receive a scenario family, its swept parameter grid — every point labeled
with the simulated severity of ALL maneuver codes and the dominant (best)
maneuver — plus the currently active rules that overlap this region.

Induce GENERAL RULES: guards over the feature language (discrete membership
lists and numeric min/max ranges) plus a contingency policy tree whose
primary_threat_maintains branch is the dominant maneuver of the region.

Feature language (guard keys):
  discrete: road_topology, threat_kind, approach_sector, pedestrian_side,
            corridor_left_free, corridor_right_free, braking_sufficient,
            on_collision_course
  numeric:  ttc_s, miss_m, lateral_dist_m, lateral_speed_mps,
            closing_speed_mps, ego_speed_mps, hj_risk, threat_dist_m

Maneuver catalogue:
{chr(10).join(f'  {k} = {v}' for k, v in config.DECISION_CODES.items())}

Requirements:
1. Numeric guard bounds must be supported by the lawful grid points — every
   lawful point inside your guards must have your maintains-code among its
   minimum-severity maneuvers. Your proposals are cross-checked numerically;
   unsupported bounds will be rejected back to you.
2. Prefer FEW, WIDE rules over many narrow ones, but never widen past what
   the grid supports.
3. Each rule needs at least 3 supporting lawful points.
4. If an existing rule is contradicted by this grid, amend it (tighten or
   deprecate) with the evidence in `reason`.
5. Always include yields and default branches (usually code 0, or 7 when no
   intervention is genuinely safe).
"""


async def consolidate(llm, sweep_result: Dict, existing_rules: List[dict],
                      conflict_notes: List[dict] = ()) -> Dict:
    """对一个扫描结果做归纳：把规律摘要 + 网格表 + 重叠既有规则 +（可选的）
    规则反例喂给 LLM，逐提案数值复核，返回 {accepted, rejected, amendments}。"""
    summary = sweep.lawful_summary(sweep_result)
    table = sweep.grid_table(sweep_result)
    overlap = [{"rule_id": r["rule_id"], "status": r["status"],
                "guards": r["guards"],
                "maintains_code": _maintains_code(r["policy"])}
               for r in existing_rules if r.get("status") != "deprecated"]
    user = (
        f"FAMILY: {json.dumps({k: sweep_result['family'][k] for k in ('family_id', 'pattern', 'template', 'hypothesis') if k in sweep_result['family']})}\n\n"
        f"SWEPT GRID ({len(sweep_result['points'])} points, "
        f"{len(sweep_result['lawful'])} lawful / {len(sweep_result['ambiguous'])} ambiguous):\n"
        f"{table}\n\n"
        f"LAWFUL_SUMMARY_JSON: {json.dumps(summary)}\n\n"
        f"EXISTING RULES IN THIS REGION:\n{json.dumps(overlap)}\n"
        + (f"\nRUNTIME CONFLICT NOTES:\n{json.dumps(list(conflict_notes))}\n"
           if conflict_notes else "")
    )

    accepted, rejected, trace = [], [], []
    feedback = ""
    for round_idx in range(1, config.CONSOLIDATOR_MAX_ROUNDS + 1):
        result = await llm.chat(
            "consolidator",
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": user + (f"\nREJECTED LAST ROUND:\n{feedback}"
                                                 if feedback else "")}],
            response_schema=schemas.RULE_PROPOSALS_SCHEMA, max_tokens=2500)
        trace.append({"agent": "consolidator", "round": round_idx,
                      "model": result.model, "elapsed_s": round(result.elapsed, 2),
                      "n_proposals": len(result.parsed["proposals"]),
                      "analysis": result.parsed["analysis"]})
        rejected_this_round = []
        for prop in result.parsed["proposals"]:
            prop["guards"] = schemas.guards_from_entries(prop["guards"])
            check = verify_proposal(prop, sweep_result)
            if check["ok"]:
                accepted.append({"proposal": prop, "verification": check})
            else:
                rejected_this_round.append({"proposal": prop, "verification": check})
        amendments = result.parsed.get("amendments", [])
        for am in amendments:
            if am.get("new_guards") is not None:
                am["new_guards"] = schemas.guards_from_entries(am["new_guards"])
        if not rejected_this_round or round_idx == config.CONSOLIDATOR_MAX_ROUNDS:
            rejected = rejected_this_round
            break
        feedback = json.dumps([{"guards": r["proposal"]["guards"],
                                "problems": r["verification"]["problems"]}
                               for r in rejected_this_round])

    return {"accepted": accepted, "rejected": rejected,
            "amendments": amendments, "trace": trace}


def verify_proposal(proposal: dict, sweep_result: Dict) -> Dict:
    """对归纳出的规则做数值交叉验证：守卫命中的每个网格点，其 maintains
    机动码都必须达到该点最优严重度；且至少命中 3 个点。返回是否通过、
    覆盖点数与问题列表（问题非空即打回）。"""
    rule = Rule(rule_id="_check", guards=proposal["guards"],
                policy=proposal["policy"])
    code = _maintains_code(proposal["policy"])
    problems, covered = [], 0
    if code is None:
        return {"ok": False, "covered": 0,
                "problems": ["policy has no primary_threat_maintains branch"]}
    for point in sweep_result["points"]:
        fv = features_from_dict(point["features"])
        if not rule.matches(fv):
            continue
        covered += 1
        best_sev = point["severities"][point["best"]]
        my_sev = point["severities"].get(code)
        if my_sev is None or my_sev > best_sev:
            problems.append(
                f"point {point['params']}: maintains code {code} has severity "
                f"{my_sev} but achievable severity is {best_sev} "
                f"(best code {point['best']})")
    if covered < 3:
        problems.append(f"only {covered} grid points match the guards (need >=3)")
    return {"ok": not problems, "covered": covered, "problems": problems[:6]}


def _maintains_code(policy: dict):
    b = next((b for b in policy.get("branches", [])
              if b["condition_type"] == "primary_threat_maintains"), None)
    return None if b is None else int(b["code"])
