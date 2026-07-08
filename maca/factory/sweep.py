"""网格扫描 —— 规律发现的确定性实验引擎（无 LLM）。

对场景族参数网格的每个点，把全部 8 种机动逐一仿真并按严重度排序：
  - 某机动明显占优（margin >= LAW_MARGIN）或存在无害解的点 -> "规律格点"
    （lawful，规则归纳的原材料）；
  - 有害的严重度接近平局、或最优解仍伤及弱势道路使用者的点 -> "模糊格点"
    （ambiguous，交 LLM 委员会做权衡、产出案例）。

这套"仿真出真值、LLM 只做归纳与权衡"的分工是本架构对抗 LLM 幻觉的
核心手段之一。
"""
from typing import Dict, List

from .. import config
from ..runtime.features import extract_features, features_to_dict
from ..runtime.monitor import assess_scene
from ..tools import simulator
from . import instantiate


def run_family(family: dict) -> Dict:
    """扫描一个场景族的整个参数网格：逐点实例化、仿真全部机动、按规律/
    模糊分类，返回含全部点、lawful 子集、ambiguous 子集的结果。"""
    points: List[dict] = []
    for params in instantiate.grid_points(family):
        scenario = instantiate.instantiate(family, params)
        trigger = scenario.trigger
        assessment = assess_scene(trigger)
        rank = simulator.rank_maneuvers(trigger,
                                        primary_id=assessment.primary_threat)
        fv = extract_features(trigger, assessment)
        best = rank["best"]
        best_sev = rank["results"][best]["severity"]
        best_col = rank["results"][best].get("collision")
        vru_in_best = bool(best_col and best_col.get("vulnerable"))
        sevs = [r["severity"] for r in rank["results"].values()]
        # 绝境格点：连最优机动都是最高严重度、且所有机动都无区分度（多为
        # 极短 TTC 处"人人都撞"）。这类点学不到有价值的决策规律，标记为
        # doomed 单独归类，既不进规则归纳也不占用委员会案例配额。
        doomed = best_sev >= 4 and rank["margin"] == 0 and min(sevs) >= 4
        if doomed:
            kind = "doomed"
        elif best_sev <= 1:
            kind = "lawful"          # a clean resolution exists: no trade-off
        elif vru_in_best:
            kind = "ambiguous"       # even the best option harms a VRU: committee
        elif rank["margin"] >= config.LAW_MARGIN:
            kind = "lawful"          # one maneuver clearly dominates
        else:
            kind = "ambiguous"       # harmful near-ties: committee
        points.append({
            "params": params,
            "scenario_name": scenario.name,
            "kind": kind,
            "best": best,
            "best_severity": best_sev,
            "margin": rank["margin"],
            "severities": {c: r["severity"] for c, r in rank["results"].items()},
            "features": features_to_dict(fv),
        })
    return {"family_id": family.get("family_id", family["pattern"]),
            "pattern": family["pattern"], "family": family, "points": points,
            "lawful": [p for p in points if p["kind"] == "lawful"],
            "ambiguous": [p for p in points if p["kind"] == "ambiguous"],
            "doomed": [p for p in points if p["kind"] == "doomed"]}


def lawful_summary(sweep_result: Dict) -> Dict:
    """Machine-readable digest for rule induction: per best-code, the feature
    ranges over the lawful points it wins. Also used by the no-LLM induction
    baseline, so LLM proposals can always be cross-checked against it."""
    by_code: Dict[int, List[dict]] = {}
    for p in sweep_result["lawful"]:
        by_code.setdefault(p["best"], []).append(p)
    summary = {}
    for code, pts in by_code.items():
        numeric_ranges = {}
        for key in pts[0]["features"]["numeric"]:
            vals = [pt["features"]["numeric"][key] for pt in pts]
            numeric_ranges[key] = {"min": round(min(vals), 3),
                                   "max": round(max(vals), 3)}
        discrete_values = {}
        for key in pts[0]["features"]["discrete"]:
            discrete_values[key] = sorted({str(pt["features"]["discrete"][key])
                                           for pt in pts})
        summary[str(code)] = {"points": len(pts),
                              "numeric_ranges": numeric_ranges,
                              "discrete_values": discrete_values,
                              "mean_margin": round(sum(pt["margin"] for pt in pts) / len(pts), 2)}
    return summary


def grid_table(sweep_result: Dict, max_rows: int = 40) -> str:
    """Compact text table of the sweep for the consolidator prompt."""
    rows = ["params | best | margin | severities(code:sev) | ttc | lat_dist | lat_speed | sector"]
    for p in sweep_result["points"][:max_rows]:
        n = p["features"]["numeric"]
        sev = ",".join(f"{c}:{s}" for c, s in sorted(p["severities"].items()))
        rows.append(
            f"{p['params']} | {p['best']} ({p['kind']}) | {p['margin']} | {sev} | "
            f"{n['ttc_s']} | {n['lateral_dist_m']} | {n['lateral_speed_mps']} | "
            f"{p['features']['discrete']['approach_sector']}")
    return "\n".join(rows)
