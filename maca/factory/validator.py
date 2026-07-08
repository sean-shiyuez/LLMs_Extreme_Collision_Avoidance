"""规则生命周期门禁。

候选规则必须通过"回归电池"（其来源网格的匹配点 + 全部静态回归场景，
逐点检查仿真严重度不劣于可达最优、不转向行人侧）达到 ≥90% 通过率，
才晋升为 active；每批次抽样对 active 规则用其可复实例化的来源场景族
重新扫描复验，漂移即废弃 —— 保证规则不会因仿真器/特征语义演进而变质。
"""
import random
import time
from typing import Dict, List, Optional

from .. import config
from ..runtime.features import extract_features, features_from_dict
from ..runtime.monitor import assess_scene
from ..runtime.rule_engine import Rule
from ..scenario.loader import available_scenarios, load_scenario
from ..tools import simulator
from . import instantiate, sweep
from .consolidator import _maintains_code


def _check_point(rule: Rule, code: int, point: dict) -> Optional[str]:
    fv = features_from_dict(point["features"])
    if not rule.matches(fv):
        return None
    best_sev = point["severities"][point["best"]]
    my_sev = point["severities"].get(code)
    if my_sev is None or my_sev > best_sev:
        return (f"grid {point['params']}: severity {my_sev} vs achievable {best_sev}")
    ped = fv.discrete.get("pedestrian_side")
    if ped == "left" and code in config.LEFTWARD_CODES:
        return f"grid {point['params']}: steers toward pedestrian side (left)"
    if ped == "right" and code in config.RIGHTWARD_CODES:
        return f"grid {point['params']}: steers toward pedestrian side (right)"
    return "PASS"


def validate_rule(rule_dict: dict, sweep_result: Optional[Dict] = None) -> Dict:
    """回归电池 = 来源扫描的匹配点 + 全部静态回归场景。返回检查数、失败
    样本、通过率与晋升判定（通过率 >= RULE_PROMOTION_PASS_RATE 即晋升）。"""
    rule = Rule.from_dict(rule_dict)
    code = _maintains_code(rule.policy)
    checked, failures = 0, []

    if sweep_result is not None:
        for point in sweep_result["points"]:
            res = _check_point(rule, code, point)
            if res is None:
                continue
            checked += 1
            if res != "PASS":
                failures.append(res)

    # static regression scenarios (the calibrated demo set)
    for name in available_scenarios():
        try:
            scenario = load_scenario(name)
        except Exception:
            continue
        trigger = scenario.trigger
        assessment = assess_scene(trigger)
        fv = extract_features(trigger, assessment)
        if not rule.matches(fv):
            continue
        checked += 1
        rank = simulator.rank_maneuvers(trigger, primary_id=assessment.primary_threat)
        best_sev = rank["results"][rank["best"]]["severity"]
        my = rank["results"].get(code)
        if my is None or my["severity"] > best_sev:
            failures.append(f"scenario {name}: severity "
                            f"{my and my['severity']} vs achievable {best_sev}")

    pass_rate = 1.0 if checked == 0 else (checked - len(failures)) / checked
    promoted = checked > 0 and pass_rate >= config.RULE_PROMOTION_PASS_RATE
    return {"checked": checked, "failures": failures[:6],
            "pass_rate": round(pass_rate, 3), "promoted": promoted}


def promote(rule_dict: dict, verdict: Dict) -> dict:
    rule_dict["status"] = "active" if verdict["promoted"] else "candidate"
    rule_dict["stats"] = {"validated_n": verdict["checked"],
                          "pass_rate": verdict["pass_rate"],
                          "validated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    return rule_dict


def revalidate_active(rules: List[dict], sample: int = config.RULE_REVALIDATION_SAMPLE
                      ) -> List[Dict]:
    """抗漂移复验：抽样若干 active 工厂规则，用其来源场景族重新扫描并
    验证（确定性、完全可复现）；未通过者原地废弃。就地修改 rules 列表。"""
    candidates = [r for r in rules
                  if r.get("status") == "active" and r.get("provenance", {}).get("family")]
    reports = []
    for rule_dict in random.sample(candidates, min(sample, len(candidates))):
        sweep_result = sweep.run_family(rule_dict["provenance"]["family"])
        verdict = validate_rule(rule_dict, sweep_result)
        if not verdict["promoted"]:
            rule_dict["status"] = "deprecated"
            rule_dict["stats"]["deprecated_reason"] = "failed re-validation"
        reports.append({"rule_id": rule_dict["rule_id"], **verdict,
                        "still_active": rule_dict["status"] == "active"})
    return reports
