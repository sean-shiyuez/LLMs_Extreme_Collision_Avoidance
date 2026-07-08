"""新颖性门控 —— 工厂只在"库尚未正确处理"的场景上花 LLM token。

"已覆盖"的定义：某个 active 规则命中特征、且其 maintains 机动码达到
该点最优严重度；或某个案例（相似度过阈值）做到同样的事。
注意："规则命中但严重度次优"不算覆盖 —— 它是一个反例，会回喂给
归纳器用于收紧/修订规则。这是不完美种子规则被证据自动纠正的机制。

这是对用户"每个案例都必须未命中才激活离线工厂"要求的直接实现。
"""
from typing import Dict, List, Optional

from ..runtime.case_index import CaseLibrary
from ..runtime.features import FeatureVector
from ..runtime.rule_engine import RuleLibrary
from .consolidator import _maintains_code


def check_coverage(fv: FeatureVector, best_severity: int,
                   severities: Dict[int, int],
                   rules: RuleLibrary, cases: CaseLibrary,
                   count_cases: bool = True) -> Dict:
    """检查一个特征点是否已被库正确覆盖。
    severities: 机动码 -> 该点仿真严重度。
    返回 {covered, by, rule_suboptimal}；rule_suboptimal 收集"命中但次优"
    的规则（反例）。count_cases=False 时只看规则（网格归纳门控用）。"""
    result = {"covered": False, "by": None, "rule_suboptimal": []}

    rule, matches, _ = rules.match(fv)
    for r in matches:
        code = _maintains_code(r.policy)
        sev = severities.get(code)
        if sev is not None and sev <= best_severity:
            result.update(covered=True, by=f"rule:{r.rule_id}")
            break
        result["rule_suboptimal"].append(
            {"rule_id": r.rule_id, "code": code, "severity": sev,
             "best_severity": best_severity})

    if not result["covered"] and count_cases:
        best_case, _, _ = cases.knn(fv)
        if best_case is not None:
            code = _maintains_code(best_case["policy"])
            sev = severities.get(code)
            if sev is not None and sev <= best_severity:
                result.update(covered=True, by=f"case:{best_case['case_id']}")

    return result


def severities_of(rank: Dict) -> Dict[int, int]:
    return {c: r["severity"] for c, r in rank["results"].items()}
