"""在线决策执行器 —— 整条运行时关键路径，全程无 LLM。

触发时刻的分层决策：
  low 带                       -> code 7，直接放行（微秒级）
  规则命中 ∩ 案例命中且一致    -> 执行一致的分支
  规则 vs 案例分歧             -> 确定性保守仲裁（物理严重度比较），
                                  冲突入队送离线辩论
  仅一方命中                   -> 执行该方
  均未命中                     -> 最小风险机动（code 0）+ gap 入队

每一级都独立计时，总和即论文汇报的"触发路径时延"。
"""
import time
from typing import Optional

from .. import config
from ..scenario.schema import Scenario, snapshot_to_dict
from ..tools import simulator
from . import arbiter, gaps
from .case_index import CaseLibrary
from .features import extract_features, features_to_dict
from .monitor import assess_scene
from .policy_match import classify_evolution, select_branch
from .rule_engine import RuleLibrary

# 最小风险机动（MRM）兜底分支：库无覆盖时无条件可用，控制权永不悬空
MRM_BRANCH = {"condition_type": "default", "condition": "no library coverage",
              "code": 0, "target_speed_mps": 0.0,
              "rationale": "Minimal-risk maneuver: no verified rule or case "
                           "covers this scene.", "confidence": 1.0}


class RuntimeExecutor:
    """在线执行器：加载两个库后即可对任意场景毫秒级决策。"""

    def __init__(self, rules: Optional[RuleLibrary] = None,
                 cases: Optional[CaseLibrary] = None):
        self.rules = rules or RuleLibrary.load()
        self.cases = cases or CaseLibrary.load()

    def decide(self, scenario: Scenario) -> dict:
        """对一个场景（预览快照 + 触发快照）完成一次完整在线决策。"""
        prev, trigger = scenario.preview, scenario.trigger
        record = {"scenario": scenario.name, "libraries": {
            "rules_active": len(self.rules.active()),
            "cases": len(self.cases.cases)}}
        timings = {}

        # ---- 第 1 级：反射层风险监视（HJ + TTC，毫秒级）----
        t0 = time.perf_counter()
        assessment = assess_scene(trigger)
        timings["monitor_ms"] = _ms(t0)
        record["assessment"] = {
            "scene_risk": assessment.scene_risk, "band": assessment.band(),
            "min_ttc_s": assessment.min_ttc,
            "primary_threat": assessment.primary_threat}

        if assessment.band() == "low":   # 低风险：直接放行
            record.update(executed_code=7, decision_source="reflex_no_intervention",
                          branch=None, observed_condition="low_risk")
            return self._finish(record, trigger, timings, t_start=t0)

        # ---- 第 2 级：特征化（本地确定性计算）----
        t1 = time.perf_counter()
        fv = extract_features(trigger, assessment)
        timings["features_ms"] = _ms(t1)
        record["features"] = {"discrete": fv.discrete,
                              "numeric": {k: round(v, 3) for k, v in fv.numeric.items()}}

        # 提交条件：critical 带，或 elevated 带但主威胁在碰撞路径上且制动
        # 已不可解 —— "刹不出去"本身就是提交点，即使组合风险分未越线
        # （远距静态前方障碍的 HJ 评分偏低）。
        commit = assessment.band() == "critical" or (
            fv.discrete["on_collision_course"]
            and not fv.discrete["braking_sufficient"])
        if not commit:   # elevated 且尚可制动解：继续监视
            record.update(executed_code=7,
                          decision_source="reflex_monitoring_elevated",
                          branch=None, observed_condition="elevated_monitoring")
            return self._finish(record, trigger, timings, t_start=t0)

        # ---- 第 3 级：双库匹配（规则微秒级 + 案例 k-NN 毫秒级）----
        t2 = time.perf_counter()
        rule, rule_matches, rule_ms = self.rules.match(fv)
        timings["rule_match_ms"] = round(rule_ms, 3)
        case, case_top, case_ms = self.cases.knn(fv)
        timings["case_knn_ms"] = round(case_ms, 3)
        record["rule_hit"] = rule.rule_id if rule else None
        record["case_hit"] = case["case_id"] if case else None
        record["case_similarity"] = round(case_top[0][1], 4) if case_top else None

        # ---- 第 4 级：场景演化分类 -> 策略树选支 ----
        t3 = time.perf_counter()
        observed = classify_evolution(prev, trigger, assessment.primary_threat)
        timings["evolution_ms"] = _ms(t3)
        record["observed_condition"] = observed

        rule_branch = select_branch(rule.policy, observed) if rule else None
        case_branch = select_branch(case["policy"], observed) if case else None

        # ---- 第 5 级：一致执行 / 保守仲裁 / 单方执行 / MRM 兜底 ----
        if rule_branch and case_branch:
            if int(rule_branch["code"]) == int(case_branch["code"]):
                branch, source = rule_branch, f"rule[{rule.rule_id}]+case[{case['case_id']}] (agree)"
            else:
                # 规则与案例分歧：本地物理仲裁取保守者（毫秒级）
                branch, chosen, report, arb_ms = arbiter.arbitrate(
                    trigger, rule_branch, case_branch)
                timings["arbitration_ms"] = round(arb_ms, 3)
                source = (f"conservative_arbitration->{chosen} "
                          f"(rule {report['rule']['code']} vs case {report['case']['code']})")
                record["conflict"] = report
                # 若胜出的案例本身就是离线冲突裁决的产物，说明该分歧已被
                # 裁决过 —— 不再每次重复入队（败诉的规则等待修订）。
                adjudicated = (chosen == "case"
                               and case.get("source") == "conflict_resolution")
                if adjudicated:
                    record["conflict_adjudicated"] = True
                else:
                    gaps.enqueue("conflict", scenario.name, snapshot_to_dict(trigger),
                                 features_to_dict(fv),
                                 {"rule_id": rule.rule_id, "case_id": case["case_id"],
                                  "observed": observed, "arbitration": report})
        elif rule_branch:
            branch, source = rule_branch, f"rule[{rule.rule_id}]"
        elif case_branch:
            branch, source = case_branch, f"case[{case['case_id']}]"
        else:
            # 无任何覆盖：MRM 兜底 + gap 入队交给工厂补齐
            branch, source = dict(MRM_BRANCH), "mrm_fallback"
            gaps.enqueue("gap", scenario.name, snapshot_to_dict(trigger),
                         features_to_dict(fv),
                         {"observed": observed, "band": assessment.band()})
            record["gap_queued"] = True

        record.update(executed_code=int(branch["code"]), decision_source=source,
                      branch=branch)
        return self._finish(record, trigger, timings, t_start=t0)

    def _finish(self, record, trigger, timings, t_start):
        """收尾：结算触发路径总时延，并用仿真器评估执行结局（结局评估
        不在关键路径上，仅用于日志与验证）。"""
        timings["trigger_path_total_ms"] = _ms(t_start)
        record["timings"] = timings
        sim = simulator.simulate_maneuver(
            trigger, record["executed_code"],
            (record.get("branch") or {}).get("target_speed_mps") or None)
        record["outcome"] = {"severity": sim["severity"], "text": sim["outcome"],
                             "collision": sim["collision"],
                             "hj_constraint": sim.get("hj")}
        return record


def _ms(t0: float) -> float:
    """自 t0 起的耗时 [ms]，保留三位小数。"""
    return round((time.perf_counter() - t0) * 1000, 3)
