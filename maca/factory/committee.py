"""深思委员会 —— 离线 LLM 推理，只处理两类场景：扫描未发现明显规律的
"模糊格点"（有害的严重度接近平局、涉及弱势道路使用者的权衡），以及
运行时反馈队列中的 gap / 冲突项。

流水线：决策 -> （必要时辩论；冲突项强制辩论）-> 确定性安全校验层
（逐分支在其自身演化假设下仿真严重度 + 硬规则 + HJ 参考约束）->
否决重议回环 -> 反思评估。通过验证的结果落为一条 CASE
（特征向量 + 应急策略树 + 经验教训）。

关键分工：仿真严重度表是物理真值，LLM 的增量价值只在"权衡判断"
（该少伤谁、机动激进到什么程度合理）。
"""
import json
import time
from typing import Optional

from .. import config
from ..runtime.features import FeatureVector, extract_features, features_to_dict
from ..runtime.monitor import assess_scene
from ..scenario.schema import Snapshot
from ..tools import simulator
from . import schemas

# 策略树分支守卫 -> 安全校验时对主威胁采用的演化假设
# （校验每个分支时，都在"该分支所声称的演化"下仿真，而非统一按当前观测）
_BEHAVIOR_OF_BRANCH = {
    "primary_threat_maintains": "maintains",
    "primary_threat_yields": "yields",
    "primary_threat_accelerates": "accelerates",
    "secondary_threat_activates": "maintains",
    "default": "maintains",
}

_DECISION_SYSTEM = f"""\
You are the decision agent of the offline law-discovery factory of a
collision-avoidance system. This scene was routed to you because simulation
found NO clearly dominant maneuver (close severities, or a vulnerable road
user in the trade-off) — your job is the judgement call, grounded in the
simulated evidence. Output a contingency policy tree; it will be stored as a
CASE and matched online in milliseconds, so cover the listed guard
conditions (always include "default").

Maneuver catalogue:
{chr(10).join(f'  {k} = {v}' for k, v in config.DECISION_CODES.items())}

Domain knowledge:
{config.DOMAIN_KNOWLEDGE}

Rules of engagement:
1. The simulated severity table is ground truth for physics; do not invent
   different physics. Your added value is the trade-off judgement (whom to
   endanger less, how much maneuver aggressiveness is justified).
2. Protect vulnerable road users with absolute priority.
3. For codes 3/4 set target_speed_mps (else 0).
4. If a safety veto is quoted, the vetoed branch codes are forbidden.
"""

_SAFETY_NOTE = ("branch severity exceeds the best achievable severity for its "
                "evolution hypothesis")


class Committee:
    def __init__(self, llm):
        self.llm = llm

    async def deliberate(self, snapshot: Snapshot, source: str,
                         conflict_details: Optional[dict] = None,
                         scenario_name: str = "") -> dict:
        """对单个触发快照跑一遍完整委员会流程，返回含已验证策略、案例
        载荷与全程 trace 的记录。冲突项（conflict_details 非空）会强制辩论。"""
        assessment = assess_scene(snapshot)
        fv = extract_features(snapshot, assessment)
        rank = simulator.rank_maneuvers(snapshot,
                                        primary_id=assessment.primary_threat)
        trace, tokens = [], {"prompt_tokens": 0, "completion_tokens": 0}

        context = self._context(snapshot, fv, rank, conflict_details)
        feedback, policy, verdict = "", None, None
        for round_idx in range(1, config.DELIBERATION_MAX_ROUNDS + 1):
            user = context + (f"\n\nSAFETY VETO (round {round_idx - 1}):\n{feedback}"
                              if feedback else "")
            result = await self.llm.chat(
                "decision",
                [{"role": "system", "content": _DECISION_SYSTEM},
                 {"role": "user", "content": user}],
                response_schema=schemas.POLICY_SCHEMA)
            policy = result.parsed
            _tally(tokens, result)
            trace.append({"agent": "decision", "round": round_idx,
                          "model": result.model, "elapsed_s": round(result.elapsed, 2),
                          "policy": policy})

            if round_idx == 1 and (conflict_details or self._debate_needed(policy, rank)):
                policy = await self._debate(context, policy, conflict_details,
                                            trace, tokens)

            verdict = self._safety_layer(snapshot, assessment, fv, policy, rank)
            trace.append({"agent": "safety_layer", "round": round_idx, **verdict})
            if verdict["approved"]:
                break
            feedback = verdict["veto_reason"]

        if verdict and not verdict["approved"]:
            self._harden(snapshot, assessment, policy, verdict)
            trace.append({"agent": "safety_layer",
                          "note": "hardened: failing branches replaced by the "
                                  "simulator's least-harm maneuver"})

        maintains = next((b for b in policy["branches"]
                          if b["condition_type"] == "primary_threat_maintains"),
                         policy["branches"][-1])
        exec_sim = simulator.simulate_maneuver(
            snapshot, int(maintains["code"]),
            maintains.get("target_speed_mps") or None,
            primary_id=assessment.primary_threat)

        evaluation = await self.llm.chat(
            "evaluation",
            [{"role": "system", "content":
                "You are the evaluation agent of a collision-avoidance law "
                "factory. Given the scene, the committee's policy and the "
                "simulated outcome, write a transferable lesson (it will be "
                "retrieved verbatim in future similar scenes) and judge "
                "effectiveness relative to what was achievable."},
             {"role": "user", "content":
                 context + "\n\nCommittee policy:\n" + json.dumps(policy)
                 + f"\n\nSimulated outcome (maintains branch): {exec_sim['outcome']}"
                 + f"\nBest achievable severity: {rank['results'][rank['best']]['severity']}"}],
            response_schema=schemas.EVALUATION_SCHEMA)
        _tally(tokens, evaluation)
        trace.append({"agent": "evaluation", "report": evaluation.parsed})

        self._export_distillation(context, policy)
        return {
            "source": source,
            "scenario_name": scenario_name,
            "features": features_to_dict(fv),
            "policy": policy,
            "executed_code": int(maintains["code"]),
            "severity": exec_sim["severity"],
            "outcome": exec_sim["outcome"],
            "lesson": evaluation.parsed["lesson"],
            "effective": bool(evaluation.parsed["decision_was_effective"]),
            "trace": trace,
            "tokens": tokens,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _context(snapshot, fv: FeatureVector, rank, conflict_details) -> str:
        """组装决策提示词的上下文：场景文本 + 特征 + 仿真严重度真值表
        + SIM_BEST（仿真最优机动）+（若为冲突项）待裁决的仲裁详情。"""
        sev_table = {str(c): {"severity": r["severity"],
                              "outcome": r["outcome"]}
                     for c, r in rank["results"].items()}
        parts = [
            "SCENE:\n" + snapshot.compact_text(),
            "FEATURES:\n" + json.dumps({"discrete": fv.discrete,
                                        "numeric": {k: round(v, 2) for k, v
                                                    in fv.numeric.items()}}),
            "SIMULATED SEVERITY TABLE (ground truth):\n" + json.dumps(sev_table),
            f"SIM_BEST: {rank['best']}",
        ]
        if conflict_details:
            parts.append("RUNTIME CONFLICT UNDER ARBITRATION:\n"
                         + json.dumps(conflict_details))
        return "\n\n".join(parts)

    @staticmethod
    def _debate_needed(policy, rank) -> bool:
        """一致性门控：仅当决策与仿真最优机动分歧、或分支置信度偏低时
        才触发辩论 —— 避免无谓辩论浪费 token。"""
        maintains = next((b for b in policy.get("branches", [])
                          if b["condition_type"] == "primary_threat_maintains"), None)
        disagrees = maintains is not None and int(maintains["code"]) != rank["best"]
        low_conf = min((b.get("confidence", 1.0)
                        for b in policy.get("branches", [])), default=1.0) < 0.6
        return disagrees or low_conf

    async def _debate(self, context, policy, conflict_details, trace, tokens):
        """安全 vs 效率两方倡导 + 仲裁者裁决（少步数推理）；仲裁的分支
        码修订就地写回策略树。"""
        brief = context + "\n\nPolicy under debate:\n" + json.dumps(policy)
        sides = {}
        for role, stance in (("safety", "Argue for minimizing worst-case harm and "
                                        "protecting vulnerable road users absolutely."),
                             ("efficiency", "Argue against overreaction: unnecessary "
                                            "emergency maneuvers cause secondary "
                                            "accidents and false triggering.")):
            result = await self.llm.chat(
                "advocate",
                [{"role": "system", "content": f"You are the {role.upper()} advocate "
                  f"in a collision-avoidance debate. {stance}"},
                 {"role": "user", "content": brief}],
                response_schema=schemas.ADVOCATE_SCHEMA, max_tokens=300)
            sides[role] = result.parsed["argument"]
            _tally(tokens, result)
        arbiter = await self.llm.chat(
            "arbiter",
            [{"role": "system", "content":
                "You are the ARBITER of a collision-avoidance debate. Resolve the "
                "disagreement with few reasoning steps, grounded in the simulated "
                "severity table (which is ground truth). List branch code revisions "
                "in revised_codes (empty if none). Severity evidence outranks "
                "style; when severities tie, choose the more conservative option."},
             {"role": "user", "content": brief
              + "\n\nSAFETY ADVOCATE: " + sides["safety"]
              + "\n\nEFFICIENCY ADVOCATE: " + sides["efficiency"]}],
            response_schema=schemas.ARBITER_SCHEMA, max_tokens=400)
        _tally(tokens, arbiter)
        revisions = {r["condition_type"]: r["code"]
                     for r in arbiter.parsed["revised_codes"]}
        for b in policy.get("branches", []):
            if b["condition_type"] in revisions:
                b["code"] = revisions[b["condition_type"]]
                b["rationale"] += " [revised by debate arbiter]"
        trace.append({"agent": "debate", "advocates": sides,
                      "resolution": arbiter.parsed["resolution"],
                      "revisions": revisions})
        return policy

    def _safety_layer(self, snapshot, assessment, fv: FeatureVector, policy, rank):
        """确定性安全校验层：逐分支在其自身演化假设下仿真，检查三类违规
        —— (1) 严重度劣于该假设下可达最优；(2) 转向行人所在侧；
        (3) 变道类机动末态 HJ 风险劣于不干预基线。任一违规该分支被否决。"""
        verdicts, ok_all = [], True
        for b in policy.get("branches", []):
            code = int(b["code"])
            behavior = _BEHAVIOR_OF_BRANCH[b["condition_type"]]
            sim = simulator.simulate_maneuver(
                snapshot, code, b.get("target_speed_mps") or None,
                primary_id=assessment.primary_threat, primary_behavior=behavior)
            best = simulator.rank_maneuvers(
                snapshot, primary_id=assessment.primary_threat,
                primary_behavior=behavior) if behavior != "maintains" else rank
            violations = []
            if sim["severity"] > best["results"][best["best"]]["severity"]:
                violations.append(
                    f"{_SAFETY_NOTE}: severity {sim['severity']} vs achievable "
                    f"{best['results'][best['best']]['severity']} (code {best['best']})")
            ped = fv.discrete.get("pedestrian_side")
            if ped == "left" and code in config.LEFTWARD_CODES:
                violations.append("steers toward the pedestrian-occupied left side")
            if ped == "right" and code in config.RIGHTWARD_CODES:
                violations.append("steers toward the pedestrian-occupied right side")
            if code not in (0, 5, 6, 7) and behavior == "maintains":
                hj = sim.get("hj") or simulator.simulate_maneuver(
                    snapshot, code, b.get("target_speed_mps") or None,
                    primary_id=assessment.primary_threat)["hj"]
                if not hj["acceptable"]:
                    violations.append(
                        f"HJ reference constraint: end-state risk {hj['worst_end_risk']}"
                        f" worsens the no-intervention baseline {hj['baseline_risk']}")
            ok_all &= not violations
            verdicts.append({"condition_type": b["condition_type"], "code": code,
                             "behavior": behavior, "severity": sim["severity"],
                             "approved": not violations, "violations": violations})
        reason = "; ".join(
            f"branch[{v['condition_type']}] code {v['code']}: {'; '.join(v['violations'])}"
            for v in verdicts if not v["approved"])
        return {"approved": ok_all, "veto_reason": reason,
                "branch_verdicts": verdicts}

    @staticmethod
    def _harden(snapshot, assessment, policy, verdict):
        """兜底：达到最大重议轮数仍有分支未过 -> 把未过分支替换为该演化
        假设下仿真的最小伤害机动（安全性由物理保证，不再依赖 LLM）。"""
        failing = {v["condition_type"]: v for v in verdict["branch_verdicts"]
                   if not v["approved"]}
        for b in policy.get("branches", []):
            v = failing.get(b["condition_type"])
            if v is None:
                continue
            best = simulator.rank_maneuvers(
                snapshot, primary_id=assessment.primary_threat,
                primary_behavior=v["behavior"])["best"]
            b["code"] = best
            b["target_speed_mps"] = 0.0
            b["rationale"] = ("Hardened to the simulator's least-harm maneuver "
                              "after repeated vetoes: " + b["rationale"])

    @staticmethod
    def _export_distillation(context: str, policy: dict):
        """把本次深思导出为一条微调样本（"慢教快"的数据出口，追加到
        maca_distill.jsonl；格式对齐 legacy 的 messages 微调格式）。"""
        sample = {"messages": [
            {"role": "system",
             "content": _DECISION_SYSTEM.split("Rules of engagement")[0].strip()},
            {"role": "user", "content": context},
            {"role": "assistant", "content": json.dumps(policy)},
        ]}
        config.DISTILL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.DISTILL_EXPORT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _tally(tokens: dict, result):
    tokens["prompt_tokens"] += result.usage.get("prompt_tokens", 0)
    tokens["completion_tokens"] += result.usage.get("completion_tokens", 0)
