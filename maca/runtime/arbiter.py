"""确定性保守仲裁 —— "辩论智能体的运行时化身"。

当规则结论与案例结论在即将执行的分支上不一致时，运行时不可能等待
LLM。它用本地物理直接把两个候选机动各仿真一遍，执行预测严重度更低的
那个（平局依次比接触能量、机动温和度）。冲突本身入队送离线辩论智能体
裁决，裁决结果反哺库使分歧不再复现（自愈）。
"""
import time
from typing import Tuple

from .. import config
from ..scenario.schema import Snapshot
from ..tools import simulator


def arbitrate(snapshot: Snapshot, rule_branch: dict, case_branch: dict
              ) -> Tuple[dict, str, dict, float]:
    """返回 (选中的分支, 选中方 "rule"/"case", 对比报告, 耗时 ms)。"""
    start = time.perf_counter()
    sims = {}
    for src, branch in (("rule", rule_branch), ("case", case_branch)):
        code = int(branch["code"])
        sim = simulator.simulate_maneuver(snapshot, code,
                                          branch.get("target_speed_mps") or None,
                                          _with_hj=False)
        col = sim.get("collision")
        sims[src] = {
            "code": code,
            "severity": sim["severity"],
            "contact_speed": col["relative_speed_mps"] if col else 0.0,
            "aggressiveness": config.AGGRESSIVENESS[code],
            "outcome": sim["outcome"],
        }
    # "保守" 的操作化定义：预测严重度低者 > 接触能量低者 > 更温和者
    key = lambda s: (sims[s]["severity"], sims[s]["contact_speed"],
                     sims[s]["aggressiveness"])
    chosen = min(("rule", "case"), key=key)
    report = {"rule": sims["rule"], "case": sims["case"], "chosen": chosen}
    branch = rule_branch if chosen == "rule" else case_branch
    return branch, chosen, report, (time.perf_counter() - start) * 1000
