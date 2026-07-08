"""触发时刻的应急策略树分支匹配（纯运动学分类，亚毫秒级）。

策略树的分支守卫是封闭集合（config.BRANCH_CONDITIONS）：反射层比较
前后两个快照，把"场景实际如何演化"分类到该集合，再选中对应分支 ——
全程不解析任何自由文本，这是毫秒级执行的前提。
"""
import time
from typing import Optional, Tuple

from ..scenario.schema import Snapshot
from .monitor import assess_scene


def classify_evolution(prev: Snapshot, now: Snapshot,
                       primary_threat: Optional[str]) -> str:
    """按运动学对场景演化分类（对应策略树的分支守卫集合）。

    判定顺序：主威胁易主 -> secondary_threat_activates；
    主威胁消失/脱离碰撞路径 -> primary_threat_yields；
    速度显著增/减 -> accelerates / yields；否则 maintains。
    """
    now_assess = assess_scene(now)
    if primary_threat and now_assess.primary_threat and \
            now_assess.primary_threat != primary_threat:
        return "secondary_threat_activates"

    prev_t = {tid: (pos, vel) for tid, pos, vel, _ in prev.targets()}
    now_t = {tid: (pos, vel) for tid, pos, vel, _ in now.targets()}
    if primary_threat not in now_t:
        return "primary_threat_yields"
    per = now_assess.per_target.get(primary_threat, {})
    if not per.get("on_collision_course", False):
        return "primary_threat_yields"
    if primary_threat in prev_t:
        (_, pv), (_, nv) = prev_t[primary_threat], now_t[primary_threat]
        prev_speed = (pv[0] ** 2 + pv[1] ** 2) ** 0.5
        now_speed = (nv[0] ** 2 + nv[1] ** 2) ** 0.5
        if now_speed > prev_speed * 1.1 + 0.5:      # 提速超 10%+0.5m/s
            return "primary_threat_accelerates"
        if now_speed < prev_speed * 0.7 - 0.5:      # 降速超 30%+0.5m/s
            return "primary_threat_yields"
    return "primary_threat_maintains"


def select_branch(policy: dict, observed: str) -> Optional[dict]:
    """按观测到的演化条件选分支；无精确匹配则回落 default 分支。"""
    branches = policy.get("branches", [])
    branch = next((b for b in branches if b["condition_type"] == observed), None)
    if branch is None:
        branch = next((b for b in branches if b["condition_type"] == "default"), None)
    return branch


def match_policy(prev: Snapshot, now: Snapshot, policy: dict,
                 primary_threat: Optional[str]) -> Tuple[Optional[dict], str, float]:
    """一步完成分类 + 选支，返回 (分支, 观测条件, 耗时 ms) —— 反射热路径。"""
    start = time.perf_counter()
    observed = classify_evolution(prev, now, primary_threat)
    branch = select_branch(policy, observed)
    return branch, observed, (time.perf_counter() - start) * 1000
