"""反射层风险监视器 —— HJ 可达性（状态风险参考）+ TTC 紧迫度。

处于毫秒级关键路径上的纯计算模块（无 LLM）。每个目标的组合风险 =
0.6 × HJ 状态风险 + 0.4 × TTC 紧迫度；全场景取最大值后按
RISK_LOW / RISK_TRIGGER 分为 low / elevated / critical 三带。
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional

from .. import config
from ..risk import hj_model
from ..scenario.schema import Snapshot
from ..tools.physics import effective_radius

W_HJ, W_URGENCY = 0.6, 0.4   # HJ 风险与 TTC 紧迫度的组合权重


@dataclass
class SceneAssessment:
    """一次反射层评估的结果。"""
    scene_risk: float                 # 场景组合风险（各目标最大值）
    min_ttc: Optional[float]          # 最小 TTC；None = 无目标在碰撞路径上
    primary_threat: Optional[str]     # 主威胁目标 id
    per_target: Dict[str, dict]       # 各目标的明细（hj/ttc/miss/urgency/…）

    def band(self) -> str:
        """风险分带：low（放行）/ elevated（监视）/ critical（触发执行）。"""
        if self.scene_risk < config.RISK_LOW:
            return "low"
        if self.scene_risk < config.RISK_TRIGGER:
            return "elevated"
        return "critical"


def assess_scene(snapshot: Snapshot) -> SceneAssessment:
    """对单个场景快照做反射层评估（毫秒级）。

    每个目标：
      - HJ 风险：值网络给出的状态风险（后方且不在碰撞路径的目标做折减，
        因为 HJ 网络只训练过前方障碍）；
      - TTC 紧迫度：最近接近点时刻（tca）线性映射到 [0,1]，仅对
        "在碰撞路径上"（预计最近距离小于等效碰撞半径）的目标计算。
    """
    ex, ey = snapshot.ego.velocity
    per_target: Dict[str, dict] = {}
    scene_risk, min_ttc = 0.0, None

    for tid, pos, vel, kind in snapshot.targets():
        hj = hj_model.hj_risk((0.0, 0.0), ex, pos)
        px, py = pos
        rvx, rvy = vel[0] - ex, vel[1] - ey   # 相对速度（目标相对自车）
        ttc, miss, on_course = None, None, False
        closing = px * rvx + py * rvy          # <0 表示正在接近
        v2 = rvx * rvx + rvy * rvy
        if closing < 0 and v2 > 1e-9:
            tca = -closing / v2                                    # 最近接近时刻
            miss = math.hypot(px + rvx * tca, py + rvy * tca)      # 最近距离
            on_course = miss <= effective_radius(kind) + 1.0       # 是否碰撞路径
            if on_course:
                ttc = tca
        urgency = 0.0
        if ttc is not None:
            urgency = max(0.0, min(1.0, 1.0 - ttc / config.TTC_URGENCY_HORIZON))
        # HJ 网络只训练过前方障碍：后方且不在碰撞路径的目标不可能是
        # 决定性威胁，其 HJ 权重折减
        hj_weight = W_HJ * (config.REAR_HJ_DISCOUNT if (px < 0 and not on_course) else 1.0)
        combined = round(hj_weight * hj + W_URGENCY * urgency, 3)
        per_target[tid] = {"hj_risk": hj, "ttc_s": None if ttc is None else round(ttc, 2),
                           "miss_m": None if miss is None else round(miss, 2),
                           "on_collision_course": on_course,
                           "urgency": round(urgency, 3), "combined": combined}
        if combined > scene_risk:
            scene_risk = combined
        if ttc is not None and (min_ttc is None or ttc < min_ttc):
            min_ttc = ttc

    # 主威胁选择：真正处于碰撞路径上的目标优先于仅仅"离得近"的旁观者
    # （并行同速的邻车 HJ 近距分很高，但它不是决定性冲突对象）。
    def _priority(item):
        tid, v = item
        return (1 if v["on_collision_course"] else 0, v["combined"])

    primary = max(per_target.items(), key=_priority)[0] if per_target else None
    return SceneAssessment(scene_risk=round(scene_risk, 3), min_ttc=min_ttc,
                           primary_threat=primary, per_target=per_target)
