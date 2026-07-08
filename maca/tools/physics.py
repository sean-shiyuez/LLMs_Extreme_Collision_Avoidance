"""基础几何/物理工具（确定性计算，无 LLM）。

只保留双平面实际使用的两个函数：
  - effective_radius : 按目标类型给出等效碰撞圆半径（车辆按圆处理）
  - lateral_clearance: 判断自车某一侧的走廊是否可用于变道

坐标约定（全项目一致）：自车在原点，x 轴指向行驶方向，y 轴正方向为
自车的"右侧"。
"""
from typing import Dict

from ..scenario.schema import Snapshot

# 各类目标的等效碰撞圆半径 [m]（按类型关键词模糊匹配；
# 点质心模型的车辆外形补偿）
_RADII = {"truck": 3.5, "suv": 2.8, "car": 2.5, "pedestrian": 1.0, "obstacle": 2.5}


def effective_radius(kind: str) -> float:
    """按目标类型关键词返回等效碰撞半径；未识别类型按普通轿车处理。"""
    k = kind.lower()
    for key, r in _RADII.items():
        if key in k:
            return r
    return 2.5


def lateral_clearance(snapshot: Snapshot, side: str) -> Dict:
    """自车某一侧走廊（|y| 在 1~5 m、x 在 -5~25 m 的矩形区域）的占用情况。

    返回：该侧占用者列表（含是否弱势道路使用者）、道路边界余量，以及
    综合判断的 lane_change_possible（无占用且边界余量 >= 3 m）。
    """
    sign = -1.0 if side == "left" else 1.0
    occupants = []
    for tid, pos, vel, kind in snapshot.targets():
        if -5.0 <= pos[0] <= 25.0 and 1.0 <= sign * pos[1] <= 5.0:
            occupants.append({"id": tid, "kind": kind,
                              "position": [round(pos[0], 1), round(pos[1], 1)],
                              "vulnerable": "pedestrian" in kind.lower()})
    ego = snapshot.ego
    boundary = ego.road_boundary_left if side == "left" else ego.road_boundary_right
    boundary_margin = None if boundary is None else round(abs(boundary) - 1.0, 2)
    lane_change_possible = (
        not occupants and (boundary_margin is None or boundary_margin >= 3.0)
    )
    return {"side": side, "occupants": occupants,
            "boundary_margin_m": boundary_margin,
            "lane_change_possible": lane_change_possible}
