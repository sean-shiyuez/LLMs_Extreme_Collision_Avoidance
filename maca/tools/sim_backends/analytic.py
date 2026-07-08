"""解析动力学后端（默认）—— 包装 tools/simulator。

确定性、毫秒级、可用于在线仲裁与网格扫描热路径。这是 MACA 的生产后端。
"""
from typing import Dict, Optional

from ...scenario.schema import Snapshot
from .. import simulator
from .base import SimBackend


class AnalyticBackend(SimBackend):
    """中等保真解析动力学后端：矩形 OBB 碰撞 + 摩擦圆约束 + 执行器
    一阶延迟/jerk 限制（实现见 tools/simulator.py）。"""

    name = "analytic"
    realtime_safe = True

    def simulate_maneuver(self, snapshot: Snapshot, code: int,
                          target_speed_mps: Optional[float] = None,
                          primary_id: Optional[str] = None,
                          primary_behavior: str = "maintains") -> Dict:
        return simulator.simulate_maneuver(
            snapshot, code, target_speed_mps=target_speed_mps, _with_hj=False,
            primary_id=primary_id, primary_behavior=primary_behavior)
