"""仿真后端抽象基类。

统一接口，使解析后端与 CARLA 后端可互换。签名对齐
tools/simulator.simulate_maneuver，返回同构的结局字典（至少含
severity / outcome / collision / min_separation_m），从而工厂与在线
仲裁无需知道底层用的是哪个后端。
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ...scenario.schema import Snapshot


class SimBackend(ABC):
    """机动仿真后端统一接口。"""

    #: 后端名称
    name = "base"
    #: 是否可用于实时/热路径（解析后端 True；CARLA False）
    realtime_safe = False

    @abstractmethod
    def simulate_maneuver(self, snapshot: Snapshot, code: int,
                          target_speed_mps: Optional[float] = None,
                          primary_id: Optional[str] = None,
                          primary_behavior: str = "maintains") -> Dict:
        """仿真单个机动，返回结局字典。

        必含字段：
          - code, severity(0-4), outcome(str)
          - min_separation_m(float), collision(dict|None)
          - end_state(dict)
        """
        raise NotImplementedError
