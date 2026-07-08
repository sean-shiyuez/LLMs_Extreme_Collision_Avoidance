"""CARLA 高保真离线后端（可选，默认不安装）。

=============================== 定位 ===============================
CARLA 是重量级的自动驾驶物理仿真平台（GB 级、建议 GPU，单次机动仿真
秒级）。它**不能**用于 MACA 的在线保守仲裁或离线网格扫描热路径 ——
那些路径依赖毫秒级确定性仿真（由 AnalyticBackend 提供）。

CARLA 后端的用途是**离线高保真交叉验证**：对解析后端标注的关键格点
（如规律边界、关键案例）抽样，在 CARLA 里重放同一机动，确认解析模型
的严重度判定与高保真物理一致。这属于研究/验证流程，不进生产决策路径。

=============================== 接入步骤 ===========================
1. 安装 CARLA（0.9.14+）与其 Python API：
     https://carla.readthedocs.io/en/latest/start_quickstart/
     pip install carla
2. 启动 CARLA 服务端（无头模式）：
     ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
3. 实现下方 `_rollout_in_carla`：
     - 连接 client = carla.Client("localhost", 2000)
     - 生成 ego 车辆与目标（按 snapshot 的位置/速度）
     - 每步施加机动对应的控制（油门/刹车/转向），或用 CARLA 的
       VehicleControl 逼近 tools/simulator 里的加速度剖面
     - 检测碰撞（carla.CollisionSensor）、记录接触相对速度与撞击面
     - 映射到 MACA 的 severity 分级（复用 simulator._score 的口径）
4. 用法：
     from maca.tools.sim_backends import get_backend
     carla = get_backend("carla", host="localhost", port=2000)
     ref = carla.simulate_maneuver(snapshot, code)

坐标对齐：MACA 用 x 前向、y 右向、自车在原点的局部系；CARLA 用左手
世界系。接入时需做一次刚体变换（以 ego 初始位姿为原点对齐）。
"""
from typing import Dict, Optional

from ...scenario.schema import Snapshot
from .base import SimBackend


class CarlaBackend(SimBackend):
    """CARLA 后端桩。实例化不报错（便于 import），实际调用时若 carla
    未安装或未实现则给出明确指引。"""

    name = "carla"
    realtime_safe = False   # 秒级，绝不进热路径

    def __init__(self, host: str = "localhost", port: int = 2000,
                 timeout: float = 10.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._client = None

    def _connect(self):
        try:
            import carla  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "CARLA Python API 未安装。CARLA 后端是可选的离线高保真校验"
                "后端；生产决策请用默认的 analytic 后端。接入步骤见本文件"
                "顶部 docstring。") from e
        import carla
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(self.timeout)
        return self._client

    def simulate_maneuver(self, snapshot: Snapshot, code: int,
                          target_speed_mps: Optional[float] = None,
                          primary_id: Optional[str] = None,
                          primary_behavior: str = "maintains") -> Dict:
        self._connect()   # 若未安装 carla，在此给出明确指引
        raise NotImplementedError(
            "CarlaBackend._rollout_in_carla 尚未实现 —— 这是留给接入 CARLA "
            "的研究者的钩子。参见本文件顶部的接入步骤。默认 analytic 后端"
            "已能满足在线决策与离线规律发现的全部需求。")
