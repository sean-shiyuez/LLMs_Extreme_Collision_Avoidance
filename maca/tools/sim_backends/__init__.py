"""仿真后端抽象层。

MACA 的默认后端是 AnalyticBackend —— 中等保真解析动力学
（tools/simulator.py），确定性、毫秒级，因此可用于在线保守仲裁与离线
网格扫描的热路径。

CarlaBackend 是可选的高保真离线后端桩：它把同一套机动码在 CARLA
物理引擎里跑出结局，用于对解析后端做离线交叉验证 / 最终结局确认。
CARLA 太重（GB 级 + GPU），单次仿真秒级，**绝不**用于在线或网格扫描
热路径 —— 仅作离线抽样校验。默认不安装，见 carla_backend.py 的接入文档。

用法：
    from maca.tools.sim_backends import get_backend
    backend = get_backend("analytic")            # 默认
    result = backend.simulate_maneuver(snapshot, code)
"""
from .base import SimBackend
from .analytic import AnalyticBackend

_BACKENDS = {"analytic": AnalyticBackend}


def get_backend(name: str = "analytic", **kwargs) -> SimBackend:
    """按名称取仿真后端实例。CARLA 后端惰性导入（避免无 carla 时报错）。"""
    if name == "carla":
        from .carla_backend import CarlaBackend
        return CarlaBackend(**kwargs)
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend '{name}'; available: "
                         f"{list(_BACKENDS) + ['carla']}")
    return _BACKENDS[name](**kwargs)
