"""HJ 可达性值网络（自 legacy risk_assessment.py 移植，数值逐位一致）。

加载与已发表 SACA 实现完全相同的 MLP 权重
（HJ_Reachability/safe_value_params.pth）与输入归一化，因此风险值与
原论文实现严格一致。本模块为纯计算（单次查询 ~1 ms 量级）。

定位（按用户要求明确）：HJ 可达性本质是对状态的风险评估依据 ——
在本项目中用作 (1) 反射层风险监视信号，(2) 机动验证时的末态安全
约束参考。它不是主判据（主判据是运动学仿真严重度）。
"""
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn as nn

from .. import config


class _CustomMLP(nn.Module):
    """与 .pth 权重形状对应的三层 MLP（4 -> 256 -> 256 -> 1）。"""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu(self.fc1(x)))))


@lru_cache(maxsize=1)
def _model() -> _CustomMLP:
    """惰性单例加载权重（键名映射自 JAX 导出格式，kernel 需转置）。"""
    params = torch.load(config.CODE_DIR / "HJ_Reachability" / "safe_value_params.pth")
    model = _CustomMLP()
    model.load_state_dict({
        "fc1.weight": params["MLP_0_Dense_0_kernel"].T.clone().detach(),
        "fc1.bias": params["MLP_0_Dense_0_bias"].clone().detach(),
        "fc2.weight": params["MLP_0_Dense_1_kernel"].T.clone().detach(),
        "fc2.bias": params["MLP_0_Dense_1_bias"].clone().detach(),
        "fc3.weight": params["OutputVDense_kernel"].T.clone().detach(),
        "fc3.bias": params["OutputVDense_bias"].clone().detach(),
    })
    model.eval()
    return model


def hj_value(ego_pos: Tuple[float, float], ego_vx: float,
             target_pos: Tuple[float, float], phi: float = 0.0) -> float:
    """自车状态相对某目标位置的原始 HJ 值。

    输入归一化与 legacy 完全一致：
      x=(rel_x+50)/50, y=|rel_y|/8, phi=|phi|/40, vx=ego_vx/18。
    """
    rel_x = ego_pos[0] - target_pos[0]
    rel_y = ego_pos[1] - target_pos[1]
    vec = torch.tensor(
        [(rel_x + 50) / 50, abs(rel_y) / 8, abs(phi) / 40, ego_vx / 18],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return _model()(vec.unsqueeze(0)).item()


def hj_risk(ego_pos: Tuple[float, float], ego_vx: float,
            target_pos: Tuple[float, float], phi: float = 0.0) -> float:
    """HJ 值仿射映射到 [0,1] 风险（与 legacy 同：(v+30)/30，越大越危险）。"""
    risk = (hj_value(ego_pos, ego_vx, target_pos, phi) + 30) / 30
    return round(max(0.0, min(1.0, risk)), 2)


def participant_type_risk(ptype: str) -> float:
    """legacy 保留的按类型脆弱度权重（行人 1.0 / 小车 0.5 / 其他 0.8）。"""
    t = ptype.lower()
    if t == "pedestrian":
        return 1.0
    if t in ("small car",):
        return 0.5
    return 0.8
