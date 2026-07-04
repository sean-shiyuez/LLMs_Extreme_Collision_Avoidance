"""HJ-reachability value network (ported from legacy risk_assessment.py).

Loads the same MLP weights (HJ_Reachability/safe_value_params.pth) with the
same input normalization, so risk values are bit-identical to the published
SACA implementation. This module is pure computation (~1 ms per query) and is
what keeps the reflex loop LLM-free.
"""
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn as nn

from .. import config


class _CustomMLP(nn.Module):
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
    """Raw HJ value for the ego state relative to a target position.

    Normalization is identical to the legacy code:
      x=(rel_x+50)/50, y=|rel_y|/8, phi=|phi|/40, vx=ego_vx/18.
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
    """HJ value mapped to [0, 1] risk, same affine map as legacy: (v+30)/30."""
    risk = (hj_value(ego_pos, ego_vx, target_pos, phi) + 30) / 30
    return round(max(0.0, min(1.0, risk)), 2)


def participant_type_risk(ptype: str) -> float:
    """Legacy heuristic vulnerability weight by participant type."""
    t = ptype.lower()
    if t == "pedestrian":
        return 1.0
    if t in ("small car",):
        return 0.5
    return 0.8
