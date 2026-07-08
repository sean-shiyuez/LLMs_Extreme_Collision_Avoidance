"""确定性场景特征化 —— 双平面共享的"特征语言"。

离线工厂在这套特征上归纳规则；在线平面微秒级求值这些特征做规则匹配，
并用它们本地构造案例 k-NN 的查询向量（不调用任何 API embedding ——
这正是"案例库再大也实时"的关键）。
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..scenario.schema import Snapshot
from ..tools import physics
from .monitor import SceneAssessment

# 方位扇区（8 向，按方位角每 45° 一格；y 正方向为自车右侧）
SECTORS = ["front", "front_right", "right", "rear_right",
           "rear", "rear_left", "left", "front_left"]
PED_SIDES = ["none", "left", "right"]
TOPOLOGIES = ["Normal Road", "Intersection", "Roundabout"]
THREAT_KINDS = ["truck", "suv", "car", "pedestrian", "obstacle", "other"]

# 数值特征的顺序与归一化尺度（用于构造 k-NN 向量）
NUMERIC_SPEC = [
    ("ttc_s", 3.0),
    ("miss_m", 5.0),
    ("lateral_dist_m", 10.0),
    ("lateral_speed_mps", 10.0),
    ("closing_speed_mps", 20.0),
    ("ego_speed_mps", 25.0),
    ("hj_risk", 1.0),
    ("threat_dist_m", 30.0),
]


@dataclass
class FeatureVector:
    """一个场景的特征三元组：离散特征 + 数值特征 + 归一化 k-NN 向量。"""
    discrete: Dict[str, object]
    numeric: Dict[str, float]
    vector: List[float] = field(default_factory=list)

    def get(self, key: str):
        """统一取值接口（规则守卫求值用）：离散优先，其次数值。"""
        if key in self.discrete:
            return self.discrete[key]
        return self.numeric.get(key)


def _sector(x: float, y: float) -> str:
    """目标方位 -> 8 向扇区。方位角 0° = 正前方；y 正方向 = 自车右侧。"""
    bearing = math.degrees(math.atan2(y, x))
    idx = int(((bearing + 22.5) % 360) // 45)
    return SECTORS[idx]


def _kind_bucket(kind: str) -> str:
    """目标类型 -> 有限桶（规则守卫的离散域）。"""
    k = kind.lower()
    for key in ("truck", "suv", "pedestrian", "obstacle"):
        if key in k:
            return key
    if "car" in k:
        return "car"
    return "other"


def extract_features(snapshot: Snapshot, assessment: SceneAssessment) -> FeatureVector:
    """从场景快照 + 反射层评估提取特征向量（毫秒级，全部本地计算）。"""
    ego_speed = snapshot.ego.velocity[0]
    primary = assessment.primary_threat
    p_pos, p_vel, p_kind = (0.0, 0.0), (0.0, 0.0), "other"
    for tid, pos, vel, kind in snapshot.targets():
        if tid == primary:
            p_pos, p_vel, p_kind = pos, vel, kind
            break
    per = assessment.per_target.get(primary, {}) if primary else {}

    # 行人所在侧（决定 T 形漂移/变道的禁行方向）
    ped_side = "none"
    for p in snapshot.participants:
        if p.is_vulnerable:
            ped_side = "left" if p.coordinate[1] < 0 else "right"
            break

    # 仿真接地的"制动是否足够"：直接推演 code 0，severity <= 1（险情以内）
    # 才算制动可解 —— 避免"斜穿目标不在本车道"造成的启发式空真误判。
    from ..tools import simulator
    brake_sim = simulator.simulate_maneuver(snapshot, 0, _with_hj=False,
                                            primary_id=primary)
    corridor_l = physics.lateral_clearance(snapshot, "left")
    corridor_r = physics.lateral_clearance(snapshot, "right")

    rvx, rvy = p_vel[0] - ego_speed, p_vel[1]
    discrete = {
        "road_topology": snapshot.ego.road_topology,
        "threat_kind": _kind_bucket(p_kind),
        "approach_sector": _sector(p_pos[0], p_pos[1]) if primary else "front",
        "pedestrian_side": ped_side,
        "corridor_left_free": bool(corridor_l["lane_change_possible"]),
        "corridor_right_free": bool(corridor_r["lane_change_possible"]),
        "braking_sufficient": bool(brake_sim["severity"] <= 1),
        "on_collision_course": bool(per.get("on_collision_course", False)),
    }
    numeric = {
        # 99.0 为"无有效值"的哨兵（如目标不在碰撞路径上则无 TTC）
        "ttc_s": per.get("ttc_s") if per.get("ttc_s") is not None else 99.0,
        "miss_m": per.get("miss_m") if per.get("miss_m") is not None else 99.0,
        "lateral_dist_m": abs(p_pos[1]),
        "lateral_speed_mps": abs(p_vel[1]),
        "closing_speed_mps": math.hypot(rvx, rvy),
        "ego_speed_mps": ego_speed,
        "hj_risk": per.get("hj_risk", 0.0),
        "threat_dist_m": math.hypot(p_pos[0], p_pos[1]) if primary else 99.0,
        "scene_risk": assessment.scene_risk,
    }
    return FeatureVector(discrete=discrete, numeric=numeric,
                         vector=_knn_vector(discrete, numeric))


def _knn_vector(discrete: Dict, numeric: Dict) -> List[float]:
    """构造 k-NN 查询向量：归一化数值段 + 离散 one-hot 段（约 29 维）。
    全程本地计算 —— 这是案例匹配不依赖网络的根本原因。"""
    vec = [min(1.0, (numeric.get(name) or 0.0) / scale) for name, scale in NUMERIC_SPEC]
    vec += [1.0 if discrete["approach_sector"] == s else 0.0 for s in SECTORS]
    vec += [1.0 if discrete["pedestrian_side"] == s else 0.0 for s in PED_SIDES]
    vec += [1.0 if discrete["road_topology"] == t else 0.0 for t in TOPOLOGIES]
    vec += [1.0 if discrete["threat_kind"] == k else 0.0 for k in THREAT_KINDS]
    vec += [1.0 if discrete["corridor_left_free"] else 0.0,
            1.0 if discrete["corridor_right_free"] else 0.0,
            1.0 if discrete["braking_sufficient"] else 0.0,
            1.0 if discrete["on_collision_course"] else 0.0]
    return vec


def features_to_dict(fv: FeatureVector) -> dict:
    """序列化（入库/入队用）。"""
    return {"discrete": fv.discrete, "numeric": fv.numeric, "vector": fv.vector}


def features_from_dict(d: dict) -> FeatureVector:
    """反序列化；缺 vector 时按当前编码规则重建。"""
    return FeatureVector(discrete=d["discrete"], numeric=d["numeric"],
                         vector=d.get("vector") or _knn_vector(d["discrete"], d["numeric"]))
