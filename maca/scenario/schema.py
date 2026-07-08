"""场景数据模型（typed scenario model）。

一个 Scenario 是一条短时间线（若干 Snapshot）：至少包含一个中风险的
"预览快照"（preview，供触发时刻做演化分类的参照）和一个"触发快照"
（trigger，反射层在此执行库匹配）。

坐标约定（沿用 legacy SACA）：自车在原点，x 轴指向行驶方向，
y 轴正方向为自车的"右侧"。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Ego:
    """自车状态与道路上下文。"""
    id: str
    type: str
    coordinate: Tuple[float, float]
    velocity: Tuple[float, float]
    road_topology: str = "Normal Road"      # Normal Road / Intersection / Roundabout
    weather: str = "Sunny (good road conditions)"
    road_boundary_left: Optional[float] = None    # 左道路边界的 y 坐标（可无）
    road_boundary_right: Optional[float] = None   # 右道路边界的 y 坐标（可无）


@dataclass
class Obstacle:
    """障碍物。可以是运动的（散落货物、失控拖车等）——
    velocity 默认 (0,0) 即静态。"""
    id: str
    center: Tuple[float, float]
    ellipse_major_axis: Optional[float] = None
    ellipse_minor_axis: Optional[float] = None
    velocity: Tuple[float, float] = (0.0, 0.0)


@dataclass
class Participant:
    """交通参与者（始终是运动的）。intention 驱动仿真器的行为模型：
    Maintain / Emergency Braking / Left|Right Lane Change。"""
    id: str
    type: str
    coordinate: Tuple[float, float]
    velocity: Tuple[float, float]
    intention: str = "Maintain"

    @property
    def is_vulnerable(self) -> bool:
        """是否弱势道路使用者（行人/骑行者，决策中享有绝对保护优先级）。"""
        return self.type.lower() in ("pedestrian", "cyclist", "bicycle", "motorcycle")


@dataclass
class Snapshot:
    """某一时刻的完整场景快照。"""
    t: float
    ego: Ego
    obstacles: List[Obstacle] = field(default_factory=list)
    participants: List[Participant] = field(default_factory=list)

    def targets(self) -> List[Tuple[str, Tuple[float, float], Tuple[float, float], str]]:
        """全部碰撞相关目标的统一视图：(id, 位置, 速度, 类型)。
        障碍物携带自身速度（运动障碍物由此进入所有下游计算）。"""
        out = []
        for o in self.obstacles:
            out.append((o.id, o.center, o.velocity, "obstacle"))
        for p in self.participants:
            out.append((p.id, p.coordinate, p.velocity, p.type))
        return out

    def compact_text(self, risk_levels: Optional[Dict[str, float]] = None) -> str:
        """紧凑的文字+数值场景编码（供工厂智能体的提示词使用；
        在线平面不用文本，只用特征向量）。"""
        risk_levels = risk_levels or {}
        e = self.ego
        lines = [
            f"EGO: {e.type}, v=({e.velocity[0]:.1f},{e.velocity[1]:.1f})m/s, "
            f"road={e.road_topology}, weather={e.weather}"
        ]
        if e.road_boundary_left is not None:
            lines.append(
                f"ROAD BOUNDS: left y={e.road_boundary_left}, right y={e.road_boundary_right} "
                "(y>0 is the ego's right side)"
            )
        for o in self.obstacles:
            r = risk_levels.get(o.id)
            moving = f", v=({o.velocity[0]:.1f},{o.velocity[1]:.1f})m/s" \
                if (abs(o.velocity[0]) > 0.05 or abs(o.velocity[1]) > 0.05) else ", static"
            lines.append(
                f"OBSTACLE {o.id}: at ({o.center[0]:.1f},{o.center[1]:.1f}), "
                f"ellipse {o.ellipse_major_axis}x{o.ellipse_minor_axis}{moving}"
                + (f", HJ_risk={r:.2f}" if r is not None else "")
            )
        for p in self.participants:
            r = risk_levels.get(p.id)
            x, y = p.coordinate
            side = "right" if y > 0 else "left"
            lines.append(
                f"{p.type.upper()} {p.id}: {abs(x):.1f}m {'ahead' if x >= 0 else 'behind'}, "
                f"{abs(y):.1f}m {side}, v=({p.velocity[0]:.1f},{p.velocity[1]:.1f})m/s, "
                f"intention={p.intention}"
                + (f", risk={r:.2f}" if r is not None else "")
            )
        return "\n".join(lines)


@dataclass
class Scenario:
    """场景 = 名称 + 描述 + 快照时间线 + （可选的）人工标注结局对照表。"""
    name: str
    description: str
    snapshots: List[Snapshot]
    # 决策码（字符串）-> 结局文本；"default" 为兜底。v2 中结局以仿真器
    # 为准，此表仅保留作定性校准对照。
    execution_outcomes: Dict[str, str] = field(default_factory=dict)

    @property
    def preview(self) -> Snapshot:
        """预览快照（时间线首帧）：触发时刻做演化分类的参照。"""
        return self.snapshots[0]

    @property
    def trigger(self) -> Snapshot:
        """触发快照（时间线末帧）：反射层在此执行决策。"""
        return self.snapshots[-1]

    def outcome_for(self, code: int) -> str:
        """查人工标注结局（仅校准对照用）。"""
        return self.execution_outcomes.get(
            str(code),
            self.execution_outcomes.get("default", "Outcome not simulated for this maneuver."),
        )


def _tup(v) -> Tuple[float, float]:
    return (float(v[0]), float(v[1]))


def snapshot_to_dict(s: Snapshot) -> dict:
    """快照序列化（gap/冲突入队时保存完整场景，供工厂复现）。"""
    e = s.ego
    d = {"t": s.t,
         "ego": {"id": e.id, "type": e.type, "coordinate": list(e.coordinate),
                 "velocity": list(e.velocity), "road_topology": e.road_topology,
                 "weather": e.weather},
         "obstacles": [{"id": o.id, "center": list(o.center),
                        "ellipse_major_axis": o.ellipse_major_axis,
                        "ellipse_minor_axis": o.ellipse_minor_axis,
                        "velocity": list(o.velocity)}
                       for o in s.obstacles],
         "participants": [{"id": p.id, "type": p.type,
                           "coordinate": list(p.coordinate),
                           "velocity": list(p.velocity), "intention": p.intention}
                          for p in s.participants]}
    if e.road_boundary_left is not None:
        d["ego"]["road_boundary_left"] = e.road_boundary_left
        d["ego"]["road_boundary_right"] = e.road_boundary_right
    return d


def snapshot_from_dict(d: dict) -> Snapshot:
    """单快照反序列化（复用 scenario_from_dict 的解析逻辑）。"""
    return scenario_from_dict({"name": "_", "snapshots": [d]}).snapshots[0]


def scenario_from_dict(d: dict) -> Scenario:
    """从 JSON dict 构建 Scenario（scenario/*.json 与生成场景共用）。"""
    snapshots = []
    for s in d["snapshots"]:
        ego_d = s["ego"]
        ego = Ego(
            id=ego_d.get("id", "ego"),
            type=ego_d.get("type", "small car"),
            coordinate=_tup(ego_d.get("coordinate", (0.0, 0.0))),
            velocity=_tup(ego_d["velocity"]),
            road_topology=ego_d.get("road_topology", "Normal Road"),
            weather=ego_d.get("weather", "Sunny (good road conditions)"),
            road_boundary_left=ego_d.get("road_boundary_left"),
            road_boundary_right=ego_d.get("road_boundary_right"),
        )
        obstacles = [
            Obstacle(
                id=o["id"],
                center=_tup(o["center"]),
                ellipse_major_axis=o.get("ellipse_major_axis"),
                ellipse_minor_axis=o.get("ellipse_minor_axis"),
                velocity=_tup(o.get("velocity", (0.0, 0.0))),
            )
            for o in s.get("obstacles", [])
        ]
        participants = [
            Participant(
                id=p["id"],
                type=p["type"],
                coordinate=_tup(p["coordinate"]),
                velocity=_tup(p["velocity"]),
                intention=p.get("intention", "Maintain"),
            )
            for p in s.get("participants", [])
        ]
        snapshots.append(Snapshot(t=float(s["t"]), ego=ego, obstacles=obstacles, participants=participants))
    return Scenario(
        name=d["name"],
        description=d.get("description", ""),
        snapshots=snapshots,
        execution_outcomes=d.get("execution_outcomes", {}),
    )
