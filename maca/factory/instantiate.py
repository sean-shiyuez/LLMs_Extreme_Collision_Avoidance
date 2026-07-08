"""参数化场景族的确定性实例化。

生成智能体只负责"设计实验"（模式 + 参数范围 + 上下文模板）；本模块把
每个网格参数点变成具体的双快照 Scenario。LLM 从不直接产出原始坐标 ——
物理合理性由构造保证（如横穿目标按构造置于碰撞路径上）。

四种场景模式（pattern）：
  lateral_crossing  目标从一侧横穿自车路径、按构造处于碰撞路径
                    （参数：ego_speed, ttc, lateral_speed）
  lead_braking      前车急刹（参数：ego_speed, gap, speed_diff；
                    可选后方跟车 follower）
  cut_in            邻道车辆切入自车车道
                    （参数：ego_speed, gap, speed_diff, lateral_offset）
  static_blockage   静态障碍堵塞自车车道（参数：ego_speed, gap）
"""
import itertools
from typing import Dict, Iterator, List, Tuple

from .. import config
from ..scenario.schema import Scenario, scenario_from_dict

PATTERNS = ["lateral_crossing", "lead_braking", "cut_in", "static_blockage"]

_THREAT_TYPES = {"lateral_crossing": "Large Truck", "lead_braking": "SUV",
                 "cut_in": "SUV", "static_blockage": "obstacle"}


def grid_points(family: dict) -> Iterator[Dict[str, float]]:
    """场景族参数范围的笛卡尔网格（点数受 SWEEP_MAX_POINTS 封顶）。"""
    axes: List[Tuple[str, List[float]]] = []
    for p in family["parameters"]:
        steps = max(2, int(p.get("steps", 3)))
        lo, hi = float(p["min"]), float(p["max"])
        axes.append((p["name"], [round(lo + (hi - lo) * i / (steps - 1), 3)
                                 for i in range(steps)]))
    count = 0
    for combo in itertools.product(*[vals for _, vals in axes]):
        if count >= config.SWEEP_MAX_POINTS:
            return
        count += 1
        yield dict(zip([n for n, _ in axes], combo))


def instantiate(family: dict, params: Dict[str, float]) -> Scenario:
    pattern = family["pattern"]
    template = family.get("template", {})
    builder = {
        "lateral_crossing": _lateral_crossing,
        "lead_braking": _lead_braking,
        "cut_in": _cut_in,
        "static_blockage": _static_blockage,
    }[pattern]
    trigger = builder(template, params)
    _apply_context(trigger, template)
    preview = _extrapolate_back(trigger, 1.0)
    name = family.get("family_id", pattern) + "@" + \
        ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return scenario_from_dict({
        "name": name,
        "description": family.get("description", ""),
        "snapshots": [preview, trigger],
    })


def _ego(template: dict, speed: float) -> dict:
    ego = {"id": "ego", "type": "small car", "coordinate": [0.0, 0.0],
           "velocity": [round(speed, 2), 0.0],
           "road_topology": template.get("road_topology", "Normal Road")}
    if template.get("road_boundary_left") is not None:
        ego["road_boundary_left"] = template["road_boundary_left"]
        ego["road_boundary_right"] = template.get("road_boundary_right", 10)
    return ego


def _side_sign(template: dict) -> float:
    return -1.0 if template.get("from_side", "left") == "left" else 1.0


def _pedestrian(template: dict, ego_speed: float, ttc: float) -> List[dict]:
    ped = template.get("pedestrian")
    if not ped or not ped.get("present"):
        return []
    side = -1.0 if ped.get("side", "left") == "left" else 1.0
    return [{"id": "pedestrian", "type": "Pedestrian",
             "coordinate": [round(max(4.0, ego_speed * ttc * 0.5), 2), side * 4.0],
             "velocity": [0.0, 0.0], "intention": "Maintain"}]


def _lateral_crossing(template: dict, p: Dict[str, float]) -> dict:
    ego_speed, ttc, lat_speed = p["ego_speed"], p["ttc"], p["lateral_speed"]
    sign = _side_sign(template)
    threat = {
        "id": "threat", "type": template.get("threat_type", _THREAT_TYPES["lateral_crossing"]),
        # collision course by construction: both reach the conflict point at ~ttc
        "coordinate": [round(ego_speed * ttc, 2), round(sign * lat_speed * ttc, 2)],
        "velocity": [0.0, round(-sign * lat_speed, 2)],
        "intention": "Maintain",
    }
    return {"t": 0.0, "ego": _ego(template, ego_speed), "obstacles": [],
            "participants": [threat] + _pedestrian(template, ego_speed, ttc)}


def _lead_braking(template: dict, p: Dict[str, float]) -> dict:
    ego_speed, gap = p["ego_speed"], p["gap"]
    dv = p.get("speed_diff", 2.0)
    parts = [{"id": "lead", "type": template.get("threat_type", "SUV"),
              "coordinate": [round(gap, 2), 0.0],
              "velocity": [round(max(0.0, ego_speed - dv), 2), 0.0],
              "intention": "Emergency Braking"}]
    if template.get("follower"):
        parts.append({"id": "follower", "type": "SUV",
                      "coordinate": [-6.0, 0.0],
                      "velocity": [round(ego_speed - 1.0, 2), 0.0],
                      "intention": "Maintain"})
    return {"t": 0.0, "ego": _ego(template, ego_speed), "obstacles": [],
            "participants": parts + _pedestrian(template, ego_speed, 1.5)}


def _cut_in(template: dict, p: Dict[str, float]) -> dict:
    ego_speed, gap = p["ego_speed"], p["gap"]
    dv = p.get("speed_diff", 0.0)
    lat = p.get("lateral_offset", 4.0)
    sign = _side_sign(template)
    parts = [{"id": "cutter", "type": template.get("threat_type", "SUV"),
              "coordinate": [round(gap, 2), round(sign * lat, 2)],
              "velocity": [round(ego_speed - dv, 2), round(-sign * 2.0, 2)],
              "intention": ("Right" if sign < 0 else "Left") + " Lane Change"}]
    return {"t": 0.0, "ego": _ego(template, ego_speed), "obstacles": [],
            "participants": parts + _pedestrian(template, ego_speed, 1.5)}


def _static_blockage(template: dict, p: Dict[str, float]) -> dict:
    ego_speed, gap = p["ego_speed"], p["gap"]
    obstacles = [{"id": "blockage", "center": [round(gap, 2), 0.0],
                  "ellipse_major_axis": 4, "ellipse_minor_axis": 2}]
    return {"t": 0.0, "ego": _ego(template, ego_speed), "obstacles": obstacles,
            "participants": _pedestrian(template, ego_speed, gap / max(ego_speed, 1.0))}


def _apply_context(snapshot_dict: dict, template: dict):
    if template.get("blocked_side"):
        # occupy one corridor with a neighboring vehicle to constrain escapes
        sign = -1.0 if template["blocked_side"] == "left" else 1.0
        snapshot_dict["participants"].append(
            {"id": "neighbor", "type": "SUV",
             "coordinate": [2.0, sign * 3.5],
             "velocity": [snapshot_dict["ego"]["velocity"][0], 0.0],
             "intention": "Maintain"})


def _extrapolate_back(trigger: dict, dt: float) -> dict:
    """按相对速度回推 dt 秒得到预览快照（触发时刻演化分类的参照帧）。"""
    ego_v = trigger["ego"]["velocity"]
    prev = {"t": trigger["t"] - dt, "ego": dict(trigger["ego"]),
            "obstacles": [], "participants": []}
    for o in trigger["obstacles"]:
        ov = o.get("velocity", [0.0, 0.0])
        prev["obstacles"].append(dict(o, center=[
            round(o["center"][0] + (ego_v[0] - ov[0]) * dt, 2),
            round(o["center"][1] + (ego_v[1] - ov[1]) * dt, 2)]))
    for p in trigger["participants"]:
        prev["participants"].append(dict(p, coordinate=[
            round(p["coordinate"][0] + (ego_v[0] - p["velocity"][0]) * dt, 2),
            round(p["coordinate"][1] + (ego_v[1] - p["velocity"][1]) * dt, 2)]))
    return prev
