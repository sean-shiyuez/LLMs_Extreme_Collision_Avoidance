"""Typed scenario model.

A Scenario is a short timeline of Snapshots: at least one moderate-risk
"preview" snapshot (where deliberation is woken) and one "trigger" snapshot
(where the reflex loop executes from the armed policy cache). Coordinates use
the legacy SACA convention: x forward, y positive to the RIGHT of the ego.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Ego:
    id: str
    type: str
    coordinate: Tuple[float, float]
    velocity: Tuple[float, float]
    road_topology: str = "Normal Road"
    weather: str = "Sunny (good road conditions)"
    road_boundary_left: Optional[float] = None
    road_boundary_right: Optional[float] = None


@dataclass
class Obstacle:
    id: str
    center: Tuple[float, float]
    ellipse_major_axis: Optional[float] = None
    ellipse_minor_axis: Optional[float] = None


@dataclass
class Participant:
    id: str
    type: str
    coordinate: Tuple[float, float]
    velocity: Tuple[float, float]
    intention: str = "Maintain"

    @property
    def is_vulnerable(self) -> bool:
        return self.type.lower() in ("pedestrian", "cyclist", "bicycle", "motorcycle")


@dataclass
class Snapshot:
    t: float
    ego: Ego
    obstacles: List[Obstacle] = field(default_factory=list)
    participants: List[Participant] = field(default_factory=list)

    def targets(self) -> List[Tuple[str, Tuple[float, float], Tuple[float, float], str]]:
        """All collision-relevant objects as (id, position, velocity, kind)."""
        out = []
        for o in self.obstacles:
            out.append((o.id, o.center, (0.0, 0.0), "obstacle"))
        for p in self.participants:
            out.append((p.id, p.coordinate, p.velocity, p.type))
        return out

    def compact_text(self, risk_levels: Optional[Dict[str, float]] = None) -> str:
        """Token-lean structured encoding for prompts (completion latency matters)."""
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
            lines.append(
                f"OBSTACLE {o.id}: at ({o.center[0]:.1f},{o.center[1]:.1f}), "
                f"ellipse {o.ellipse_major_axis}x{o.ellipse_minor_axis}"
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
    name: str
    description: str
    snapshots: List[Snapshot]
    # decision code (as str) -> outcome text; "default" is the fallback
    execution_outcomes: Dict[str, str] = field(default_factory=dict)

    @property
    def preview(self) -> Snapshot:
        return self.snapshots[0]

    @property
    def trigger(self) -> Snapshot:
        return self.snapshots[-1]

    def outcome_for(self, code: int) -> str:
        return self.execution_outcomes.get(
            str(code),
            self.execution_outcomes.get("default", "Outcome not simulated for this maneuver."),
        )


def _tup(v) -> Tuple[float, float]:
    return (float(v[0]), float(v[1]))


def scenario_from_dict(d: dict) -> Scenario:
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
