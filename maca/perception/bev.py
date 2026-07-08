"""Bird's-eye-view renderer.

Renders a Snapshot into a BEV image for the vision-language PerceptionAgent.
Convention: x forward (right of the plot), y positive to the ego's RIGHT
(plotted downward so that "left of the car" appears above it, matching a
driver's mental image when the map is drawn heading-up... rotated).
"""
import base64
import io
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrow, Rectangle

from ..scenario.schema import Snapshot

_VEHICLE_SIZES = {  # (length, width) in meters by loose type keyword
    "truck": (10.0, 2.6),
    "suv": (4.8, 1.9),
    "car": (4.5, 1.8),
}


def _size_for(ptype: str):
    t = ptype.lower()
    for key, size in _VEHICLE_SIZES.items():
        if key in t:
            return size
    return (4.5, 1.8)


def _draw_arrow(ax, x, y, vx, vy, color):
    if abs(vx) < 0.05 and abs(vy) < 0.05:
        return
    ax.add_patch(FancyArrow(x, y, vx * 0.5, vy * 0.5, width=0.15,
                            head_width=0.7, color=color, alpha=0.9, zorder=5))


def render_bev(snapshot: Snapshot, save_path: Optional[str] = None) -> str:
    """Render the snapshot; returns a base64-encoded PNG (optionally saved)."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    ego = snapshot.ego

    # Road boundaries (y is lateral; drawn as horizontal lines across the plot)
    if ego.road_boundary_left is not None:
        for yb, label in ((ego.road_boundary_left, "left boundary"),
                          (ego.road_boundary_right, "right boundary")):
            ax.axhline(yb, color="dimgray", lw=2)
            ax.text(-8, yb, label, fontsize=7, color="dimgray", va="bottom")

    # Ego vehicle at origin
    el, ew = _size_for(ego.type)
    ax.add_patch(Rectangle((-el / 2, -ew / 2), el, ew, color="tab:blue", zorder=4))
    ax.text(0, -ew / 2 - 0.8, f"EGO {ego.velocity[0]:.0f} m/s", color="tab:blue",
            fontsize=8, ha="center", weight="bold")
    _draw_arrow(ax, el / 2, 0, ego.velocity[0], ego.velocity[1], "tab:blue")

    for o in snapshot.obstacles:
        major = o.ellipse_major_axis if isinstance(o.ellipse_major_axis, (int, float)) else 3.0
        minor = o.ellipse_minor_axis if isinstance(o.ellipse_minor_axis, (int, float)) else 3.0
        ax.add_patch(Ellipse(o.center, 2 * major, 2 * minor, facecolor="tab:gray",
                             edgecolor="black", alpha=0.7, zorder=3))
        ax.text(o.center[0], o.center[1], o.id, fontsize=7, ha="center", zorder=6)

    for p in snapshot.participants:
        x, y = p.coordinate
        if p.is_vulnerable:
            ax.plot(x, y, "o", color="tab:red", markersize=9, zorder=4)
            ax.text(x, y - 1.2, p.id, fontsize=8, color="tab:red", ha="center")
        else:
            pl, pw = _size_for(p.type)
            ax.add_patch(Rectangle((x - pl / 2, y - pw / 2), pl, pw,
                                   color="tab:orange", zorder=4))
            ax.text(x, y - pw / 2 - 0.8, f"{p.id} ({p.intention})",
                    fontsize=7, color="tab:orange", ha="center")
        _draw_arrow(ax, x, y, p.velocity[0], p.velocity[1], "tab:red" if p.is_vulnerable else "tab:orange")

    xs = [t[1][0] for t in snapshot.targets()] + [0]
    ys = [t[1][1] for t in snapshot.targets()] + [0]
    ax.set_xlim(min(xs) - 12, max(xs) + 12)
    ax.set_ylim(min(ys) - 6, max(ys) + 6)
    ax.invert_yaxis()  # +y (ego's right) points down in the image
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, ls=":")
    ax.set_xlabel("longitudinal x [m] (driving direction ->)")
    ax.set_ylabel("lateral y [m] (up = ego's LEFT)")
    ax.set_title(f"BEV @ t={snapshot.t:+.1f}s | {ego.road_topology}, {ego.weather}",
                 fontsize=9)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = buf.getvalue()
    if save_path:
        with open(save_path, "wb") as f:
            f.write(data)
    return base64.b64encode(data).decode("ascii")
