"""运动学/动力学机动仿真器 —— 离线工厂的"物理真值"来源，也是在线保守
仲裁的严重度比较器。纯计算、确定性、毫秒级（在线仲裁与网格扫描都依赖
它的速度，因此不接重型平台如 CARLA；CARLA 作可选离线后端，见 sim_backends）。

中等保真解析动力学模型（相对早期"点质心 + 圆"的升级）：

1. 矩形车身 OBB 碰撞：自车按长×宽矩形（含航向）建模，与目标（按类型
   等效为圆）做有向包围盒-圆的间距检测；撞击面在车体坐标系判定，随
   航向旋转 —— 这对 T 形漂移尤为关键：漂移时车身横过来，来自侧向的
   目标会命中车头/车尾而非车侧。

2. 摩擦圆约束：常规机动(0-4)的合加速度受 |a| <= mu*g 限制（纵横耦合，
   真实车辆不能同时全力制动+全力转向）；T 形漂移(5/6)是主动失稳滑移
   操作，轮胎已饱和，单列一个更高的等效上限 DRIFT_ACCEL_CAP。

3. 执行器动力学：加速度指令不再瞬时理想执行，经一阶滞后(ACTUATOR_TAU)
   + jerk 上限(JERK_MAX)逐步建立，短 TTC 下更真实地体现"来不及"。

自车机动加速度指令剖面（世界系，随后过执行器+摩擦圆再积分）：
  code 0  AEB：ax=-8，直行。
  1/2     急变道：侧向 bang-bang（±3）到相邻车道(4 m)侧速归零；纵向匀速。
  3/4     变道+制动：侧向同上，叠加纵向 -4。
  5/6     T 形漂移（两段式）：旋转段纵向 -5 + 侧向甩尾 ±5 把航向转至垂直；
          旋转完成后进入高滑移滑行段，摩擦沿速度方向擦除动能至停。
  7       不干预：匀速。
时域 <= 2.5 s，步长 0.05 s。

周围目标同样运动：障碍物携带速度；参与者按意图演化（见 _target_pos_vel）。

严重度分级 0~4：接触相对速度定基础级，再按撞击面（尾碰 -1 / 侧碰 +1）
与是否涉及弱势道路使用者(+2)修正。
"""
import math
from typing import Dict, List, Optional, Tuple

from .. import config
from ..risk import hj_model
from ..scenario.schema import Snapshot
from .physics import effective_radius

_T_ROT = 0.9  # T 形漂移航向旋转至垂直所需时间 [s]（甩尾旋转较快）


# ---------------------------------------------------------------------------
# 自车轨迹生成：加速度指令 -> 执行器动力学 -> 摩擦圆 -> 数值积分
# ---------------------------------------------------------------------------

def _accel_command(code: int, t: float, x: float, y: float, vx: float, vy: float,
                   v0: float, target_speed: Optional[float]) -> Tuple[float, float, float]:
    """返回 t 时刻的世界系加速度指令 (ax_cmd, ay_cmd) 与目标航向角[度]。

    含变道 bang-bang 的闭环（据当前横移量切换加/减速段，到位后归零）与
    漂移的持续侧向指令。"""
    if code == 0:                                   # AEB 全力制动
        return -config.AEB_DECEL, 0.0, 0.0
    if code == 7:                                   # 不干预
        return 0.0, 0.0, 0.0
    if code in (1, 2, 3, 4):                        # 变道类
        sign = -1.0 if code in (1, 3) else 1.0
        a_lat = config.LANE_CHANGE_LAT_ACCEL
        offset = config.LANE_CHANGE_OFFSET
        cur = sign * y                              # 朝目标方向的已横移量
        # bang-bang：前半程加速、后半程反向减速，到位(offset)侧速归零
        if cur >= offset:                           # 已到相邻车道，锁定
            ay = -vy / max(config.SIM_DT, 1e-6)     # 使侧速在一步内归零
            ay = max(-a_lat, min(a_lat, ay))
        elif cur >= offset / 2.0:                   # 减速段
            ay = -sign * a_lat
        else:                                       # 加速段
            ay = sign * a_lat
        ax = -config.LC_BRAKE_DECEL if code in (3, 4) else 0.0
        if code in (3, 4) and target_speed is not None and vx <= target_speed:
            ax = 0.0                                # 已减到目标速度
        return ax, ay, 0.0
    if code in (5, 6):                              # T 形漂移（两段式）
        sign = -1.0 if code == 5 else 1.0
        heading = sign * 90.0 * min(1.0, t / _T_ROT)
        if t < _T_ROT:
            # ① 旋转段：纵向制动 + 侧向甩尾，建立大滑移角、把车尾甩向撞击点
            return -config.TDRIFT_DECEL, sign * config.TDRIFT_LAT_ACCEL, heading
        # ② 滑行段：车身已横过来（大滑移），轮胎饱和 —— 摩擦沿当前速度
        # 方向擦除动能直至停下（不再持续侧向加速，避免无界侧滑）
        sp = math.hypot(vx, vy)
        if sp > 1e-6:
            return (-config.DRIFT_ACCEL_CAP * vx / sp,
                    -config.DRIFT_ACCEL_CAP * vy / sp, heading)
        return 0.0, 0.0, heading
    return 0.0, 0.0, 0.0


def _apply_actuator(a_cur: float, a_cmd: float) -> float:
    """执行器一阶滞后 + jerk 上限：加速度指令逐步建立而非瞬时到位。"""
    dt = config.SIM_DT
    # 一阶滞后离散：向指令逼近
    a_next = a_cur + (a_cmd - a_cur) * (dt / (config.ACTUATOR_TAU + dt))
    # jerk 硬上限
    da_max = config.JERK_MAX * dt
    a_next = a_cur + max(-da_max, min(da_max, a_next - a_cur))
    return a_next


def _friction_clip(ax: float, ay: float, code: int) -> Tuple[float, float]:
    """摩擦圆约束：常规机动合加速度 <= mu*g；漂移用更高的滑移上限。
    超限时按比例缩放（保持方向）。"""
    cap = (config.DRIFT_ACCEL_CAP if code in (5, 6)
           else config.FRICTION_MU * config.GRAVITY)
    mag = math.hypot(ax, ay)
    if mag > cap and mag > 1e-9:
        s = cap / mag
        return ax * s, ay * s
    return ax, ay


def _rollout_ego(code: int, v0: float,
                 target_speed: Optional[float]) -> List[Tuple[float, float, float, float, float, float]]:
    """数值积分自车轨迹，返回逐步状态 [(t, x, y, vx, vy, heading_deg), ...]。"""
    dt = config.SIM_DT
    steps = int(config.SIM_HORIZON_S / dt)
    x = y = 0.0
    vx, vy = v0, 0.0
    ax = ay = 0.0
    traj = []
    for i in range(1, steps + 1):
        t = i * dt
        if t <= config.REACTION_DELAY:              # 反应延迟内保持原运动
            ax_cmd = ay_cmd = 0.0
            heading = 0.0
        else:
            ax_cmd, ay_cmd, heading = _accel_command(code, t - config.REACTION_DELAY,
                                                     x, y, vx, vy, v0, target_speed)
        ax = _apply_actuator(ax, ax_cmd)
        ay = _apply_actuator(ay, ay_cmd)
        ax, ay = _friction_clip(ax, ay, code)
        vx = max(0.0, vx + ax * dt)                 # 车辆不倒退
        vy = vy + ay * dt
        x += vx * dt
        y += vy * dt
        traj.append((t, x, y, vx, vy, heading))
    return traj


# ---------------------------------------------------------------------------
# 矩形车身 OBB 碰撞检测
# ---------------------------------------------------------------------------

def _obb_circle_gap(ex: float, ey: float, heading_deg: float,
                    cx: float, cy: float, r: float) -> Tuple[float, str]:
    """自车矩形 OBB（长 EGO_LENGTH、宽 EGO_WIDTH、航向 heading）与目标圆
    （中心 (cx,cy)、半径 r）的间距，及撞击面（车体系判定）。gap<0 即重叠。"""
    h = math.radians(heading_deg)
    dx, dy = cx - ex, cy - ey
    # 旋转到车体系：lx 纵向（车头 +），ly 横向（右 +）
    lx = dx * math.cos(h) + dy * math.sin(h)
    ly = -dx * math.sin(h) + dy * math.cos(h)
    hl, hw = config.EGO_LENGTH / 2.0, config.EGO_WIDTH / 2.0
    ox = max(abs(lx) - hl, 0.0)
    oy = max(abs(ly) - hw, 0.0)
    gap = math.hypot(ox, oy) - r
    # 撞击面：按接触点在车体系的归一化边距判定（随航向自然旋转）
    if abs(lx) / hl >= abs(ly) / hw:
        face = "front" if lx > 0 else "rear"
    else:
        face = "side"
    return gap, face


def _ego_lateral_extent(heading_deg: float) -> float:
    """自车 OBB 在世界横向(y)上的半跨度（漂移时车身横过来，横向占用增大）。"""
    h = math.radians(heading_deg)
    return (config.EGO_LENGTH / 2.0) * abs(math.sin(h)) + \
           (config.EGO_WIDTH / 2.0) * abs(math.cos(h))


# ---------------------------------------------------------------------------
# 目标行为模型（按意图 intention 驱动）
# ---------------------------------------------------------------------------

def _target_pos_vel(pos, vel, intention: str, t: float):
    """按意图驱动的目标行为模型：返回 t 时刻的 (位置, 速度)。

    Maintain（含运动障碍物）：恒速直线运动。
    Emergency Braking：沿运动方向以 TARGET_BRAKE_DECEL 减速至停。
    Lane Change：侧向分量在横移 TARGET_LC_OFFSET(4 m)后归零。
    """
    intent = (intention or "").lower()
    if "brak" in intent:
        speed = math.hypot(*vel)
        if speed > 1e-6:
            decel = config.TARGET_BRAKE_DECEL
            t_stop = speed / decel
            tc = min(t, t_stop)
            scale_dist = (speed * tc - 0.5 * decel * tc * tc) / speed
            new_speed_scale = max(0.0, 1.0 - decel * min(t, t_stop) / speed)
            return ((pos[0] + vel[0] * scale_dist, pos[1] + vel[1] * scale_dist),
                    (vel[0] * new_speed_scale, vel[1] * new_speed_scale))
        return pos, vel
    if "lane change" in intent and abs(vel[1]) > 1e-6:
        t_done = config.TARGET_LC_OFFSET / abs(vel[1])
        if t <= t_done:
            return (pos[0] + vel[0] * t, pos[1] + vel[1] * t), vel
        return ((pos[0] + vel[0] * t, pos[1] + vel[1] * t_done),
                (vel[0], 0.0))
    return (pos[0] + vel[0] * t, pos[1] + vel[1] * t), vel


# ---------------------------------------------------------------------------
# 单机动仿真
# ---------------------------------------------------------------------------

def simulate_maneuver(snapshot: Snapshot, code: int,
                      target_speed_mps: Optional[float] = None,
                      _with_hj: bool = True,
                      primary_id: Optional[str] = None,
                      primary_behavior: str = "maintains") -> Dict:
    """仿真单个机动，返回严重度、结局文本与证据明细。

    `primary_behavior` 允许调用方在"该分支自己的演化假设"下验证策略树
    分支：maintains（按观测演化）/ yields（主威胁以 6 m/s² 制动让行）/
    accelerates（主威胁速度放大 1.3 倍）。
    """
    if code not in config.DECISION_CODES:
        return {"error": f"unknown maneuver code {code}"}
    v0 = snapshot.ego.velocity[0]
    # 组装目标列表；对主威胁按演化假设改写其意图/速度
    targets = []
    for tid, pos, vel, kind in snapshot.targets():
        intention = next((p.intention for p in snapshot.participants
                          if p.id == tid), "")
        if tid == primary_id:
            if primary_behavior == "yields":
                intention = "Emergency Braking"
            elif primary_behavior == "accelerates":
                vel = (vel[0] * 1.3, vel[1] * 1.3)
        targets.append((tid, pos, vel, kind, intention))

    traj = _rollout_ego(code, v0, target_speed_mps)
    min_sep, min_sep_target = float("inf"), None
    collision = None
    boundary_l = snapshot.ego.road_boundary_left
    boundary_r = snapshot.ego.road_boundary_right

    for (t, ex, ey, evx, evy, heading) in traj:
        # 道路边界：自车 OBB 横向最外点越界视同撞护栏（侧碰）
        half = _ego_lateral_extent(heading)
        if ((boundary_l is not None and ey - half < boundary_l)
                or (boundary_r is not None and ey + half > boundary_r)):
            collision = {
                "target": "road_boundary", "kind": "boundary", "t": round(t, 2),
                "relative_speed_mps": round(math.hypot(evx, evy), 2),
                "impact_face": "side", "vulnerable": False,
            }
            break
        for tid, pos, vel, kind, intention in targets:
            (tx, ty), (tvx, tvy) = _target_pos_vel(pos, vel, intention, t)
            gap, face = _obb_circle_gap(ex, ey, heading, tx, ty,
                                        effective_radius(kind))
            if gap < min_sep:
                min_sep, min_sep_target = gap, tid
            if gap < 0 and collision is None:       # 首次接触即定局
                rel_speed = math.hypot(tvx - evx, tvy - evy)
                collision = {
                    "target": tid, "kind": kind, "t": round(t, 2),
                    "relative_speed_mps": round(rel_speed, 2),
                    "impact_face": face,
                    "vulnerable": "pedestrian" in kind.lower() or "cyclist" in kind.lower(),
                }
        if collision is not None:
            break

    severity, outcome = _score(code, min_sep, collision)
    end = traj[-1]
    result = {
        "code": code,
        "maneuver": config.DECISION_CODES[code],
        "severity": severity,
        "outcome": outcome,
        "min_separation_m": round(min_sep, 2),
        "closest_target": min_sep_target,
        "collision": collision,
        "end_state": {"x": round(end[1], 2), "y": round(end[2], 2),
                      "vx": round(end[3], 2), "vy": round(end[4], 2),
                      "heading_deg": round(end[5], 1)},
    }
    if _with_hj:
        result["hj"] = _hj_constraint(snapshot, code)
    return result


def _score(code: int, min_sep: float, collision: Optional[dict]):
    """严重度评分：无接触按最小间距分 0/1 级；有接触按相对速度定基础级，
    再按撞击面与是否涉及弱势道路使用者修正。"""
    if collision is None:
        if min_sep > config.SEVERITY_NEAR_MISS_SEP:
            return 0, ("No collision; minimum separation "
                       f"{min_sep:.1f} m." if min_sep < 90 else "No conflict.")
        return 1, f"Near miss: minimum separation {max(0.0, min_sep):.2f} m."
    speed = collision["relative_speed_mps"]
    if speed < config.IMPACT_MINOR_SPEED:
        sev = 2
    elif speed < config.IMPACT_MODERATE_SPEED:
        sev = 3
    else:
        sev = 4
    # 撞击面修正（领域知识）：前后吸能结构 —— 尾碰减轻一级；
    # 侧面结构弱 / EV 电池 —— 侧碰加重一级。任何接触不低于"轻微"级。
    if collision["impact_face"] == "side":
        sev = min(4, sev + config.SIDE_IMPACT_PENALTY)
    elif collision["impact_face"] == "rear":
        sev = max(2, sev - config.REAR_IMPACT_BONUS)
    if collision["vulnerable"]:
        sev = min(4, sev + config.VULNERABLE_PENALTY)
    face = collision["impact_face"]
    labels = {2: "minor damage, no injuries expected",
              3: "moderate damage, injuries possible",
              4: "severe damage, serious injuries likely"}
    return sev, (f"Collision with {collision['target']} at t={collision['t']}s, "
                 f"{face} impact at {speed:.1f} m/s relative speed"
                 + (", involving a vulnerable road user" if collision["vulnerable"] else "")
                 + f": {labels[sev]}.")


def _hj_constraint(snapshot: Snapshot, code: int) -> Dict:
    """HJ 可达性作为状态风险的"参考约束"（非主判据）：机动末态的 HJ 风险
    不得比"不干预"基线更差（超出容差即不可接受）。"""
    def end_worst(c):
        end = _rollout_ego(c, snapshot.ego.velocity[0], None)[-1]
        sim_ex, sim_ey, sim_vx = end[1], end[2], end[3]
        worst = 0.0
        for tid, pos, vel, kind in snapshot.targets():
            intention = next((p.intention for p in snapshot.participants
                              if p.id == tid), "")
            (tx, ty), _ = _target_pos_vel(pos, vel, intention, config.SIM_HORIZON_S)
            risk = hj_model.hj_risk((sim_ex, sim_ey), sim_vx, (tx, ty))
            # HJ 网络只训练过前方障碍：末态已在自车后方且（异车道或不再
            # 接近）的目标做权重折减，避免语义失真
            rel_x, rel_y = tx - sim_ex, ty - sim_ey
            behind = rel_x < 0
            if behind and (abs(rel_y) > 2.5 or vel[0] <= sim_vx):
                risk = round(risk * config.REAR_HJ_DISCOUNT, 2)
            worst = max(worst, risk)
        return worst

    worst = end_worst(code)
    baseline = worst if code == 7 else end_worst(7)
    return {"worst_end_risk": worst, "baseline_risk": baseline,
            "acceptable": worst <= baseline + config.HJ_BASELINE_TOLERANCE}


def rank_maneuvers(snapshot: Snapshot, codes=None,
                   primary_id: Optional[str] = None,
                   primary_behavior: str = "maintains") -> Dict:
    """仿真全部候选机动并排序（网格扫描的逐格点标注器）。

    排序键：严重度低者优先；同级则接触相对速度低者优先（减小伤害能量）；
    再同则取更温和的机动（AGGRESSIVENESS 小者）。
    """
    codes = list(codes) if codes is not None else list(config.DECISION_CODES)
    results = {c: simulate_maneuver(snapshot, c, _with_hj=False,
                                    primary_id=primary_id,
                                    primary_behavior=primary_behavior)
               for c in codes}

    def _key(c):
        col = results[c]["collision"]
        return (results[c]["severity"],
                col["relative_speed_mps"] if col else 0.0,
                config.AGGRESSIVENESS[c])

    ranked = sorted(codes, key=_key)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else best
    margin = results[runner]["severity"] - results[best]["severity"]
    return {"results": results, "ranked": ranked, "best": best,
            "margin": margin,   # 最优机动的严重度优势（级），规律判定用
            "vulnerable_involved": any(r.get("collision") and r["collision"]["vulnerable"]
                                       for r in results.values())}


def policy_severity(snapshot: Snapshot, policy: dict) -> Dict:
    """应急策略树的严重度画像：全分支最坏情况（对抗性演化）+ maintains
    分支严重度。在线保守仲裁用 —— 纯本地计算，毫秒级。"""
    branch_sev = {}
    for b in policy.get("branches", []):
        sim = simulate_maneuver(snapshot, int(b["code"]),
                                b.get("target_speed_mps") or None, _with_hj=False)
        branch_sev[b["condition_type"]] = sim["severity"]
    worst = max(branch_sev.values()) if branch_sev else 4
    maintains = branch_sev.get("primary_threat_maintains",
                               branch_sev.get("default", worst))
    max_aggr = max((config.AGGRESSIVENESS[int(b["code"])]
                    for b in policy.get("branches", [])), default=4)
    return {"worst": worst, "maintains": maintains, "max_aggressiveness": max_aggr,
            "per_branch": branch_sev}
