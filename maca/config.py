"""MACA 全局配置 —— 经验编译式双平面架构（Experience-Compiled Two-Plane）。

在线平面（maca/runtime）：规则库/案例库匹配，无 LLM、无网络。
离线平面（maca/factory）：多智能体规律发现工厂（LLM 只在这里运行）。

API key 存放在 Code/.env（参见 .env.example），仅离线工厂需要。
严禁在源码中硬编码任何密钥。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent

load_dotenv(CODE_DIR / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# 工厂模型注册表：角色 -> (模型, 思考强度 reasoning_effort)。
# 离线不吝啬思考：场景生成与规律归纳用高/中思考；委员会角色轻量运行。
# 在线平面完全不使用 LLM。
# 可按角色用环境变量覆写：MACA_MODEL_<ROLE> / MACA_EFFORT_<ROLE>。
# 模型名已对照 2026-07 的 OpenAI API 现役型号核验。
# ---------------------------------------------------------------------------
FACTORY_ROLES = {
    "generator": ("gpt-5.5", "high"),        # 场景族设计（实验设计者）
    "consolidator": ("gpt-5.5", "medium"),   # 从扫描网格归纳规律
    "decision": ("gpt-5.4-mini", "none"),    # 委员会：策略树合成
    "advocate": ("gpt-5.4-mini", "low"),     # 辩论正反方（"少步数推理"）
    "arbiter": ("gpt-5.4-mini", "low"),      # 辩论仲裁者
    "safety": ("gpt-5.4-mini", "low"),       # 安全校验层的 LLM 复核（可选）
    "evaluation": ("gpt-5.4-nano", "none"),  # 反思 / 经验教训撰写
}


def model_for(role: str) -> str:
    """按角色取模型名（环境变量优先）。"""
    return os.environ.get(f"MACA_MODEL_{role.upper()}", FACTORY_ROLES[role][0])


def effort_for(role: str) -> str:
    """按角色取思考强度（环境变量优先）。"""
    return os.environ.get(f"MACA_EFFORT_{role.upper()}", FACTORY_ROLES[role][1])


# ---------------------------------------------------------------------------
# 库文件（两个平面共享的"编译产物"）
# ---------------------------------------------------------------------------
LIBRARY_DIR = PACKAGE_DIR / "library"
RULES_PATH = LIBRARY_DIR / "rules.jsonl"     # 规则库（规律）
CASES_PATH = LIBRARY_DIR / "cases.jsonl"     # 案例库（模糊区域）
QUEUE_PATH = LIBRARY_DIR / "queue.jsonl"     # 运行时反馈队列（gap + 冲突）

RUNS_DIR = CODE_DIR / "Results" / "runs"                 # 在线运行日志
FACTORY_REPORTS_DIR = CODE_DIR / "Results" / "factory"   # 工厂批次报告
# 库成长日志：案例生成、判断、规则完善的全部事件按时间追加到这一个文件
GROWTH_LOG_PATH = FACTORY_REPORTS_DIR / "library_growth.log"
GENERATED_SCENARIOS_DIR = PACKAGE_DIR / "scenario" / "scenarios" / "generated"
DISTILL_EXPORT_PATH = CODE_DIR / "fine-tuning" / "maca_distill.jsonl"  # 蒸馏样本出口

# ---------------------------------------------------------------------------
# 在线匹配阈值
# ---------------------------------------------------------------------------
CASE_SIM_THRESHOLD = 0.92    # 案例 k-NN 命中的余弦相似度门槛
CASE_KNN_K = 3               # k-NN 返回的近邻数

# 反射层风险分带（HJ 状态风险 + TTC 紧迫度的加权组合；经四个基准场景标定）
RISK_LOW = 0.40              # 低于此值：不干预（code 7）
RISK_TRIGGER = 0.63          # 达到此值：进入 critical 带，执行库匹配
TTC_URGENCY_HORIZON = 3.0    # TTC（秒）线性映射为紧迫度 [0..1] 的时间尺度
REAR_HJ_DISCOUNT = 0.3       # 后方且不在碰撞路径目标的 HJ 权重折减
                             # （HJ 网络只训练过前方障碍，对后方目标语义失真）

# ---------------------------------------------------------------------------
# 运动学/动力学仿真器（中等保真解析动力学：矩形车身 OBB 碰撞 + 摩擦圆
# 约束 + 执行器一阶延迟/jerk 限制；时域 <=2.5 s）
# ---------------------------------------------------------------------------
SIM_HORIZON_S = 2.5          # 仿真时域上限 [s]
SIM_DT = 0.05                # 仿真步长 [s]
AEB_DECEL = 8.0              # code 0 全力制动的纵向减速度指令 [m/s^2]
LANE_CHANGE_LAT_ACCEL = 3.0  # codes 1-4 变道的 bang-bang 侧向加速度指令 [m/s^2]
LANE_CHANGE_OFFSET = 4.0     # codes 1-4 变道侧向位移（到位后侧向速度归零）[m]
LC_BRAKE_DECEL = 4.0         # codes 3/4 变道同时的纵向减速度指令 [m/s^2]
TDRIFT_DECEL = 5.0           # codes 5/6 T 形避撞的纵向减速度指令 [m/s^2]
TDRIFT_LAT_ACCEL = 5.0       # codes 5/6 甩尾的持续侧向加速度指令 [m/s^2]
                             # （T 形漂移是大侧滑失稳操作，侧向加速度可达
                             #  0.5g+；合加速度 sqrt(5^2+5^2)=7.07 ≈ 漂移上限）

# --- 车辆几何（矩形 OBB 碰撞检测；替代此前的圆形近似）---
EGO_LENGTH = 4.6             # 自车车长 [m]
EGO_WIDTH = 1.9             # 自车车宽 [m]

# --- 车辆动力学（让机动指令不再瞬时理想执行）---
FRICTION_MU = 0.9           # 轮胎-路面附着系数（干沥青约 0.85~0.95）
GRAVITY = 9.81
# 常规机动(0-4)受摩擦圆约束 |a| <= mu*g；T 形漂移(5/6)是主动失稳操作，
# 轮胎已饱和滑移，不适用常规摩擦圆，单列一个更高的等效上限。
DRIFT_ACCEL_CAP = 7.0       # 漂移机动的合加速度上限 [m/s^2]
ACTUATOR_TAU = 0.12         # 执行器一阶时间常数 [s]（加速度指令的建立滞后）
JERK_MAX = 40.0             # 加加速度上限 [m/s^3]（限制加速度爬升速率）
REACTION_DELAY = 0.0        # 触发时刻起的反应延迟 [s]（触发已是决策后，默认 0）

# 目标行为模型（按意图 intention 驱动）
TARGET_BRAKE_DECEL = 6.0     # "Emergency Braking" 意图的减速度 [m/s^2]
TARGET_LC_OFFSET = 4.0       # "Lane Change" 意图：横移此距离后侧向速度归零 [m]

# 严重度模型：0 无事 / 1 险情 / 2 轻微 / 3 中等 / 4 严重
SEVERITY_NEAR_MISS_SEP = 0.5     # 最小间距低于此值（但未接触）判为险情 [m]
IMPACT_MINOR_SPEED = 3.0         # 接触相对速度低于此值 -> 轻微 [m/s]
IMPACT_MODERATE_SPEED = 8.0      # 低于此值 -> 中等；否则 -> 严重 [m/s]
SIDE_IMPACT_PENALTY = 1          # 侧碰加重一级（侧面结构弱 / 电池防护差）
REAR_IMPACT_BONUS = 1            # 尾碰减轻一级（前后吸能结构）
VULNERABLE_PENALTY = 2           # 涉及行人/骑行者的接触加重两级

# HJ 末态约束（作为参考项而非主判据）：机动末态的 HJ 风险
# 不得比"不干预"基线更差超过此容差。
HJ_BASELINE_TOLERANCE = 0.05

# ---------------------------------------------------------------------------
# 工厂：网格扫描与规则生命周期
# ---------------------------------------------------------------------------
SWEEP_MAX_POINTS = 60            # 每个场景族的网格点数上限
LAW_MARGIN = 1.0                 # "规律格点"要求的最优机动严重度优势（级）
RULE_PROMOTION_PASS_RATE = 0.9   # 候选规则晋升 active 所需的电池通过率
RULE_REVALIDATION_SAMPLE = 5     # 每批次抽样复验的 active 规则数（抗漂移）
CONSOLIDATOR_MAX_ROUNDS = 2      # 归纳提案被网格复核打回的最大重提轮数
DELIBERATION_MAX_ROUNDS = 2      # 委员会 否决->重议 的最大轮数

# ---------------------------------------------------------------------------
# 机动目录 —— 语义与已发表的 SACA 框架完全一致
# ---------------------------------------------------------------------------
DECISION_CODES = {
    0: "Full emergency braking",
    1: "Turn left sharply to change lanes and resume direction",
    2: "Turn right sharply to change lanes and resume direction",
    3: "Turn left to change lanes, with braking",
    4: "Turn right to change lanes, with braking",
    5: "T-type drift avoidance maneuver, ending with the car perpendicular to the lane, facing left",
    6: "T-type drift avoidance maneuver, ending with the car perpendicular to the lane, facing right",
    7: "No need to intervene",
}

LEFTWARD_CODES = {1, 3, 5}   # 车头/轨迹朝向 -y（左侧）的机动
RIGHTWARD_CODES = {2, 4, 6}  # 朝向 +y（右侧）的机动

# 同严重度平局时的裁决用"激进度"排序（越小越温和，优先选温和的）
AGGRESSIVENESS = {7: 0, 0: 1, 3: 2, 4: 2, 1: 3, 2: 3, 5: 4, 6: 4}

# 应急策略树的分支守卫条件（封闭集合）：反射层在触发时刻用纯运动学
# 对场景演化分类并毫秒级匹配分支，不解析任何自由文本。
BRANCH_CONDITIONS = [
    "primary_threat_maintains",   # 主威胁保持当前运动
    "primary_threat_yields",      # 主威胁减速/让行/驶离
    "primary_threat_accelerates", # 主威胁比预期更快逼近
    "secondary_threat_activates", # 其他参与者成为主要约束
    "default",                    # 兜底分支（必须存在）
]

# ---------------------------------------------------------------------------
# 注入工厂智能体的领域知识（改写自 legacy 的 DECISION_PROMPT_References，
# 内容保持一致：吸能结构、EV 电池侧碰风险、T 形漂移触发条件等）
# ---------------------------------------------------------------------------
DOMAIN_KNOWLEDGE = """\
Vehicle-structure knowledge for crash-severity reasoning:
1. Energy-absorbing structures sit at the front and rear of a vehicle; the
   sides are structurally weak. EV battery protection is likewise designed for
   frontal and rear impacts, so side collisions carry fire risk for EVs.
2. When a collision has become unavoidable, prefer to take the impact on the
   ego vehicle's front or rear, never the side.
3. T-type drift maneuver (codes 5/6): for a high-speed laterally approaching
   target with TTC <= 1.3 s that is (a) ahead of the ego vehicle, (b) within
   8 m lateral distance, and (c) closing laterally faster than 4 m/s, rotate
   the car so its REAR faces the incoming target. Drift toward whichever side
   is safer: code 5 points the nose left (rear takes impact from the right),
   code 6 points the nose right (rear takes impact from the left).
4. Codes 3/4 (lane change with braking) must state a target reduced speed and
   justify it against the remaining gap and road friction.
5. Never steer toward a side occupied by a pedestrian or cyclist, and never
   cross a road boundary.
"""
