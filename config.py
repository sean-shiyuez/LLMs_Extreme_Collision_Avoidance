"""Central configuration for MACA (Multi-Agent Collision Avoidance).

API keys live in Code/.env (see .env.example) and are loaded here via
python-dotenv. Never hardcode keys in source files.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent

load_dotenv(CODE_DIR / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Latency profiles. "realtime" (default) uses the lightest models with
# reasoning disabled and strips every non-essential LLM call from the
# deliberation loop; "quality" is the original flagship configuration.
# Model names verified against the OpenAI API lineup as of 2026-07.
# Any role can be overridden via environment, e.g. MACA_MODEL_DECISION=...
# ---------------------------------------------------------------------------
_MODELS = {
    "quality": {
        "decision": "gpt-5.5",        # deliberation-phase policy tree synthesis
        "decision_fast": "gpt-5.4-mini",  # near-deadline downgraded re-decision
        "arbiter": "gpt-5.5",         # debate arbiter
        "advocate": "gpt-5.4-mini",   # safety / efficiency advocates in debate
        "safety": "gpt-5.4",          # safety shield LLM review
        "perception": "gpt-5.4",      # VLM over the BEV image
        "risk": "gpt-5.4-mini",       # risk report summarizer
        "evaluation": "gpt-5.4-mini", # post-hoc reflection
    },
    "realtime": {
        "decision": "gpt-5.4-mini",
        "decision_fast": "gpt-5.4-nano",
        "arbiter": "gpt-5.4-mini",
        "advocate": "gpt-5.4-nano",
        "safety": "gpt-5.4-mini",     # unused unless SKIP_SAFETY_LLM is off
        "perception": "gpt-5.4-mini",
        "risk": "gpt-5.4-nano",       # unused unless SKIP_RISK_LLM is off
        "evaluation": "gpt-5.4-nano",
    },
}

PROFILE = os.environ.get("MACA_PROFILE", "realtime")

# "none" fully disables thinking on gpt-5.4+ (supported values: none, low,
# medium, high, xhigh; none is required for low-latency use).
REASONING_EFFORT = os.environ.get("MACA_REASONING_EFFORT", "none")

# Realtime-profile switches (recomputed by apply_profile):
SKIP_RISK_LLM = True      # risk summary from a deterministic template
SKIP_SAFETY_LLM = True    # shield = deterministic physics checks only
EVIDENCE_UPFRONT = True   # precompute the physics evidence pack; the decision
                          # agent runs one structured call, no tool round-trips


def apply_profile(name: str):
    """Switch latency profile at runtime (used by the CLI --profile flag)."""
    global PROFILE, REASONING_EFFORT, SKIP_RISK_LLM, SKIP_SAFETY_LLM, EVIDENCE_UPFRONT
    if name not in _MODELS:
        raise ValueError(f"unknown profile {name}; use one of {list(_MODELS)}")
    PROFILE = name
    realtime = name == "realtime"
    SKIP_RISK_LLM = realtime
    SKIP_SAFETY_LLM = realtime
    EVIDENCE_UPFRONT = realtime
    REASONING_EFFORT = os.environ.get(
        "MACA_REASONING_EFFORT", "none" if realtime else "low")


apply_profile(PROFILE)


def model_for(role: str) -> str:
    return os.environ.get(f"MACA_MODEL_{role.upper()}", _MODELS[PROFILE][role])


EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = str(CODE_DIR / "collision_avoidance_chroma_db")
CASE_COLLECTION = "maca_cases"
LEGACY_COLLECTION = "langchain"  # collection written by the legacy SACA code
RUNS_DIR = CODE_DIR / "Results" / "runs"
DISTILL_EXPORT_PATH = CODE_DIR / "fine-tuning" / "maca_distill.jsonl"

NUM_SIMILAR_CASES = 2            # k for case retrieval
PRECOMPUTED_POLICY_MAX_DISTANCE = 0.15  # cosine distance gate for reusing a cached policy

# ---------------------------------------------------------------------------
# Reflex-loop thresholds and timing (seconds)
# ---------------------------------------------------------------------------
RISK_LOW = 0.40      # below: no intervention (code 7), deliberation not woken
RISK_TRIGGER = 0.63  # at/above: execute from the armed policy cache
TTC_URGENCY_HORIZON = 3.0   # TTC (s) mapped linearly onto urgency [0..1]
REAR_HJ_DISCOUNT = 0.3  # HJ weight for behind-the-ego targets not on a collision course
T_EXEC_TTC = 1.2     # anticipated TTC at the trigger instant; deliberation projects
                     # the scene forward to this state before validating maneuvers

T_ACTUATION = 0.3    # actuation + control latency reserved out of the TTC budget
DEADLINE_MARGIN = 0.2
SAFETY_MAX_ROUNDS = 2   # veto -> re-decide rounds before falling back to code 0
ROLLOUT_HORIZON = 1.0   # s, kinematic rollout horizon for the safety shield
HJ_SAFE_RISK = 0.70     # rollout end-state HJ risk above this fails a branch

# ---------------------------------------------------------------------------
# Maneuver catalogue — semantics identical to the published SACA framework.
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

LEFTWARD_CODES = {1, 3, 5}   # maneuvers that move/point the nose toward -y (left)
RIGHTWARD_CODES = {2, 4, 6}  # toward +y (right)

# Contingency-policy branch guards. The LLM must label every branch with one of
# these so the reflex loop can match the observed scene evolution in
# milliseconds, without parsing free text.
BRANCH_CONDITIONS = [
    "primary_threat_maintains",   # primary threat keeps its current motion
    "primary_threat_yields",      # primary threat brakes / turns away
    "primary_threat_accelerates", # primary threat closes faster than predicted
    "secondary_threat_activates", # another participant becomes the binding constraint
    "default",
]

# ---------------------------------------------------------------------------
# Domain knowledge injected into the decision agent (rewritten from the
# legacy DECISION_PROMPT_References; content preserved).
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
