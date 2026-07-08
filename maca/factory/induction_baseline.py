"""无 LLM 的规则归纳基线：把规律摘要中观测到的特征范围直接作为守卫，
生成规则提案。两个用途：(1) 供 MockLLM 在 --mock 离线工厂中调用；
(2) 作为真实归纳器必须超越的"合理性下限"（sanity floor）。
"""
from typing import Dict, List

from .. import config

# Guards worth generalizing on (subset of the feature language; the LLM
# consolidator may use any feature, the baseline keeps to the discriminative core).
_NUMERIC_GUARD_KEYS = ["ttc_s", "lateral_speed_mps", "lateral_dist_m", "ego_speed_mps"]
_DISCRETE_GUARD_KEYS = ["approach_sector", "pedestrian_side", "threat_kind",
                        "on_collision_course", "braking_sufficient"]


def _parse_discrete(values: List[str]) -> List:
    out = []
    for v in values:
        if v == "True":
            out.append(True)
        elif v == "False":
            out.append(False)
        else:
            out.append(v)
    return out


def induce(lawful_summary: Dict, family_id: str) -> List[dict]:
    """对每个占优机动码，把其规律区域的离散取值/数值范围转成守卫，
    生成一条规则提案（证据点数 <3 的跳过）。"""
    proposals = []
    for code_str, region in lawful_summary.items():
        code = int(code_str)
        if region["points"] < 3:
            continue  # too little evidence for a law
        guards = {}
        for key in _DISCRETE_GUARD_KEYS:
            vals = region["discrete_values"].get(key)
            if vals and len(vals) <= 3:
                guards[key] = _parse_discrete(vals)
        for key in _NUMERIC_GUARD_KEYS:
            rng = region["numeric_ranges"].get(key)
            if rng and rng["max"] < 90:  # skip unbounded sentinels
                pad = max(0.05, (rng["max"] - rng["min"]) * 0.05)
                guards[key] = {"min": round(rng["min"] - pad, 3),
                               "max": round(rng["max"] + pad, 3)}
        yields_code = 0 if code != 7 else 7
        policy = {"branches": [
            {"condition_type": "primary_threat_maintains",
             "condition": "threat evolves as observed in the sweep",
             "code": code, "target_speed_mps": 0.0,
             "rationale": f"Dominant maneuver over {region['points']} lawful grid "
                          f"points (mean margin {region['mean_margin']}).",
             "confidence": min(0.95, 0.6 + 0.05 * region["points"])},
            {"condition_type": "primary_threat_yields",
             "condition": "threat brakes or turns away",
             "code": yields_code, "target_speed_mps": 0.0,
             "rationale": "Milder response suffices once the threat yields.",
             "confidence": 0.85},
            {"condition_type": "default", "condition": "any other evolution",
             "code": 0 if code != 7 else 7, "target_speed_mps": 0.0,
             "rationale": "Minimal-risk fallback.", "confidence": 0.85},
        ]}
        proposals.append({
            "guards": guards,
            "policy": policy,
            "rationale": f"Grid-swept law: maneuver {code} "
                         f"({config.DECISION_CODES[code]}) dominates the region "
                         f"described by the guards ({region['points']} points).",
            "expected_points": region["points"],
        })
    return proposals
