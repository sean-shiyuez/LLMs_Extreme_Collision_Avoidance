"""工厂各智能体结构化输出的严格 JSON schema。

注意 OpenAI strict 模式的两条约束（本文件多处为其绕行）：
每个 object 的所有属性都必须列入 required；不允许自由格式的 dict。
故规则守卫以"条目数组"（array-of-entries）传输，解析后再用
guards_from_entries 还原成引擎的 dict 形式。
"""
from .. import config


def strict_schema(name: str, properties: dict, required=None) -> dict:
    return {
        "name": name,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
            "additionalProperties": False,
        },
    }


_BRANCH_ITEM = {
    "type": "object",
    "properties": {
        "condition_type": {"type": "string", "enum": config.BRANCH_CONDITIONS},
        "condition": {"type": "string"},
        "code": {"type": "integer", "minimum": 0, "maximum": 7},
        "target_speed_mps": {"type": "number"},
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["condition_type", "condition", "code", "target_speed_mps",
                 "rationale", "confidence"],
    "additionalProperties": False,
}

POLICY_PROPS = {
    "branches": {"type": "array", "items": _BRANCH_ITEM},
    "overall_rationale": {"type": "string"},
}

POLICY_SCHEMA = strict_schema("contingency_policy", POLICY_PROPS)

# Scenario family designed by the generator agent.
FAMILY_SCHEMA = strict_schema("scenario_family", {
    "family_id": {"type": "string", "description": "short-kebab-case id"},
    "description": {"type": "string"},
    "pattern": {"type": "string",
                "enum": ["lateral_crossing", "lead_braking", "cut_in",
                         "static_blockage"]},
    "template": {
        "type": "object",
        "properties": {
            "road_topology": {"type": "string",
                              "enum": ["Normal Road", "Intersection", "Roundabout"]},
            "threat_type": {"type": "string"},
            "from_side": {"type": "string", "enum": ["left", "right"]},
            "pedestrian": {
                "type": "object",
                "properties": {"present": {"type": "boolean"},
                               "side": {"type": "string", "enum": ["left", "right"]}},
                "required": ["present", "side"], "additionalProperties": False},
            "road_boundary_left": {"type": ["number", "null"]},
            "road_boundary_right": {"type": ["number", "null"]},
            "follower": {"type": "boolean"},
            "blocked_side": {"type": ["string", "null"],
                             "enum": ["left", "right", None]},
        },
        "required": ["road_topology", "threat_type", "from_side", "pedestrian",
                     "road_boundary_left", "road_boundary_right", "follower",
                     "blocked_side"],
        "additionalProperties": False,
    },
    "parameters": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "min": {"type": "number"}, "max": {"type": "number"},
                       "steps": {"type": "integer", "minimum": 2, "maximum": 6}},
        "required": ["name", "min", "max", "steps"],
        "additionalProperties": False}},
    "hypothesis": {"type": "string",
                   "description": "what law this family is expected to reveal"},
})

# OpenAI strict mode forbids free-form dicts, so guards travel as an array of
# entries and are converted back to the engine's dict form after parsing.
FEATURE_KEYS = ["road_topology", "threat_kind", "approach_sector",
                "pedestrian_side", "corridor_left_free", "corridor_right_free",
                "braking_sufficient", "on_collision_course",
                "ttc_s", "miss_m", "lateral_dist_m", "lateral_speed_mps",
                "closing_speed_mps", "ego_speed_mps", "hj_risk", "threat_dist_m"]

_GUARD_ENTRIES = {"type": "array", "items": {
    "type": "object",
    "properties": {
        "feature": {"type": "string", "enum": FEATURE_KEYS},
        "allowed_values": {"type": ["array", "null"],
                           "items": {"type": ["string", "boolean"]},
                           "description": "for discrete features; null for numeric"},
        "min": {"type": ["number", "null"]},
        "max": {"type": ["number", "null"]},
    },
    "required": ["feature", "allowed_values", "min", "max"],
    "additionalProperties": False}}


def guards_from_entries(entries) -> dict:
    """Array-of-entries -> engine dict form. Passes dicts through unchanged
    (MockLLM emits the dict form directly)."""
    if isinstance(entries, dict):
        return entries
    guards = {}
    for e in entries or []:
        if e.get("allowed_values"):
            guards[e["feature"]] = e["allowed_values"]
        else:
            rng = {}
            if e.get("min") is not None:
                rng["min"] = e["min"]
            if e.get("max") is not None:
                rng["max"] = e["max"]
            if rng:
                guards[e["feature"]] = rng
    return guards


RULE_PROPOSALS_SCHEMA = strict_schema("rule_proposals", {
    "proposals": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "guards": _GUARD_ENTRIES,
            "policy": {"type": "object", "properties": POLICY_PROPS,
                       "required": list(POLICY_PROPS.keys()),
                       "additionalProperties": False},
            "rationale": {"type": "string"},
        },
        "required": ["guards", "policy", "rationale"],
        "additionalProperties": False}},
    "amendments": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "rule_id": {"type": "string"},
            "action": {"type": "string", "enum": ["deprecate", "tighten"]},
            "new_guards": {"anyOf": [_GUARD_ENTRIES, {"type": "null"}]},
            "reason": {"type": "string"},
        },
        "required": ["rule_id", "action", "new_guards", "reason"],
        "additionalProperties": False}},
    "analysis": {"type": "string"},
})

EVALUATION_SCHEMA = strict_schema("evaluation_report", {
    "assessment": {"type": "string"},
    "decision_was_effective": {"type": "boolean"},
    "lesson": {"type": "string"},
})

ADVOCATE_SCHEMA = strict_schema("advocate_position", {
    "argument": {"type": "string", "description": "<=4 sentences"},
})

ARBITER_SCHEMA = strict_schema("arbiter_resolution", {
    "resolution": {"type": "string"},
    "revised_codes": {"type": "array", "items": {
        "type": "object",
        "properties": {"condition_type": {"type": "string",
                                          "enum": config.BRANCH_CONDITIONS},
                       "code": {"type": "integer", "minimum": 0, "maximum": 7}},
        "required": ["condition_type", "code"], "additionalProperties": False}},
})
