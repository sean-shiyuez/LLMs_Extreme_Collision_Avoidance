"""编译库的 JSONL 持久化（规则库 / 案例库 / 反馈队列）。

选用 JSONL 的理由：每行一条、可 git diff、可人工审阅、追加写入天然
安全；库的在线读取由 rule_engine/case_index 各自负责，这里只管落盘。
"""
import json
import time
import uuid
from typing import List, Optional

from .. import config


def _read_jsonl(path) -> List[dict]:
    items = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def _append_jsonl(path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl(path, records: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- 规则库 ----------------

def load_rules() -> List[dict]:
    return _read_jsonl(config.RULES_PATH)


def save_rules(rules: List[dict]):
    _write_jsonl(config.RULES_PATH, rules)


def upsert_rule(rule: dict):
    """按 rule_id 更新或新增规则；更新时版本号自增。"""
    rules = load_rules()
    for i, r in enumerate(rules):
        if r["rule_id"] == rule["rule_id"]:
            rule["version"] = int(r.get("version", 1)) + 1
            rules[i] = rule
            save_rules(rules)
            return rule
    rules.append(rule)
    save_rules(rules)
    return rule


# ---------------- 案例库 ----------------

def add_case(features: dict, policy: dict, lesson: str, severity: int,
             outcome: str, scenario_name: str, source: str,
             executed_code: Optional[int] = None) -> dict:
    """追加一条案例。source 标注来源：committee（委员会标注模糊格点）/
    conflict_resolution（冲突裁决）/ gap_fill（运行时缺口补齐）/ seed。"""
    case = {
        "case_id": f"case-{uuid.uuid4().hex[:8]}",
        "scenario_name": scenario_name,
        "source": source,
        "features": features,
        "policy": policy,
        "executed_code": executed_code,
        "severity": severity,
        "outcome": outcome,
        "lesson": lesson,
        "stored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _append_jsonl(config.CASES_PATH, case)
    return case


def load_cases() -> List[dict]:
    return _read_jsonl(config.CASES_PATH)


def retire_case(case_id: str, reason: str) -> bool:
    """将案例标记为退役（保留在磁盘上作为来源追溯，但不再进入在线索引）。
    用于离线冲突裁决判其败诉时 —— 库的自愈机制之一。"""
    cases = load_cases()
    hit = False
    for c in cases:
        if c["case_id"] == case_id:
            c["retired"] = True
            c["retired_reason"] = reason
            hit = True
    if hit:
        _write_jsonl(config.CASES_PATH, cases)
    return hit


# ---------------- 反馈队列 ----------------

def read_queue() -> List[dict]:
    return _read_jsonl(config.QUEUE_PATH)


def drain_queue() -> List[dict]:
    """取出全部队列项并清空队列文件（工厂每批次开头调用）。"""
    items = read_queue()
    if config.QUEUE_PATH.exists():
        config.QUEUE_PATH.unlink()
    return items


def stats() -> dict:
    """三个库的规模统计（CLI --stats 与工厂报告用）。"""
    rules = load_rules()
    return {
        "rules_total": len(rules),
        "rules_active": sum(1 for r in rules if r.get("status") == "active"),
        "rules_candidate": sum(1 for r in rules if r.get("status") == "candidate"),
        "rules_deprecated": sum(1 for r in rules if r.get("status") == "deprecated"),
        "cases": len(load_cases()),
        "queued": len(read_queue()),
    }
