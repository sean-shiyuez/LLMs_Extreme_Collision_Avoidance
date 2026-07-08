"""运行时 -> 工厂的反馈通道：未命中场景（gap）与规则-案例冲突（conflict）
追加到 queue.jsonl，工厂每批次优先消化。这是"覆盖门控激活"的入口 ——
只有库处理不了的场景才会消耗离线 LLM。"""
import json
import time

from .. import config


def enqueue(kind: str, scenario_name: str, snapshot_dict: dict,
            features: dict, details: dict):
    """入队一条反馈记录。kind: "gap"（无覆盖）| "conflict"（规则案例分歧）。"""
    config.QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": kind,
        "scenario_name": scenario_name,
        "snapshot": snapshot_dict,     # 完整场景快照（工厂可复现）
        "features": features,          # 入队时刻的特征（覆盖复查用）
        "details": details,            # gap: 观测条件；conflict: 仲裁报告
        "queued_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config.QUEUE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
