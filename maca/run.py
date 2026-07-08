"""MACA 在线运行时 CLI —— 编译产物平面。无 LLM、无网络、无需 API key。

    python -m maca.run --scenario test1     # 对某场景做一次在线决策
    python -m maca.run --list               # 列出全部场景
    python -m maca.run --stats              # 打印三个库的规模统计
"""
import argparse
import json
import time

from . import config
from .library import store
from .runtime.executor import RuntimeExecutor
from .scenario.loader import available_scenarios, load_scenario


def _print(record: dict):
    div = "=" * 72
    print(f"\n{div}\nMACA runtime — scenario '{record['scenario']}' "
          f"(rules={record['libraries']['rules_active']} active, "
          f"cases={record['libraries']['cases']})\n{div}")
    a = record["assessment"]
    t = record["timings"]
    print(f"[Monitor]   scene_risk={a['scene_risk']} ({a['band']}), "
          f"min_TTC={a['min_ttc_s']}, primary={a['primary_threat']} "
          f"({t['monitor_ms']} ms)")
    if "features" in record:
        d = record["features"]["discrete"]
        print(f"[Features]  sector={d['approach_sector']}, threat={d['threat_kind']}, "
              f"ped_side={d['pedestrian_side']}, braking_ok={d['braking_sufficient']}, "
              f"corridors L/R={d['corridor_left_free']}/{d['corridor_right_free']} "
              f"({t.get('features_ms')} ms)")
        print(f"[Match]     rule={record.get('rule_hit')} ({t.get('rule_match_ms')} ms) | "
              f"case={record.get('case_hit')} sim={record.get('case_similarity')} "
              f"({t.get('case_knn_ms')} ms)")
    if "conflict" in record:
        c = record["conflict"]
        tail = ("already adjudicated offline; rule awaits amendment"
                if record.get("conflict_adjudicated")
                else "queued for offline debate")
        print(f"[Conflict]  rule code {c['rule']['code']} (sev {c['rule']['severity']}) vs "
              f"case code {c['case']['code']} (sev {c['case']['severity']}) -> "
              f"{c['chosen']} ({t.get('arbitration_ms')} ms); {tail}")
    print(f"[Decision]  code {record['executed_code']} "
          f"({config.DECISION_CODES[record['executed_code']]})")
    print(f"            source: {record['decision_source']} | "
          f"observed: {record.get('observed_condition')}")
    if record.get("gap_queued"):
        print("            [gap] scene not covered by any library -> queued for the factory")
    print(f"[Latency]   TRIGGER PATH TOTAL: {t['trigger_path_total_ms']} ms "
          f"(zero LLM calls, zero network)")
    o = record["outcome"]
    print(f"[Outcome]   severity {o['severity']}/4 — {o['text']}")
    print(div)


def main():
    parser = argparse.ArgumentParser(
        prog="maca.run",
        description="MACA online runtime: rule/case library matching, LLM-free")
    parser.add_argument("--scenario", default="test1",
                        help=f"one of: {', '.join(available_scenarios())}")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stats", action="store_true", help="print library stats")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in available_scenarios():
            print(name)
        return
    if args.stats:
        print(json.dumps(store.stats(), indent=1))
        return

    scenario = load_scenario(args.scenario)
    executor = RuntimeExecutor()
    record = executor.decide(scenario)
    record["ran_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RUNS_DIR / f"{scenario.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1, default=str)
    record["log_path"] = str(path)

    if args.json:
        print(json.dumps(record, indent=1, ensure_ascii=False, default=str))
    else:
        _print(record)
        print(f"Full record: {path}")


if __name__ == "__main__":
    main()
