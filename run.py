"""MACA CLI.

    python -m maca.run --scenario test1 [--mock] [--debate auto|on|off]
                       [--no-vision] [--enforce-deadline]
    python -m maca.run --list
"""
import argparse
import asyncio
import json

from . import config
from .orchestrator import Orchestrator
from .scenario.loader import available_scenarios


def _print_summary(record: dict):
    div = "=" * 72
    print(f"\n{div}\nMACA run — scenario '{record['scenario']}'"
          + (" [MOCK]" if record["mock"] else "") + f"\n{div}")

    rp = record["phases"]["reflex_preview"]
    print(f"[Reflex @ preview]  scene_risk={rp['scene_risk']} ({rp['band']}), "
          f"min_TTC={rp['min_ttc_s']}s, primary_threat={rp['primary_threat']}, "
          f"assessed in {rp['elapsed_ms']} ms")
    print(f"                    cognition budget: tier={rp['budget']['tier']}, "
          f"deadline={rp['budget']['deadline_s']:.2f}s, "
          f"vision={rp['budget']['allow_vision']}, debate={rp['budget']['allow_debate']}")

    d = record["phases"]["deliberation"]
    if d.get("skipped"):
        print(f"[Deliberation]      skipped ({d['reason']})")
    else:
        ap = d["armed_policy"]
        print(f"[Deliberation]      {d['elapsed_s']}s "
              f"(deadline {d['deadline_s']}s, met={d['deadline_met']}), "
              f"safety rounds={d['safety_rounds']}, debate={d['debate_ran']}, "
              f"source={ap['source']}")
        for b in ap["branches"]:
            print(f"   ├─ if {b['condition_type']:<28} -> code {b['code']} "
                  f"({config.DECISION_CODES[b['code']]}); conf={b['confidence']}")

    rt = record["phases"]["reflex_trigger"]
    print(f"[Reflex @ trigger]  scene_risk={rt['scene_risk']} ({rt['band']}) -> "
          f"code {rt['executed_code']} ({rt['maneuver']})")
    print(f"                    decision source: {rt['decision_source']}")
    print(f"                    TRIGGER-PATH LATENCY: "
          f"{rt['trigger_decision_latency_ms']} ms "
          f"(assess {rt['assess_ms']} + match {rt['match_ms']}; zero LLM calls)")

    print(f"[Execution]         {record['execution_outcome']}")
    if "consolidation" in record["phases"]:
        c = record["phases"]["consolidation"]
        print(f"[Consolidation]     lesson: {c['lesson']}")
    print(f"\nFull trace: {record['log_path']}\n{div}")


def main():
    parser = argparse.ArgumentParser(
        prog="maca", description="MACA — multi-agent collision avoidance decision demo")
    parser.add_argument("--scenario", default="test1",
                        help=f"one of: {', '.join(available_scenarios())}")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--mock", action="store_true",
                        help="run offline with deterministic mock agents (no API key)")
    parser.add_argument("--debate", choices=["auto", "on", "off"], default="auto",
                        help="consistency-gated debate mode (default: auto)")
    parser.add_argument("--no-vision", action="store_true",
                        help="skip BEV rendering / the VLM perception channel")
    parser.add_argument("--enforce-deadline", action="store_true",
                        help="hard-cancel deliberation at the TTC deadline "
                             "(anytime fallback to minimal-risk policy)")
    parser.add_argument("--json", action="store_true", help="print the raw run record")
    args = parser.parse_args()

    if args.list:
        for name in available_scenarios():
            print(name)
        return

    if not args.mock and not config.OPENAI_API_KEY:
        parser.error("OPENAI_API_KEY is not set (see .env.example); "
                     "use --mock for an offline run")

    orch = Orchestrator(mock=args.mock, debate_mode=args.debate,
                        no_vision=args.no_vision,
                        enforce_deadline=args.enforce_deadline)
    record = asyncio.run(orch.run_scenario(args.scenario))
    if args.json:
        print(json.dumps(record, indent=1, ensure_ascii=False, default=str))
    else:
        _print_summary(record)


if __name__ == "__main__":
    main()
