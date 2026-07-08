"""Factory CLI — the offline law-discovery loop.

    python -m maca.factory --batch 1 [--mock] [--source queue|generate|both]

Each batch: (1) drain the runtime queue (gaps -> committee -> cases;
conflicts -> mandatory debate -> cases), then (2) design N scenario families
(thinking LLM), sweep their parameter grids (deterministic simulation),
induce rules from the lawful regions (thinking LLM, numerically
cross-checked), gate them through the promotion battery, and send ambiguous
grid points to the committee as cases. Active rules are sample-re-validated
against their re-instantiated source families every batch.
"""
import argparse
import asyncio
import json
import time

from .. import config
from ..library import store
from ..llm import MockLLM, OpenAIClient
from ..runtime import monitor
from ..runtime.case_index import CaseLibrary
from ..runtime.features import features_from_dict
from ..runtime.rule_engine import RuleLibrary
from ..scenario.schema import snapshot_from_dict
from ..tools import simulator
from . import consolidator, coverage, generator, growth_log, instantiate, sweep, validator
from .committee import Committee

AMBIGUOUS_CASES_PER_FAMILY = 3


async def run_batch(n_families: int, mock: bool, source: str) -> dict:
    """跑一个工厂批次：阶段 1 消化运行时反馈队列（gap/冲突 -> 案例），
    阶段 2 生成 n_families 个场景族并做 扫描->归纳->晋升 与 模糊格点->
    委员会->案例，最后抗漂移复验现役规则。返回批次报告。"""
    llm = MockLLM() if mock else OpenAIClient()
    committee = Committee(llm)
    report = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "mock": mock,
              "source": source, "queue_processed": [], "families": [],
              "revalidation": [], "library_before": store.stats()}
    growth_log.batch_start(mock, source, report["library_before"])

    # ---- 阶段 1：消化运行时反馈（gap + 冲突）----
    if source in ("queue", "both"):
        for item in store.drain_queue():
            is_conflict = item["kind"] == "conflict"
            # 新颖性门控：库已能正确处理的项直接跳过（如更早的项已覆盖它）
            # (e.g. duplicates queued before an earlier item covered them).
            snapshot = snapshot_from_dict(item["snapshot"])
            assessment = monitor.assess_scene(snapshot)
            rank = simulator.rank_maneuvers(snapshot,
                                            primary_id=assessment.primary_threat)
            cov = coverage.check_coverage(
                features_from_dict(item["features"]),
                rank["results"][rank["best"]]["severity"],
                coverage.severities_of(rank),
                RuleLibrary.load(), CaseLibrary.load())
            if cov["covered"]:
                report["queue_processed"].append(
                    {"kind": item["kind"], "scenario": item.get("scenario_name"),
                     "skipped": True, "covered_by": cov["by"]})
                growth_log.queue_skipped(item["kind"], item.get("scenario_name", "?"),
                                         cov["by"])
                continue
            # 绝境场景过滤（与网格扫描同一判据）：所有机动都是最高严重度且
            # 无区分度的 gap，学不到有价值的决策规律 —— 丢弃，不建无意义案例
            # （避免"推荐某机动"却其实人人都撞的误导性案例污染安全库）。
            sevs = [r["severity"] for r in rank["results"].values()]
            best_sev = rank["results"][rank["best"]]["severity"]
            if not is_conflict and best_sev >= 4 and rank["margin"] == 0 and min(sevs) >= 4:
                report["queue_processed"].append(
                    {"kind": item["kind"], "scenario": item.get("scenario_name"),
                     "skipped": True, "reason": "doomed"})
                growth_log.queue_doomed(item["kind"], item.get("scenario_name", "?"))
                continue
            growth_log.queue_processed(item["kind"], item.get("scenario_name", "?"))
            rec = await committee.deliberate(
                snapshot,
                source="conflict_resolution" if is_conflict else "gap_fill",
                conflict_details=item["details"] if is_conflict else None,
                scenario_name=item.get("scenario_name", "queue-item"))
            case = store.add_case(rec["features"], rec["policy"], rec["lesson"],
                                  rec["severity"], rec["outcome"],
                                  rec["scenario_name"], rec["source"],
                                  rec["executed_code"])
            growth_log.case_added(case, rec)
            entry = {"kind": item["kind"], "scenario": rec["scenario_name"],
                     "case_id": case["case_id"],
                     "executed_code": rec["executed_code"],
                     "severity": rec["severity"], "effective": rec["effective"]}
            if is_conflict:
                # Self-healing: if the committee rules against the stored
                # case's verdict, retire that case so the conflict cannot
                # recur (the rule side is covered by re-validation).
                losing = item["details"].get("case_id")
                case_code = item["details"].get("arbitration", {}).get("case", {}).get("code")
                if losing and case_code is not None \
                        and int(case_code) != rec["executed_code"]:
                    if store.retire_case(losing,
                                         f"overruled by conflict resolution "
                                         f"{case['case_id']}"):
                        entry["retired_case"] = losing
                        growth_log.case_retired(
                            losing, f"冲突裁决败诉，被 {case['case_id']} 取代")
            report["queue_processed"].append(entry)

    # ---- 阶段 2：规律发现（生成场景族 -> 扫描 -> 归纳 -> 晋升 / 案例）----
    if source in ("generate", "both"):
        for _ in range(n_families):
            rules_all = store.load_rules()
            briefing = generator.coverage_briefing(rules_all, store.stats())
            family = await generator.design_family(llm, briefing)
            meta = family.pop("_generator_meta", {})
            _save_family(family)

            sweep_result = sweep.run_family(family)

            # Novelty gate over the grid: LLM consolidation only runs on the
            # lawful region NOT already covered by an optimal active rule;
            # rules that match but are severity-suboptimal become
            # counterexample notes for the consolidator.
            rule_lib, case_lib = RuleLibrary.load(), CaseLibrary.load()
            uncovered_lawful, counterexamples = [], []
            for point in sweep_result["lawful"]:
                cov = coverage.check_coverage(
                    features_from_dict(point["features"]),
                    point["severities"][point["best"]], point["severities"],
                    rule_lib, case_lib, count_cases=False)
                counterexamples.extend(cov["rule_suboptimal"])
                if not cov["covered"]:
                    uncovered_lawful.append(point)
            fam_report = {
                "family_id": family["family_id"], "pattern": family["pattern"],
                "hypothesis": family.get("hypothesis", ""),
                "generator": meta,
                "points": len(sweep_result["points"]),
                "lawful": len(sweep_result["lawful"]),
                "lawful_uncovered": len(uncovered_lawful),
                "ambiguous": len(sweep_result["ambiguous"]),
                "doomed": len(sweep_result.get("doomed", [])),
                "rule_counterexamples": len(counterexamples),
                "rules": [], "cases": [], "skipped_covered_points": 0}
            growth_log.family_swept(fam_report)

            if len(uncovered_lawful) < 3 and not counterexamples:
                fam_report["consolidator"] = {
                    "skipped": True,
                    "reason": f"lawful region already covered "
                              f"({len(sweep_result['lawful'])} points, "
                              f"{len(uncovered_lawful)} uncovered)"}
                growth_log.consolidation_skipped(fam_report["consolidator"]["reason"])
                cons = {"accepted": [], "rejected": [], "amendments": []}
            else:
                cons = await consolidator.consolidate(llm, sweep_result, rules_all,
                                                      conflict_notes=counterexamples[:8])
                fam_report["consolidator"] = {
                    "accepted": len(cons["accepted"]),
                    "rejected": len(cons["rejected"]),
                    "trace": cons["trace"]}
                for rej in cons["rejected"]:
                    growth_log.proposal_rejected(rej["proposal"]["guards"],
                                                 rej["verification"]["problems"])
            for i, acc in enumerate(cons["accepted"]):
                rule_dict = {
                    "rule_id": f"{family['family_id']}-r{i + 1}",
                    "version": 1, "status": "candidate", "priority": 1,
                    "guards": acc["proposal"]["guards"],
                    "policy": acc["proposal"]["policy"],
                    "rationale": acc["proposal"]["rationale"],
                    "provenance": {"source": "consolidator", "family": family,
                                   "grid_coverage": acc["verification"]["covered"],
                                   "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                    "stats": {}}
                verdict = validator.validate_rule(rule_dict, sweep_result)
                rule_dict = validator.promote(rule_dict, verdict)
                store.upsert_rule(rule_dict)
                growth_log.rule_validated(
                    rule_dict["rule_id"],
                    consolidator._maintains_code(rule_dict["policy"]),
                    verdict, rule_dict["status"])
                fam_report["rules"].append({"rule_id": rule_dict["rule_id"],
                                            "status": rule_dict["status"],
                                            **verdict})
            _apply_amendments(cons.get("amendments", []), fam_report)

            deliberated = 0
            for point in sweep_result["ambiguous"]:
                if deliberated >= AMBIGUOUS_CASES_PER_FAMILY:
                    break
                # Novelty gate: committee time is only spent on points no
                # rule or case (incl. ones just added) handles correctly.
                cov = coverage.check_coverage(
                    features_from_dict(point["features"]),
                    point["severities"][point["best"]], point["severities"],
                    RuleLibrary.load(), CaseLibrary.load())
                if cov["covered"]:
                    fam_report["skipped_covered_points"] += 1
                    continue
                deliberated += 1
                scenario = instantiate.instantiate(family, point["params"])
                rec = await committee.deliberate(scenario.trigger,
                                                 source="committee",
                                                 scenario_name=scenario.name)
                case = store.add_case(rec["features"], rec["policy"], rec["lesson"],
                                      rec["severity"], rec["outcome"],
                                      rec["scenario_name"], rec["source"],
                                      rec["executed_code"])
                growth_log.case_added(case, rec)
                fam_report["cases"].append({
                    "case_id": case["case_id"], "params": point["params"],
                    "executed_code": rec["executed_code"],
                    "severity": rec["severity"]})
            if fam_report["skipped_covered_points"]:
                growth_log.consolidation_skipped(
                    f"{fam_report['skipped_covered_points']} 个模糊格点已被库覆盖，跳过委员会")
            report["families"].append(fam_report)

        # ---- 批末：抗漂移复验现役规则 ----
        rules_all = store.load_rules()
        report["revalidation"] = validator.revalidate_active(rules_all)
        store.save_rules(rules_all)
        for rv in report["revalidation"]:
            growth_log.revalidation(rv)

    report["library_after"] = store.stats()
    config.FACTORY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FACTORY_REPORTS_DIR / f"batch-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    report["report_path"] = str(path)
    growth_log.batch_end(report["library_after"], str(path))
    return report


def _save_family(family: dict):
    config.GENERATED_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GENERATED_SCENARIOS_DIR / f"{family['family_id']}.family.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(family, f, ensure_ascii=False, indent=1)


def _apply_amendments(amendments, fam_report):
    if not amendments:
        return
    rules = store.load_rules()
    applied = []
    for am in amendments:
        for r in rules:
            if r["rule_id"] != am["rule_id"]:
                continue
            if am["action"] == "deprecate":
                r["status"] = "deprecated"
                r.setdefault("stats", {})["deprecated_reason"] = am["reason"]
            elif am["action"] == "tighten" and am.get("new_guards"):
                r["guards"] = am["new_guards"]
                r["version"] = int(r.get("version", 1)) + 1
                r["status"] = "candidate"  # must re-earn promotion
            applied.append({"rule_id": am["rule_id"], "action": am["action"],
                            "reason": am["reason"]})
            growth_log.amendment(am["action"], am["rule_id"], am["reason"])
    if applied:
        store.save_rules(rules)
    fam_report["amendments"] = applied


def _print(report: dict):
    div = "=" * 72
    print(f"\n{div}\nMACA factory batch"
          + (" [MOCK]" if report["mock"] else "") + f"\n{div}")
    print(f"[Library]   before: {report['library_before']}")
    for q in report["queue_processed"]:
        if q.get("skipped"):
            why = (f"already covered by {q['covered_by']}" if q.get("covered_by")
                   else q.get("reason", "skipped"))
            print(f"[Queue]     {q['kind']}: {q['scenario']} -> SKIPPED ({why})")
        else:
            print(f"[Queue]     {q['kind']}: {q['scenario']} -> case {q['case_id']} "
                  f"(code {q['executed_code']}, severity {q['severity']})")
    for fam in report["families"]:
        print(f"[Family]    {fam['family_id']} ({fam['pattern']}) — "
              f"{fam['points']} points: {fam['lawful']} lawful "
              f"({fam['lawful_uncovered']} uncovered) / {fam['ambiguous']} ambiguous"
              + (f"; {fam['rule_counterexamples']} rule counterexamples"
                 if fam.get("rule_counterexamples") else ""))
        print(f"            hypothesis: {fam['hypothesis']}")
        if fam.get("consolidator", {}).get("skipped"):
            print(f"   ├─ consolidation SKIPPED: {fam['consolidator']['reason']}")
        if fam.get("skipped_covered_points"):
            print(f"   ├─ {fam['skipped_covered_points']} ambiguous points skipped "
                  "(already covered)")
        for r in fam["rules"]:
            print(f"   ├─ rule {r['rule_id']}: {r['status']} "
                  f"(battery {r['checked']} checks, pass_rate {r['pass_rate']})")
        for c in fam["cases"]:
            print(f"   ├─ case {c['case_id']} @ {c['params']} -> "
                  f"code {c['executed_code']} (severity {c['severity']})")
        for a in fam.get("amendments", []):
            print(f"   ├─ amendment: {a['action']} {a['rule_id']} — {a['reason']}")
    for rv in report["revalidation"]:
        print(f"[Revalidate] {rv['rule_id']}: {rv['checked']} checks, "
              f"pass_rate {rv['pass_rate']} "
              f"-> {'active' if rv['still_active'] else 'DEPRECATED'}")
    print(f"[Library]   after:  {report['library_after']}")
    print(f"Report: {report['report_path']}\n{div}")


def main():
    parser = argparse.ArgumentParser(
        prog="maca.factory",
        description="MACA offline law-discovery factory (LLM lives here)")
    parser.add_argument("--batch", type=int, default=1,
                        help="number of scenario families to design and sweep")
    parser.add_argument("--mock", action="store_true",
                        help="offline run with deterministic mock agents")
    parser.add_argument("--source", choices=["queue", "generate", "both"],
                        default="both")
    args = parser.parse_args()

    if not args.mock and not config.OPENAI_API_KEY:
        parser.error("OPENAI_API_KEY is not set (see .env.example); use --mock")

    report = asyncio.run(run_batch(args.batch, args.mock, args.source))
    _print(report)


if __name__ == "__main__":
    main()
