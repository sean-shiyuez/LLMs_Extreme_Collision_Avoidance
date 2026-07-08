"""库成长日志 —— 把"生成案例、判断、完善规则"的每一个事件按时间顺序
追加到同一个人类可读文件（config.GROWTH_LOG_PATH）。

与 Results/factory/batch-*.json（机器可读的完整批次报告）互补：
本日志面向人工审阅，一行一个事件，可直接 tail -f 观察工厂工作过程。
事件类型：批次开始/结束、队列项处理/跳过、案例生成（含判断与教训）、
案例退役、场景族扫描、规则提案接受/被拒、规则晋升、规则修订、抗漂移复验。
"""
import json
import time

from .. import config


_LEGEND = """\
# ======================================================================
# MACA 库成长日志 (library_growth.log)
# 记录离线工厂"生成案例 / 判断 / 完善规则"的每一步，供人工审阅。
# 与 batch-*.json（机器可读完整报告）互补。可 `tail -f` 实时观察。
#
# 事件标签含义：
#   [批次开始]/[批次结束]  一次工厂批次的起止 + 库现状统计
#   [场景族]              生成器设计的参数化场景族被网格扫描；给出
#                         规律/模糊/绝境点数 + 规则反例数 + 假设
#     · 规律点：某机动明显最优 → 用于归纳规则
#     · 模糊点：严重度接近/涉行人权衡 → 交委员会深思成案例
#     · 绝境点：所有机动都最严重、无区分 → 丢弃(学不到规律)
#     · 规则反例：现有规则命中但非最优的点 → 回喂归纳器收紧规则
#   [队列处理]/[队列跳过]  运行时反馈(gap/冲突)：未覆盖→深思；已覆盖或
#                         绝境(所有机动均最严重、无区分度)→跳过，不建无意义案例
#   [案例生成]            新案例三行：① 决策机动码 ② 仿真判断(严重度+结局)
#                         ③ 评估(是否有效 + 可迁移教训)
#   [案例退役]            案例被冲突裁决否决，退出在线索引(保留追溯)
#   [规则完善]            候选规则过晋升电池：检查数/通过率 → active 或 candidate
#                         (通过率<0.9 留 candidate，附失败样本 = 数值复核在起作用)
#   [提案被拒]            归纳提案未过网格数值复核(幻觉边界被拦下)，附具体不符点
#   [归纳跳过]            区域/格点已被库覆盖 → 不花 LLM
#   [规则修订]            既有规则被收紧/废弃(evidence-driven)
#   [规则复验]            批末抗漂移：现役规则重新验证，失效即废弃
# ======================================================================
"""


def _write(line: str):
    config.GROWTH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 首次写入(文件不存在或为空)先落一份图例，使日志自解释
    if not config.GROWTH_LOG_PATH.exists() or config.GROWTH_LOG_PATH.stat().st_size == 0:
        with open(config.GROWTH_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(_LEGEND)
    with open(config.GROWTH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")


def batch_start(mock: bool, source: str, stats: dict):
    _write("=" * 70)
    _write(f"[批次开始] mock={mock} source={source} | 库现状: {json.dumps(stats, ensure_ascii=False)}")


def batch_end(stats: dict, report_path: str):
    _write(f"[批次结束] 库现状: {json.dumps(stats, ensure_ascii=False)} | 报告: {report_path}")


def queue_skipped(kind: str, scenario: str, covered_by: str):
    _write(f"[队列跳过] {kind} @ {scenario} —— 库已正确覆盖（{covered_by}），不消耗 LLM")


def queue_processed(kind: str, scenario: str):
    _write(f"[队列处理] {kind} @ {scenario} —— 未覆盖，交委员会深思")


def queue_doomed(kind: str, scenario: str):
    _write(f"[队列跳过] {kind} @ {scenario} —— 绝境场景（所有机动均最严重、"
           f"无区分度），学不到决策规律，丢弃不建案例")


def case_added(case: dict, rec: dict):
    """记录一条新案例：执行机动、仿真判断、有效性与教训。"""
    code = rec["executed_code"]
    _write(f"[案例生成] {case['case_id']} @ {rec['scenario_name']} 来源={rec['source']}")
    _write(f"           决策: code {code}（{config.DECISION_CODES.get(code, '?')}）")
    _write(f"           判断: 仿真结局 severity {rec['severity']}/4 —— {rec['outcome']}")
    _write(f"           评估: effective={rec['effective']} | 教训: {rec['lesson']}")


def case_retired(case_id: str, reason: str):
    _write(f"[案例退役] {case_id} —— {reason}")


def family_swept(fam_report: dict):
    _write(f"[场景族] {fam_report['family_id']}（{fam_report['pattern']}）"
           f"网格 {fam_report['points']} 点：规律 {fam_report['lawful']}"
           f"（未覆盖 {fam_report['lawful_uncovered']}）/ 模糊 {fam_report['ambiguous']}"
           + (f" / 绝境 {fam_report['doomed']}" if fam_report.get("doomed") else "")
           + (f" / 规则反例 {fam_report['rule_counterexamples']}"
              if fam_report.get("rule_counterexamples") else ""))
    _write(f"           假设: {fam_report.get('hypothesis', '')}")


def consolidation_skipped(reason: str):
    _write(f"[归纳跳过] {reason}")


def proposal_rejected(guards: dict, problems: list):
    _write(f"[提案被拒] 守卫={json.dumps(guards, ensure_ascii=False)}")
    _write(f"           网格复核问题: {problems[0] if problems else '?'}")


def rule_validated(rule_id: str, maintains_code, verdict: dict, status: str):
    _write(f"[规则完善] {rule_id} maintains=code {maintains_code} —— "
           f"晋升电池 {verdict['checked']} 项检查, 通过率 {verdict['pass_rate']}"
           f" → {status}"
           + (f" | 失败样本: {verdict['failures'][0]}" if verdict.get("failures") else ""))


def amendment(action: str, rule_id: str, reason: str):
    _write(f"[规则修订] {action} {rule_id} —— {reason}")


def revalidation(rv: dict):
    _write(f"[规则复验] {rv['rule_id']}: {rv['checked']} 项检查, 通过率 "
           f"{rv['pass_rate']} → {'仍在役' if rv['still_active'] else '废弃'}")
