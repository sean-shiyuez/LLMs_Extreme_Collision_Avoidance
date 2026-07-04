"""Three-timescale orchestrator: Reflex — Deliberation — Consolidation.

Reflex loop (ms, LLM-free): HJ+TTC scene assessment; below RISK_LOW nothing
happens, in the elevated band the deliberation loop is woken to deliberate
*ahead of* the emergency, at/above RISK_TRIGGER the armed contingency policy
is matched and executed by pure kinematics.

Deliberation loop (s, deadline-aware): perception ∥ risk ∥ memory fan-out,
tool-augmented decision producing a contingency policy tree, consistency-gated
debate, and the HJ safety shield with a veto → re-decide loop.

Consolidation loop (offline): reflection, episodic memory write-back, and
fine-tuning sample export (slow system teaches the fast one).
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .agents.debate import DebatePanel, debate_needed
from .agents.decision import DecisionAgent
from .agents.evaluation import EvaluationAgent
from .agents.memory import MemoryAgent
from .agents.perception import PerceptionAgent
from .agents.risk import RiskAgent
from .agents.safety import SafetyAgent
from .budget import CognitionBudget, SceneAssessment, assess_scene, budget_from
from .llm import MockLLM, OpenAIClient
from .memory.case_store import CaseStore
from .perception.bev import render_bev
from .policy_cache import ArmedPolicy, PolicyBranch, PolicyCache
from .scenario.loader import load_scenario
from .scenario.schema import Scenario, Snapshot
from .tools.physics import project_snapshot
from .tools.registry import build_registry


@dataclass
class Blackboard:
    scenario: Scenario
    snapshot: Snapshot                      # deliberation (preview) snapshot
    assessment: SceneAssessment
    budget: CognitionBudget
    case_store: CaseStore
    exec_snapshot: Snapshot = None          # scene projected to the anticipated
    exec_assessment: SceneAssessment = None  # execution (trigger) state
    registry: object = None
    bev_b64: Optional[str] = None
    perception_report: dict = field(default_factory=dict)
    risk_report: dict = field(default_factory=dict)
    memory_report: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    safety_verdicts: List[dict] = field(default_factory=list)
    debate_report: dict = field(default_factory=dict)
    evaluation_report: dict = field(default_factory=dict)
    trace: List[dict] = field(default_factory=list)


class Orchestrator:
    def __init__(self, mock: bool = False, debate_mode: str = "auto",
                 no_vision: bool = False, enforce_deadline: bool = False):
        self.mock = mock
        self.debate_mode = debate_mode
        self.no_vision = no_vision
        self.enforce_deadline = enforce_deadline
        self.llm = MockLLM() if mock else OpenAIClient()
        self.case_store = CaseStore(mock=mock)
        self.cache = PolicyCache()

    # ------------------------------------------------------------------
    async def run_scenario(self, name: str) -> dict:
        scenario = load_scenario(name)
        preview, trigger = scenario.preview, scenario.trigger
        record = {"scenario": name, "description": scenario.description,
                  "mock": self.mock, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "phases": {}}

        # ---- Reflex @ preview -------------------------------------------------
        t0 = time.perf_counter()
        assessment = assess_scene(preview)
        reflex_preview_ms = (time.perf_counter() - t0) * 1000
        budget = budget_from(assessment)
        record["phases"]["reflex_preview"] = {
            "elapsed_ms": round(reflex_preview_ms, 2),
            "scene_risk": assessment.scene_risk, "band": assessment.band(),
            "min_ttc_s": assessment.min_ttc, "primary_threat": assessment.primary_threat,
            "per_target": assessment.per_target,
            "budget": vars(budget),
        }

        bb = Blackboard(scenario=scenario, snapshot=preview, assessment=assessment,
                        budget=budget, case_store=self.case_store)

        # ---- Deliberation (anticipatory, off the critical path) ---------------
        deliberation_s = None
        if assessment.band() != "low" and budget.tier != "reflex_only":
            t0 = time.perf_counter()
            try:
                if self.enforce_deadline:
                    await asyncio.wait_for(self._deliberate(bb),
                                           timeout=budget.deadline_s)
                else:
                    await self._deliberate(bb)
            except asyncio.TimeoutError:
                reason = ("TTC deadline exceeded" if self.enforce_deadline
                          else "wall-clock LLM timeout")
                bb.policy = {"branches": [
                    {"condition_type": "default", "condition": reason,
                     "code": 0, "target_speed_mps": 0.0,
                     "rationale": f"Deliberation aborted ({reason}); anytime fallback "
                                  "to the minimal-risk maneuver.", "confidence": 1.0}],
                    "overall_rationale": "anytime fallback"}
                bb.trace.append({"phase": "deliberation", "agent": "orchestrator",
                                 "output": f"{reason} -> minimal-risk policy"})
            deliberation_s = time.perf_counter() - t0
            source = bb.policy.pop("_source", "deliberation")
            self.cache.arm(ArmedPolicy(
                scenario_name=name,
                branches=[PolicyBranch.from_dict(b) for b in bb.policy.get("branches", [])],
                source=source,
                primary_threat=bb.risk_report.get("primary_threat")
                or assessment.primary_threat,
                deliberation_s=deliberation_s,
            ))
            record["phases"]["deliberation"] = {
                "elapsed_s": round(deliberation_s, 3),
                "deadline_s": round(budget.deadline_s, 2),
                "deadline_met": deliberation_s <= budget.deadline_s,
                "armed_policy": self.cache.armed.to_dict(),
                "safety_rounds": len(bb.safety_verdicts),
                "debate_ran": bool(bb.debate_report),
            }
        else:
            record["phases"]["deliberation"] = {
                "skipped": True,
                "reason": "low risk" if assessment.band() == "low"
                else "insufficient time window (reflex_only tier)"}

        # ---- Reflex @ trigger (the real-time critical path) --------------------
        t0 = time.perf_counter()
        trigger_assessment = assess_scene(trigger)
        assess_ms = (time.perf_counter() - t0) * 1000
        executed_branch = None
        if trigger_assessment.band() == "critical":
            branch, observed, match_ms = self.cache.match(preview, trigger)
            if branch is not None:
                executed_code, executed_branch = branch.code, vars(branch)
                decision_source = f"armed_policy[{observed}]"
            else:
                executed_code = 0
                decision_source = f"reflex_fallback({observed})"
                match_ms = 0.0
        else:
            executed_code = 7
            observed, match_ms = trigger_assessment.band(), 0.0
            decision_source = "reflex_no_intervention"
        trigger_latency_ms = assess_ms + match_ms
        record["phases"]["reflex_trigger"] = {
            "scene_risk": trigger_assessment.scene_risk,
            "band": trigger_assessment.band(),
            "assess_ms": round(assess_ms, 2),
            "match_ms": round(match_ms, 2),
            "trigger_decision_latency_ms": round(trigger_latency_ms, 2),
            "executed_code": executed_code,
            "maneuver": config.DECISION_CODES.get(executed_code, "unknown"),
            "decision_source": decision_source,
            "matched_branch": executed_branch,
        }

        # ---- Execution + Consolidation ----------------------------------------
        outcome = scenario.outcome_for(executed_code)
        record["execution_outcome"] = outcome
        if self.cache.armed is not None:
            evaluation = EvaluationAgent(self.llm)
            memory = MemoryAgent(self.llm)
            report = await evaluation.run(bb, executed_code, executed_branch, outcome)
            await memory.consolidate(bb, executed_code, outcome, report["lesson"])
            evaluation.export_distillation_sample(bb, executed_code)
            record["phases"]["consolidation"] = report

        record["trace"] = bb.trace
        record["latency_summary"] = {
            "reflex_preview_ms": round(reflex_preview_ms, 2),
            "deliberation_s": None if deliberation_s is None else round(deliberation_s, 3),
            "trigger_decision_latency_ms": round(trigger_latency_ms, 2),
            "llm_calls_on_trigger_path": 0,
        }
        self._save(record)
        return record

    # ------------------------------------------------------------------
    async def _deliberate(self, bb: Blackboard):
        # Project the scene to the anticipated execution state: the armed
        # policy fires at the trigger instant (TTC ~= T_EXEC_TTC), so every
        # maneuver is validated against that geometry, not today's.
        gap = bb.scenario.trigger.t - bb.snapshot.t
        t_fwd = 0.0
        if bb.assessment.min_ttc is not None:
            t_fwd = max(0.0, min(bb.assessment.min_ttc - config.T_EXEC_TTC, gap))
        bb.exec_snapshot = project_snapshot(bb.snapshot, t_fwd)
        bb.exec_assessment = assess_scene(bb.exec_snapshot)
        bb.trace.append({"phase": "deliberation", "agent": "orchestrator",
                         "output": f"scene projected {t_fwd:.2f}s forward to the "
                                   f"anticipated execution state "
                                   f"(risk {bb.exec_assessment.scene_risk})"})

        if bb.budget.allow_vision and not self.no_vision:
            path = config.RUNS_DIR / f"{bb.scenario.name}_bev.png"
            config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            bb.bev_b64 = await asyncio.to_thread(render_bev, bb.snapshot, str(path))
        bb.registry = build_registry(bb.exec_snapshot, bb.case_store)

        # parallel fan-out: semantic, numeric and episodic channels
        await asyncio.gather(
            PerceptionAgent(self.llm).run(bb),
            RiskAgent(self.llm).run(bb),
            MemoryAgent(self.llm).run(bb),
        )

        decision = DecisionAgent(self.llm)
        safety = SafetyAgent(self.llm)
        precomputed = bb.memory_report.get("precomputed_policy")

        feedback, verdict = "", None
        for round_idx in range(1, config.SAFETY_MAX_ROUNDS + 1):
            if round_idx == 1 and precomputed:
                bb.policy = dict(precomputed["policy_tree"])
                bb.policy["_source"] = "precomputed"
                bb.trace.append({"phase": "deliberation", "agent": "memory",
                                 "output": f"precomputed policy reused "
                                           f"(distance {precomputed['distance']})"})
            else:
                await decision.run(bb, veto_feedback=feedback, round_idx=round_idx)
            if round_idx == 1 and debate_needed(bb, self.debate_mode):
                await DebatePanel(self.llm).run(bb)
            verdict = await safety.run(bb, round_idx=round_idx)
            if verdict["approved"]:
                break
            feedback = verdict["veto_reason"]
        if verdict is not None and not verdict["approved"]:
            det_failed = any(not v["approved"] for v in verdict["branch_verdicts"])
            if det_failed:
                SafetyAgent.harden_policy(bb, verdict)
                bb.trace.append({"phase": "deliberation", "agent": "safety",
                                 "output": "policy hardened: deterministically failing "
                                           "branches replaced with code 0 "
                                           "(minimal-risk maneuver)"})
            else:
                # The physics shield passed every branch; only the LLM
                # reviewer's judgement call persisted. Physics evidence wins
                # after max rounds — arm the deterministically-safe policy.
                bb.trace.append({"phase": "deliberation", "agent": "safety",
                                 "output": "LLM reviewer objection persisted after "
                                           "max rounds, but the deterministic shield "
                                           "passed all branches; arming the "
                                           "physics-approved policy"})

    # ------------------------------------------------------------------
    def _save(self, record: dict):
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = config.RUNS_DIR / f"{record['scenario']}-{stamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1, default=str)
        record["log_path"] = str(path)
