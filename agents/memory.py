"""MemoryAgent — episodic retrieval and post-episode consolidation.

Retrieval is deterministic (no LLM call, keeps the parallel fan-out cheap):
similar past cases with their lessons, plus a confidence-gated precomputed
policy candidate if one clears the similarity gate.
"""
import asyncio
import time

from .base import Agent


class MemoryAgent(Agent):
    name = "memory"
    role = "evaluation"  # unused for retrieval; writes use no LLM either

    async def run(self, bb):
        start = time.perf_counter()
        query = bb.snapshot.compact_text()
        cases = await asyncio.to_thread(bb.case_store.retrieve, query)
        precomputed = await asyncio.to_thread(
            bb.case_store.retrieve_precomputed_policy, query)
        bb.memory_report = {
            "cases": [
                {"scenario": c["document"],
                 "decision_code": c["metadata"].get("decision_code"),
                 "outcome": c["metadata"].get("outcome"),
                 "lesson": c["metadata"].get("lesson"),
                 "distance": round(c["distance"], 4)}
                for c in cases
            ],
            "precomputed_policy": precomputed and {
                "distance": round(precomputed["distance"], 4),
                "policy_tree": precomputed["policy_tree"],
                "scenario_name": precomputed["metadata"].get("scenario_name"),
            },
        }
        self.record(bb, "deliberation", bb.memory_report,
                    extra={"elapsed_s": round(time.perf_counter() - start, 3)})

    async def consolidate(self, bb, executed_code: int, outcome: str, lesson: str):
        case_id = await asyncio.to_thread(
            bb.case_store.store_case,
            bb.snapshot.compact_text(),
            executed_code,
            outcome,
            lesson,
            bb.policy,
            bb.scenario.name,
        )
        self.record(bb, "consolidation", {"stored_case_id": case_id})
        return case_id
