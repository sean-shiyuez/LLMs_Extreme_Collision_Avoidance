"""Episodic case memory on ChromaDB (direct client; langchain dependency dropped).

Each case stores the compact scenario text as the document plus structured
metadata: decision code, outcome, distilled lesson, and (optionally) the full
safety-approved contingency policy tree as JSON — which is what enables the
precomputed-policy fast path with confidence gating.

The legacy langchain collection in the same persist dir is left untouched
(its ada-002 embedding space is incompatible with text-embedding-3-small).
"""
import hashlib
import json
import math
import time
import uuid
from typing import Dict, List, Optional

import chromadb

from .. import config


class HashEmbedder:
    """Deterministic offline embedder for --mock runs (no API key needed)."""

    dim = 256

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbedder:
    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=config.EMBEDDING_MODEL, input=text)
        return resp.data[0].embedding


class CaseStore:
    def __init__(self, mock: bool = False):
        self._embedder = HashEmbedder() if mock else OpenAIEmbedder()
        self._client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        # Mock runs use a separate collection: hash embeddings must never mix
        # with the real text-embedding-3-small space.
        name = config.CASE_COLLECTION + ("_mock" if mock else "")
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def store_case(
        self,
        scenario_text: str,
        decision_code: int,
        outcome: str,
        lesson: str,
        policy_tree: Optional[dict] = None,
        scenario_name: str = "",
    ) -> str:
        case_id = f"{scenario_name or 'case'}-{uuid.uuid4().hex[:8]}"
        metadata = {
            "decision_code": str(decision_code),
            "outcome": outcome,
            "lesson": lesson,
            "scenario_name": scenario_name,
            "stored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if policy_tree is not None:
            metadata["policy_tree"] = json.dumps(policy_tree)
        self._collection.add(
            ids=[case_id],
            embeddings=[self._embedder.embed(scenario_text)],
            documents=[scenario_text],
            metadatas=[metadata],
        )
        return case_id

    def retrieve(self, query_text: str, k: int = config.NUM_SIMILAR_CASES) -> List[Dict]:
        if self._collection.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[self._embedder.embed(query_text)],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        cases = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            cases.append({"document": doc, "metadata": meta, "distance": dist})
        return cases

    def retrieve_precomputed_policy(self, query_text: str) -> Optional[Dict]:
        """Confidence-gated policy reuse: return the closest stored case that
        carries a policy tree, if it clears the distance gate."""
        for case in self.retrieve(query_text, k=3):
            if "policy_tree" in case["metadata"] and \
                    case["distance"] <= config.PRECOMPUTED_POLICY_MAX_DISTANCE:
                case = dict(case)
                case["policy_tree"] = json.loads(case["metadata"]["policy_tree"])
                return case
        return None
