from __future__ import annotations

from typing import Dict, Any, List
from sentence_transformers import CrossEncoder
import torch


class RerankTool:
    name = "rerank"

    def looks_like(self, question: str) -> bool:
        return False

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query: str = params.get("query", "")
        docs: List[dict] = params.get("documents", [])
        threshold: float = float(params.get("threshold", 0.5))
        top_k: int = int(params.get("top_k", 10))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        except Exception:
            model = None

        if model is None or not docs:
            return {"documents": docs[:top_k]}

        pairs = [(query, d.get("content", "")) for d in docs]
        scores = model.predict(pairs)
        for d, s in zip(docs, scores):
            d["score"] = float(s)
        docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        reranked = [d for d in docs if d.get("score", 0.0) >= threshold] or docs[:top_k]
        return {"documents": reranked[:top_k]}


