"""PrivateDataAgent: handles private KB operations (RAG retrieval, etc.).

Currently exposes one capability: run_rag_retrieval(question, chat_history) →
returns reranked documents and meta from the RAG tool flow. In the future, we
can add more private tools here (e.g., internal SQL, CRM, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_utils import get_advanced_rag_pipeline


class PrivateDataAgent:
    def __init__(self):
        self._pipeline = get_advanced_rag_pipeline()

    def run_rag_retrieval(self, question: str, chat_history: List[Dict[str, Any]] | None = None,
                          retrieval_k: int = 5, rrf_k: float = 60.0,
                          rerank_top_k: int = 10, rerank_threshold: float = 0.5) -> Dict[str, Any]:
        # 1. Multi-query retrieval
        query_results = self._pipeline.multi_query_retrieval(question, chat_history)
        # 2. RRF fusion
        fused_docs = self._pipeline.apply_rrf(query_results)
        # 3. Cross-Encoder rerank
        effective_query = getattr(self._pipeline, "_effective_query", None) or question
        reranked_docs = self._pipeline.apply_reranking(effective_query, fused_docs)
        # meta
        meta = {
            "expanded_queries": [qr[0].source_query for qr in query_results if qr],
            "retrieved_docs_count": sum(len(qr) for qr in query_results),
            "fused_docs_count": len(fused_docs),
            "final_docs_count": len(reranked_docs),
        }
        final_documents = [
            {
                "content": getattr(d, "content", ""),
                "score": getattr(d, "score", 0.0),
                "metadata": getattr(d, "metadata", {}),
            }
            for d in reranked_docs
        ]
        return {
            "effective_query": effective_query,
            "meta": meta,
            "final_documents": final_documents,
        }


