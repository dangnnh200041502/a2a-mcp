from __future__ import annotations

from typing import Any, Dict, List
import os

from langchain_groq import ChatGroq
from tools.vector_search import VectorSearchTool


class AgentPrivate:
    """Agent điều phối pipeline Private Search qua function-calling tools: expansion → retrieve → fusion → rerank."""

    def __init__(self) -> None:
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )
        # Single composed tool (no MCP)
        self._vector_search = VectorSearchTool()

    def _contextualize(self, original_query: str, chat_history: List[Dict[str, Any]] | None) -> str:
        """Rewrite query into standalone form using brief chat history (ported from langchain_utils)."""
        if not chat_history:
            return original_query
        # Flatten history
        hist_lines: List[str] = []
        for m in chat_history:
            role = m.get("role")
            content = m.get("content", "")
            if role == "human":
                hist_lines.append(f"User: {content}")
            elif role == "ai":
                hist_lines.append(f"Assistant: {content}")
        hist_text = "\n".join(hist_lines)
        prompt = (
            "Rewrite the user's last message into a standalone question that can be understood without prior context. "
            "Resolve pronouns and references using the chat history. If rewriting fails, return the original.\n\n"
            f"Chat History:\n{hist_text}\n\nOriginal: {original_query}\nStandalone:"
        )
        try:
            resp = self._llm.invoke(prompt)
            text = (getattr(resp, "content", None) or str(resp)).strip()
            return text or original_query
        except Exception:
            return original_query

    def search(self, question: str, chat_history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        # 0) Contextualize (optional, improves coreference handling)
        question = self._contextualize(question, chat_history)

        # Unified vector search pipeline
        result = self._vector_search.invoke({
            "query": question,
            "chat_history": chat_history,
            "threshold": 0.5,
            "top_k": 10,
        })

        meta: Dict[str, Any] = result.get("meta", {})
        return {
            "original_query": question,
            "effective_query": result.get("effective_query", question),
            "expanded_queries": meta.get("expanded_queries", []),
            "final_documents": [
                {
                    "content": d.get("content", ""),
                    "score": d.get("score", 0.0),
                    "metadata": d.get("metadata", {}),
                }
                for d in result.get("final_documents", [])
            ],
            "meta": meta,
            "retrieved_docs_count": meta.get("retrieved_docs_count", 0),
            "fused_docs_count": meta.get("fused_docs_count", 0),
            "final_docs_count": meta.get("final_docs_count", 0),
        }

