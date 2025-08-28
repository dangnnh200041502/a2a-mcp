from __future__ import annotations

from typing import Any, Dict, List
import os

from mcp.types import MCPClient
from pinecone_utils import vectorstore
from langchain_groq import ChatGroq


class AgentPrivate:
    """Agent điều phối pipeline Private Search qua MCP tools: expansion → retrieve → fusion → rerank."""

    def __init__(self, client: MCPClient | None = None) -> None:
        if client is None:
            raise ValueError("AgentPrivate requires an MCPClient instance (provided by provider).")
        self._mcp = client
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )

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

        # 1) Expansion
        exp = self._mcp.call("expansion", {"query": question})
        queries: List[str] = exp.get("queries", [question])

        # 2) Retrieve for each query
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        all_results: List[List[Dict[str, Any]]] = []
        for q in queries:
            try:
                docs = retriever.invoke(q)
            except Exception:
                docs = retriever.get_relevant_documents(q)
            # Chuẩn hóa sang DocumentScore-like dicts
            rows: List[Dict[str, Any]] = []
            for i, doc in enumerate(docs):
                content = getattr(doc, 'page_content', getattr(doc, 'content', str(doc))) or ""
                rows.append({
                    "content": content,
                    "metadata": getattr(doc, 'metadata', {}),
                    "score": 1.0 - (i * 0.1),
                    "source_query": q,
                    "rank": i,
                })
            all_results.append(rows)

        # 3) Fusion (RRF)
        fused = self._mcp.call("fusion", {"queries": queries, "retrieved": all_results}).get("fused", [])

        # 4) Rerank
        reranked = self._mcp.call("rerank", {"query": question, "documents": fused, "threshold": 0.5, "top_k": 10}).get("documents", [])

        return {
            "original_query": question,
            "expanded_queries": queries[1:] if len(queries) > 1 else [],
            "final_documents": [
                {
                    "content": d.get("content", ""),
                    "score": d.get("score", 0.0),
                    "metadata": d.get("metadata", {}),
                }
                for d in reranked
            ],
            "retrieved_docs_count": sum(len(r) for r in all_results),
            "fused_docs_count": len(fused),
            "final_docs_count": len(reranked),
        }


