"""FinalAgent: synthesizes final answers from tool outputs.

Input structure example:
{
  "subtasks": [
    {"type": "rag", "question": str, "effective_query": str, "final_documents": [...], "meta": {...}},
    {"type": "calculator", "question": str, "result": 2}
  ],
  "original_question": str,
  "chat_history": [...]
}
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
from langchain_groq import ChatGroq


class FinalAgent:
    def __init__(self):
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )

    def _format_context(self, subtasks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for idx, st in enumerate(subtasks, start=1):
            if st.get("type") == "rag":
                docs = st.get("final_documents", [])
                doc_snippets = []
                for d in docs[:10]:
                    content = (d.get("content") or "")
                    if len(content) > 400:
                        content = content[:400] + "..."
                    doc_snippets.append(f"- {content}")
                parts.append(
                    f"Question: {st.get('question')}\nRewritten: {st.get('effective_query')}\nEvidence:\n" +
                    ("\n".join(doc_snippets) if doc_snippets else "(no docs)")
                )
            elif st.get("type") == "calculator":
                parts.append(
                    f"Computation: {st.get('question')}\nResult: {st.get('result')}"
                )
            elif st.get("type") == "weather":
                parts.append(
                    f"Weather query: {st.get('question')}\nWeather: {st.get('status')}"
                )
        return "\n\n".join(parts)

    def generate(self, original_question: str, subtasks: List[Dict[str, Any]], chat_history: List[Dict[str, Any]] | None = None) -> str:
        context = self._format_context(subtasks)
        if chat_history:
            hist_lines = []
            for m in chat_history:
                role = m.get("role")
                content = m.get("content", "")
                if role == "human":
                    hist_lines.append(f"User: {content}")
                elif role == "ai":
                    hist_lines.append(f"Assistant: {content}")
            hist_text = "\n".join(hist_lines)
            prompt = (
                "You are a helpful assistant. You are given tool outputs (retrieved evidence and/or calculator results). "
                "Use ONLY this information to answer. If multiple questions appear, answer each clearly in order. "
                "Do NOT mention tools, internal steps, or words like 'subtask'/'RAG'. Keep the answer concise.\n\n"
                f"Original Question: {original_question}\n\nEvidence and computations:\n{context}\n\nChat History:\n{hist_text}\n\nFinal Answer (no tool labels):"
            )
        else:
            prompt = (
                "You are a helpful assistant. You are given tool outputs (retrieved evidence and/or calculator results). "
                "Use ONLY this information to answer. If multiple questions appear, answer each clearly in order. "
                "Do NOT mention tools, internal steps, or words like 'subtask'/'RAG'. Keep the answer concise.\n\n"
                f"Original Question: {original_question}\n\nEvidence and computations:\n{context}\n\nFinal Answer (no tool labels):"
            )
        resp = self._llm.invoke(prompt)
        return getattr(resp, "content", str(resp))


