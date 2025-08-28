from __future__ import annotations

from typing import Dict, Any, List
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
_model = genai.GenerativeModel('gemini-1.5-flash')


class ExpansionTool:
    name = "expansion"

    def looks_like(self, question: str) -> bool:  # optional, not used
        return False

    def _expand_query(self, original_query: str, chat_history: List[Dict[str, Any]] | None) -> List[str]:
        # Build lightweight history text for coreference resolution
        hist_lines: List[str] = []
        for m in (chat_history or []):
            role = m.get("role")
            content = m.get("content", "")
            if role == "human":
                hist_lines.append(f"User: {content}")
            elif role == "ai":
                hist_lines.append(f"Assistant: {content}")
        hist_text = "\n".join(hist_lines)

        prompt = f"""
You are a query planner.

Task:
- Rewrite the user's latest question into a minimal set of standalone search queries.
- Resolve pronouns using the chat history if present.
- If the question contains multiple distinct asks (e.g., founded year and headquarters), output one query per ask.
- If it is a single ask, output exactly one query.
- Keep each query concise and unambiguous (include the entity name explicitly).

Chat History (optional):
{hist_text}

Latest Question: {original_query}

Output rules:
- Write one query per line.
- Do not add numbering or extra text.
"""
        try:
            response = _model.generate_content(prompt)
            lines = [l.strip() for l in (response.text or "").split("\n") if l.strip()]
            return lines if lines else [original_query]
        except Exception as e:
            print(f"Expansion error: {e}")
            return [original_query]

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        chat_history = params.get("chat_history")
        queries = self._expand_query(query, chat_history)
        # Ensure original is included first if not already
        if query and (not queries or queries[0].strip().lower() != query.strip().lower()):
            if query not in queries:
                queries = [query] + queries
        # Deduplicate while preserving order
        seen = set()
        dedup: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                dedup.append(q)
        return {"queries": dedup}


