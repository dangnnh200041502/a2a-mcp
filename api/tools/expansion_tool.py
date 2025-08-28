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

    def _expand_query(self, original_query: str) -> List[str]:
        prompt = f"""
        Generate exactly 3 semantically similar search queries (one per line) for: "{original_query}"
        Keep meaning, vary phrasing.
        """
        try:
            response = _model.generate_content(prompt)
            lines = [l.strip() for l in (response.text or "").split("\n") if l.strip()]
            if len(lines) >= 3:
                return lines[:3]
            while len(lines) < 3:
                lines.append(original_query)
            return lines
        except Exception as e:
            print(f"Expansion error: {e}")
            return [original_query] * 3

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        expanded = self._expand_query(query)
        return {"queries": [query] + expanded}


