"""Query Analyzer: LLM-based task planning (unified).

This module provides a single source of truth for query understanding:
- plan_tasks: split input into 1-3 ordered tasks with selected tool or 'direct'
- Robust handling of pronouns (it/there) via standalone rewriting
- Weather task is kept even if location is unresolved (location=None)
- Calculator expression extracted when possible

Public helpers:
- plan_tasks(user_text, chat_history, available_tools) -> List[Task]
- is_multi_intent(query) -> bool
- get_subtasks(query) -> List[str] (standalone questions)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import json
import os
from langchain_groq import ChatGroq


# Tools normalized for downstream handling
_ALLOWED_TOOLS = {"rag", "calculator", "weather"}
_EXPR_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)")


class QueryAnalyzer:
    """LLM-based planner for intelligent task planning (single source of truth)."""
    
    def __init__(self):
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )
    
    # ------------------------- Planner API ----------------------------
    def plan_tasks(
        self,
        user_text: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Split input into ordered tasks with standalone questions.
        Output schema per task:
          { tool: one of available tools OR "direct", standalone: str,
            location?: str|null, expression?: str }
        Contract:
        - Always return 1..3 tasks in the order implied by the user's input.
        - If the user asks for weather but location cannot be inferred, still emit the task with location=null.
        - Rewrite each task as a standalone question (resolve pronouns using chat history or earlier tasks in this turn).
        - If tool=calculator and an expression is clear (like "24 + 43"), include it.
        - If no tool is needed, use tool="direct" to indicate answering directly without tools.
        - Return ONLY a JSON array, no commentary.
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return []
        
        # Build compact chat history text
        hist_lines: List[str] = []
        for m in (chat_history or []):
            if isinstance(m, dict):
                role = (m.get("role") or "").lower()
                content = m.get("content") or m.get("user_query") or m.get("question") or m.get("ai_response") or m.get("answer") or ""
                if role in ("human", "user"):
                    hist_lines.append(f"User: {content}")
                elif role in ("ai", "assistant"):
                    hist_lines.append(f"Assistant: {content}")
                else:
                    if m.get("user_query"):
                        hist_lines.append(f"User: {m.get('user_query')}")
                    if m.get("ai_response"):
                        hist_lines.append(f"Assistant: {m.get('ai_response')}")
        hist_text = "\n".join([h for h in hist_lines if h])

        tools = list(available_tools or ["vector_search", "weather", "calculator"])  # fallback
        tools_json = json.dumps(tools)
        prompt = (
            "You are a planner. Split the user's input into 1-3 ordered tasks. "
            "For each task, set 'tool' to one of the names in the Tool Catalog below if a tool is needed. "
            "If no tool is needed, set 'tool' to null.\n"
            f"Tool Catalog (JSON array of strings): {tools_json}\n"
            "Rewrite each task as a standalone question: resolve pronouns like 'it', 'there', 'that' using the chat history and "
            "earlier tasks in THIS SAME TURN. If resolution is impossible, keep the task but write the most specific standalone you can. "
            "If input contains multiple clauses (e.g., separated by 'and', commas, or question marks), create multiple tasks in order. "
            "If a weather-like question is present and location can be inferred, include 'location'. If it cannot be inferred, set 'location' to null. "
            "If calculator-like, include 'expression' like '24 + 43' if clear. "
            "Return ONLY a JSON array (no code block, no comments). Examples:\n"
            "[ {\"tool\": null, \"standalone\": \"Say hello\"} ]\n"
            "[ {\"tool\": \"vector_search\", \"standalone\": \"Where is GreenGrow Innovations headquartered?\"},"
            "  {\"tool\": \"weather\", \"standalone\": \"What is the weather there?\", \"location\": null},"
            "  {\"tool\": \"calculator\", \"standalone\": \"What is 234 + 43?\", \"expression\": \"234 + 43\"} ]\n\n"
            f"Chat History:\n{hist_text}\n\nUser Input: {user_text}\nJSON:"
        )
        try:
            resp = self._llm.invoke(prompt)
            txt = (getattr(resp, "content", str(resp)) or "").strip()
            # Extract JSON block
            start = txt.find("[")
            end = txt.rfind("]")
            if start != -1 and end != -1 and end > start:
                txt = txt[start:end+1]
            tasks = json.loads(txt)
            cleaned = self._post_process_tasks(tasks, user_text)
            if cleaned:
                return cleaned
        except Exception as e:
            # Log locally (optional) and fall back
            print(f"Planner failed: {e}")
        # Fallback: single direct task
        return [{"tool": "direct", "standalone": user_text}]

    # ------------------------- Helpers -------------------------------
    def _post_process_tasks(self, tasks: Any, user_text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(tasks, list):
            return out
        for t in tasks:
            if not isinstance(t, dict):
                continue
            tool = str(t.get("tool") or "").strip().lower()
            # map common synonyms
            if tool in ("calc", "calculation"):
                tool = "calculator"
            # Map MCP tool name to internal category expected by AgentManager
            if tool == "vector_search":
                tool = "rag"
            # Allow 'direct'
            if tool not in _ALLOWED_TOOLS:
                tool = "direct"
            standalone = (t.get("standalone") or t.get("question") or user_text or "").strip()
            # ensure a question has a trailing '?'
            if standalone and not standalone.endswith("?") and standalone[:1].isalpha():
                if re.search(r"\b(what|where|when|how|who|which|do|does|is|are|can|should)\b", standalone, re.I):
                    standalone += "?"
            item: Dict[str, Any] = {"tool": tool, "standalone": standalone}
            # optional fields
            if "location" in t:
                loc = t.get("location")
                item["location"] = loc if (isinstance(loc, str) and loc.strip()) else None
            # expression: keep if provided or try to extract
            expr = t.get("expression")
            if not expr:
                expr = self._extract_expression(standalone)
            if expr:
                item["expression"] = expr
            out.append(item)
        # Cap to 3 as contract
        return out[:3]

    def _extract_expression(self, text: str) -> Optional[str]:
        if not text:
            return None
        m = _EXPR_REGEX.search(text)
        if m:
            return f"{m.group(1)} {m.group(2)} {m.group(3)}"
        # lossy cleanup to try simple forms
        cleaned = re.sub(r"[^0-9+\-*/().\s]", "", text or "").strip()
        m2 = _EXPR_REGEX.search(cleaned)
        if m2:
            return f"{m2.group(1)} {m2.group(2)} {m2.group(3)}"
        return None

    # ------------------------- Convenience ---------------------------
    def is_multi_intent(self, query: str) -> bool:
        tasks = self.plan_tasks(query)
        return len(tasks) > 1
    
    def get_subtasks(self, query: str) -> List[str]:
        tasks = self.plan_tasks(query)
        return [(t.get("standalone") or t.get("question") or query).strip() for t in tasks if isinstance(t, dict)]


# Convenience factory

def create_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()
