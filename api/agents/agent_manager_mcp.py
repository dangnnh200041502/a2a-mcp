"""Agent Manager: decide and execute tools for Agentic RAG using MCP.

Tools:
- rag_retrieval: Multi-Query → RRF → Rerank (returns final reranked documents)
- calculator: Evaluate simple arithmetic expressions when user asks to compute
- weather: Get weather information (toy implementation)

Usage (khuyến nghị):
    manager = AgentManager()
    res = await manager.answer(user_query, chat_history)
    print(res["answer"])
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
import os

from langchain_groq import ChatGroq
from query_analyzer import QueryAnalyzer
from mcp_client import MCPClient


class AgentManager:
    """Selects and runs tools for queries (multi-intent aware) using MCP."""

    def __init__(self):
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )
        # MCP client for tool execution
        self._mcp_client = MCPClient()
        self._query_analyzer = QueryAnalyzer()

    async def initialize_mcp(self):
        """Initialize MCP connection if not already connected."""
        if not self._mcp_client.session:
            await self._mcp_client.connect_to_server()

    # ---------- Query analysis -----------------------------------------

    def _analyze_query_with_llm(self, raw: str) -> List[str]:
        """Derive standalone questions from planner tasks (compat helper)."""
        try:
            tasks = self._query_analyzer.plan_tasks(raw, chat_history=None)
            if isinstance(tasks, list) and tasks:
                return [
                    (t.get("standalone") or t.get("question") or raw).strip()
                    for t in tasks
                    if isinstance(t, dict)
                ]
        except Exception:
            pass
        # Fallback: tách thô theo '?' và 'and'
        return [s.strip() for s in re.split(r"[?]|(?:\s+and\s+)", raw) if s.strip()]

    # ---------- Intent helpers -----------------------------------------

    def _looks_like_greeting(self, question: str) -> bool:
        text = (question or "").strip().lower()
        greeting_pattern = (
            r"\b(?:hello|hi|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|"
            r"xin\s+chào|chào|chao|chào\s+bạn|chào\s+anh|chào\s+chị|yo|sup)\b"
        )
        if len(text) <= 40 and re.search(greeting_pattern, text):
            return True
        return text in ("hello", "hi", "hey", "xin chào", "chào")

    def _classify_intent_llm(self, question: str) -> str:
        """Classify which tool to use: weather | calculator | rag.
        Returns one of: 'weather', 'calculator', 'rag'.
        """
        prompt = (
            "You are a router. Classify the user's question into exactly one tool: "
            "weather | calculator | rag.\n"
            "- weather: questions asking about forecast, temperature, rain, etc.\n"
            "- calculator: contains a simple arithmetic expression to compute (+,-,*,/).\n"
            "- rag: general knowledge/information retrieval.\n\n"
            f"Question: {question}\n"
            "Answer with only one word: weather or calculator or rag."
        )
        try:
            resp = self._llm.invoke(prompt)
            label = (getattr(resp, "content", str(resp)) or "").strip().lower()
            if "weather" in label:
                return "weather"
            if "calculator" in label or label == "calc":
                return "calculator"
            return "rag"
        except Exception:
            return "rag"

    def generate_direct_answer(
        self,
        user_text: str,
        chat_history: List[Dict[str, Any]] | None = None
    ) -> str:
        """Generate a short, natural reply directly (no tools). Used for greetings/small talk."""
        if chat_history:
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
                "You are a friendly, concise assistant. Respond naturally to the user's greeting or small talk.\n\n"
                f"Chat History:\n{hist_text}\n\nUser: {user_text}\nAssistant:"
            )
        else:
            prompt = (
                "You are a friendly, concise assistant. Respond naturally to the user's greeting or small talk.\n\n"
                f"User: {user_text}\nAssistant:"
            )
        resp = self._llm.invoke(prompt)
        return getattr(resp, "content", str(resp))

    async def _detect_calculator_intent(self, question: str) -> bool:
        await self.initialize_mcp()
        return await self._mcp_client.detect_tool_intent(question, "calculator")

    async def _extract_calculation_expression(self, question: str) -> str:
        # Local lightweight extractor (no MCP dependency)
        import re as _re
        text = _re.sub(r'^\s*\d+[\.)]\s+', '', question)
        text = _re.sub(r'^\s*[\-–]\s+', '', text)
        expr_match = _re.search(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)', text)
        if expr_match:
            return f"{expr_match.group(1)} {expr_match.group(2)} {expr_match.group(3)}"
        candidate = _re.sub(r'[^0-9\+\-\*/\(\)\.\s]', '', text).strip()
        return candidate

    # ---------- Tool wrappers ------------------------------------------

    async def use_weather(self, question: str, location: str | None = None) -> Dict[str, Any]:
        await self.initialize_mcp()
        payload = {"question": question}
        if location:
            payload["location"] = location
        return await self._mcp_client.call_tool("weather", payload)

    async def use_calculator(self, expression: str) -> Dict[str, Any]:
        await self.initialize_mcp()
        return await self._mcp_client.call_tool("calculator", {"expression": expression})

    async def use_vector_search(
        self, query: str, chat_history: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        await self.initialize_mcp()
        # Giữ nguyên khóa payload như bạn đang dùng (query / threshold / top_k)
        return await self._mcp_client.call_tool(
            "vector_search",
            {
                "query": query,
                "chat_history": chat_history,
                "top_k": 2,
            },
        )

    # ---------- Decider (single vs multi) ------------------------------

    async def decide_tool(
        self, question: str, chat_history: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """Decide routing using QueryAnalyzer.plan_tasks for robust pronoun/location handling.
        Handle greetings as direct LLM responses (no tools).
        """
        # Direct greetings -> no tools
        if self._looks_like_greeting(question):
            return {"tool": "direct", "reason": "greeting/small talk"}

        await self.initialize_mcp()
        available_tools = [t.get("name") for t in (self._mcp_client.tools or [])] or ["vector_search", "weather", "calculator"]
        tasks = self._query_analyzer.plan_tasks(question, chat_history, available_tools=available_tools)
        if len(tasks) > 1:
            return {"tool": "multi", "reason": "planner split multi-intent", "tasks": tasks}
        # Single-task decision
        if not tasks:
            return {"tool": "direct", "reason": "empty tasks -> direct"}
        t0 = tasks[0]
        label = t0.get("tool")
        if label is None:
            return {"tool": "direct", "reason": "planner chose direct", "tasks": tasks}
        return {"tool": label, "reason": f"planner chose {label}", "tasks": tasks}

    # ---------- LLM Planner -------------------------------------------

    def _plan_tasks(self, user_text: str, chat_history: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        """Delegates planning to QueryAnalyzer.plan_tasks to ensure a single source of truth."""
        return self._query_analyzer.plan_tasks(user_text, chat_history)

    # ---------- Orchestrator (multi-intent via planner) ----------------

    async def orchestrate_answer(
        self, question: str, chat_history: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        await self.initialize_mcp()

        # Use planner from QueryAnalyzer to get ordered tasks with resolved fields
        available_tools = [t.get("name") for t in (self._mcp_client.tools or [])] or ["vector_search", "weather", "calculator"]
        tasks = self._query_analyzer.plan_tasks(question, chat_history, available_tools=available_tools)
        local_history: List[Dict[str, Any]] = list(chat_history or [])

        subtasks: List[Dict[str, Any]] = []
        tools_used: List[str] = []

        last_location = None

        for task in tasks:
            tool = task.get("tool")
            standalone = (task.get("standalone") or question).strip()

            # Direct (no tool) task
            if tool is None:
                da = self.generate_direct_answer(standalone, chat_history)
                subtasks.append({
                    "type": "direct",
                    "question": standalone,
                    "answer": da,
                })
                # update local history lightly
                local_history.extend([
                    {"role": "human", "content": standalone},
                    {"role": "ai", "content": da}
                ])
                continue

            if tool == "rag":
                rag = await self.use_vector_search(standalone, local_history)
                tools_used.append("rag_retrieval")
                snippets = rag.get("snippets", [])
                subtasks.append({
                    "type": "rag",
                    "question": standalone,
                    "effective_query": rag.get("effective_query"),
                    "snippets": snippets,
                })
                # update local history with short evidence
                if snippets:
                    local_history.extend([
                        {"role": "human", "content": standalone},
                        {"role": "ai", "content": snippets[0]}
                    ])
                    # try capture location for later weather
                    m = re.search(r"(?:headquarter(?:ed)?|based|located) in ([^.,]+)", snippets[0], re.I)
                    if m:
                        last_location = m.group(1).strip()
                continue

            if tool == "calculator":
                expr = task.get("expression") or await self._extract_calculation_expression(standalone)
                calc = await self.use_calculator(expr)
                tools_used.append("calculator")
                subtasks.append({
                    "type": "calculator",
                    "question": standalone,
                    "result": calc.get("result"),
                    "error": calc.get("error"),
                })
                local_history.extend([
                    {"role": "human", "content": standalone},
                    {"role": "ai", "content": str(calc.get("result"))}
                ])
                continue

            if tool == "weather":
                loc = task.get("location") or last_location
                w = await self.use_weather(standalone, loc)
                tools_used.append("weather")
                subtasks.append({
                    "type": "weather",
                    "question": standalone,
                    "status": w.get("status"),
                    "location": w.get("location"),
                })
                loc_txt = f" in {w.get('location')}" if w.get("location") else ""
                local_history.extend([
                    {"role": "human", "content": standalone},
                    {"role": "ai", "content": f"Weather{loc_txt}: {w.get('status')}"}
                ])
                continue

        # Sufficient?
        sufficient = all(
            (st["type"] != "calculator" or st.get("error") is None)
            and (st["type"] != "rag" or bool(st.get("snippets")))
            for st in subtasks
        )

        final_answer = self.generate_final_answer(question, subtasks, chat_history)

        return {
            "tools_used": tools_used,
            "answer": final_answer,
            "subtasks": subtasks,
            "sufficient": sufficient,
        }

    # ---------- Final answer composer ----------------------------------

    def generate_final_answer(
        self,
        original_question: str,
        subtasks: List[Dict[str, Any]],
        chat_history: List[Dict[str, Any]] | None = None,
    ) -> str:
        """
        Generate final answer using GPT-OSS from tool outputs.
        """
        context_parts: List[str] = []
        for st in subtasks:
            stype = st.get("type")
            if stype == "rag":
                snips = st.get("snippets", [])
                doc_snippets = []
                for s in snips[:10]:
                    content = s or ""
                    if len(content) > 400:
                        content = content[:400] + "..."
                    doc_snippets.append(f"- {content}")
                context_parts.append(
                    f"Evidence for '{st.get('question')}':\n"
                    + ("\n".join(doc_snippets) if doc_snippets else "(no docs)")
                )
            elif stype == "calculator":
                context_parts.append(
                    f"Calculation for '{st.get('question')}': result = {st.get('result')}"
                )
            elif stype == "weather":
                context_parts.append(
                    f"Weather for '{st.get('question')}': {st.get('status')}"
                )
            elif stype == "error":
                context_parts.append(f"Error on '{st.get('question')}': {st.get('error')}")

        context = "\n\n".join(context_parts)
        # Try to extract target_entity from any rag subtask meta.debug_info
        target_entity = ""
        for st in subtasks:
            if st.get("type") == "rag":
                meta = st.get("meta") or {}
                debug = meta.get("debug_info") or {}
                te = debug.get("target_entity") or ""
                if te:
                    target_entity = te
                    break

        entity_instruction = (
            f"Only use evidence that explicitly mentions the entity: '{target_entity}'. "
            f"Ignore snippets about other entities. " if target_entity else ""
        )
        prompt = (
            "You are a helpful AI assistant. Using ONLY the provided information below, "
            "answer the user's question(s). If multiple questions appear, answer each clearly in order. "
            "Do not mention your process or any tools. Keep the answer concise and factual.\n\n"
            + (f"Entity: {target_entity}\n" if target_entity else "")
            + entity_instruction
            + "\n\n"
            + f"Original Question: {original_question}\n\nContext:\n{context}\n\nFinal Answer:"
        )

        resp = self._llm.invoke(prompt)
        return getattr(resp, "content", str(resp))

    # ---------- Public API ---------------------------------------------

    async def answer(
        self, question: str, chat_history: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """Main entrypoint for answering questions using MCP tools."""
        # Early guard: greetings/small talk -> direct LLM (no tools)
        if self._looks_like_greeting(question):
            final = self.generate_direct_answer(question, chat_history)
            return {"answer": final, "sufficient": True}

        decision = await self.decide_tool(question, chat_history)

        # Multi-intent ⇒ orchestrate per sub-question
        if decision["tool"] == "multi":
            return await self.orchestrate_answer(question, chat_history)

        # Single-intent fast paths
        if decision["tool"] == "weather":
            w = await self.use_weather(question)
            subtasks = [{"type": "weather", "question": question, "status": w.get("status")}]
            final = self.generate_final_answer(question, subtasks, chat_history)
            return {
                "tools_used": ["weather"],
                "answer": final,
                "subtasks": subtasks,
                "sufficient": True,
            }

        if decision["tool"] == "calculator":
            expr = await self._extract_calculation_expression(question)
            calc = await self.use_calculator(expr)
            subtasks = [{"type": "calculator", "question": question, "result": calc.get("result"), "error": calc.get("error")}]
            final = self.generate_final_answer(question, subtasks, chat_history)
            return {
                "tools_used": ["calculator"],
                "answer": final,
                "subtasks": subtasks,
                "sufficient": calc.get("error") is None,
            }

        # Default: RAG for knowledge questions
        rag = await self.use_vector_search(question, chat_history)
        snippets = rag.get("snippets", [])
        subtasks = [
            {
                "type": "rag",
                "question": question,
                "effective_query": rag.get("effective_query"),
                "snippets": snippets,
            }
        ]
        final = self.generate_final_answer(question, subtasks, chat_history)
        return {
            "tools_used": ["rag_retrieval"],
            "answer": final,
            "subtasks": subtasks,
            "sufficient": bool(snippets),
            "snippets": snippets[:20],
        }

    async def cleanup(self):
        """Cleanup MCP connection."""
        if self._mcp_client:
            await self._mcp_client.cleanup()
