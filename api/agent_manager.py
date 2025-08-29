"""Agent Manager: decide and execute tools for Agentic RAG.

Tools:
- rag_retrieval: Multi-Query → RRF → Rerank (returns final reranked documents)
- calculator: Evaluate simple arithmetic expressions when user asks to compute
- weather: Get weather information (toy implementation)

Usage (khuyến nghị):
    manager = AgentManager()
    res = manager.answer(user_query, chat_history)
    print(res["answer"])
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
import os

from langchain_groq import ChatGroq
from query_analyzer import QueryAnalyzer
from tools.weather_tool import WeatherTool
from tools.calculator_tool import CalculatorTool
from agents.agent_private import AgentPrivate


class AgentManager:
    """Selects and runs tools for queries (multi-intent aware)."""

    # ---------- class-level regex / tokens ----------
    # legacy calc/weather hints now handled in tool modules
    _PRIVATE_HINTS = (
        # simple heuristics for private data needs (entities/docs/internal)
        "greengrow", "quantumnext", "techwave", "greenfields",
        "document", "kb", "knowledge base", "internal", "private",
    )
    _GREETING_HINTS = (
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "xin chào", "chào", "chao", "chào bạn", "chào anh", "chào chị", "yo", "sup"
    )
    _MULTI_SEP = re.compile(r"\?+|\s+(?:and|&)\s+|,\s+", re.I)

    def __init__(self):
        # Lazy pipeline creation; initialized on first RAG use (nếu cần)
        self._pipeline = None
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )
        # Direct tool instances
        self._weather = WeatherTool()
        self._calculator = CalculatorTool()
        self._private = AgentPrivate()
        self._query_analyzer = QueryAnalyzer()

    # ==========================
    # Query Analysis Integration
    # ==========================
    def _analyze_query_with_llm(self, raw: str) -> List[str]:
        """Use QueryAnalyzer to analyze and split query into subtasks."""
        return self._query_analyzer.analyze_query(raw)

    def _detect_calculator_intent(self, question: str) -> bool:
        """Use LLM to detect if question contains arithmetic calculation."""
        prompt = f"""
Analyze if this question contains a mathematical calculation that can be computed.
Look for arithmetic operations: +, -, *, / with numbers.

Question: {question}

Respond with only "YES" if it contains a calculable expression, or "NO" if not.
"""
        try:
            resp = self._llm.invoke(prompt)
            text = (getattr(resp, "content", None) or str(resp)).strip().upper()
            return text == "YES"
        except Exception:
            return False

    def _extract_calculation_expression(self, question: str) -> str:
        """Use LLM to extract arithmetic expression from question."""
        prompt = f"""
Extract the mathematical expression from this question.
Return ONLY the expression with numbers and operators (+, -, *, /), no text.

Question: {question}

Expression:"""
        try:
            resp = self._llm.invoke(prompt)
            expr = (getattr(resp, "content", None) or str(resp)).strip()
            # Clean up any extra text or formatting
            expr = re.sub(r'[^0-9\+\-\*/\(\)\.\s]', '', expr).strip()
            return expr
        except Exception:
            return ""

    # ==========================
    # Decision
    # ==========================
    def decide_tool(self, question: str, chat_history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """
        Quyết định cho lượt hỏi hiện tại:
        - 'multi' nếu phát hiện nhiều câu hỏi.
        - 'calculator' nếu 1 câu đơn là phép tính.
        - 'weather' nếu 1 câu đơn hỏi thời tiết.
        - 'rag' nếu cần private data.
        - 'direct' cho tất cả trường hợp còn lại (bao gồm greeting/general).
        """
        chunks = self._analyze_query_with_llm(question)
        if len(chunks) > 1:
            return {"tool": "multi", "reason": "multi-intent detected", "chunks": chunks}

        # If the single chunk differs from original (LLM likely normalized entity), prefer using it for downstream
        normalized = chunks[0] if chunks else question
        if normalized and normalized != question:
            question = normalized

        if self._weather.looks_like_weather(question):
            return {"tool": "weather", "reason": "weather intent"}

        if self._detect_calculator_intent(question):
            return {"tool": "calculator", "reason": "detected arithmetic expression"}

        # Heuristic: use private search if looks like private data OR relies on prior context
        if self._looks_like_private_data(question) or self._needs_context_from_history(question, chat_history):
            return {"tool": "rag", "reason": "private data likely needed"}

        return {"tool": "direct", "reason": "direct LLM response (no tools)"}

    def _needs_context_from_history(self, question: str, chat_history: List[Dict[str, Any]] | None) -> bool:
        """Detect questions that probably refer to previous entity (pronouns/ellipsis)."""
        if not chat_history:
            return False
        text = (question or "").strip().lower()
        # simple pronoun/ellipsis hints
        hints = (" it ", " its ", " they ", " their ", "where is it", "when was it", "headquartered", "founded")
        # pad spaces to detect whole-word for ' it '
        padded = f" {text} "
        return any(h in padded for h in hints)

    def _looks_like_private_data(self, question: str) -> bool:
        text = (question or "").strip().lower()
        return any(h in text for h in self._PRIVATE_HINTS)

    # calculator/weather detection delegated to tool modules

    def _looks_like_greeting(self, question: str) -> bool:
        text = (question or "").strip().lower()
        # Match whole words only to avoid 'yo' matching in 'you'
        greeting_pattern = (
            r"\b(?:hello|hi|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|"
            r"xin\s+chào|chào|chao|chào\s+bạn|chào\s+anh|chào\s+chị|yo|sup)\b"
        )
        if len(text) <= 40 and re.search(greeting_pattern, text):
            return True
        return text in ("hello", "hi", "hey", "xin chào", "chào")

    def use_weather(self, question: str) -> Dict[str, Any]:
        return self._weather.use_weather(question)

    def use_calculator(self, expression: str) -> Dict[str, Any]:
        return self._calculator.calculate(expression)

    def extract_calculation_expression(self, question: str) -> str:
        """Public wrapper for extracting arithmetic expression from a question."""
        return self._extract_calculation_expression(question)

    # ==========================
    # Generation via GPT-OSS (main method for final answer generation)
    # ==========================
    def _format_context(self, subtasks: List[Dict[str, Any]]) -> str:
        """Format context from tool outputs for final answer generation."""
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
                    f"Evidence for '{st.get('question')}':\n" +
                    ("\n".join(doc_snippets) if doc_snippets else "(no docs)")
                )
            elif st.get("type") == "calculator":
                parts.append(
                    f"Calculation: {st.get('question')} = {st.get('result')}"
                )
            elif st.get("type") == "weather":
                parts.append(
                    f"Weather: {st.get('status')}"
                )
            elif st.get("type") == "greeting":
                parts.append(
                    f"Greeting: {st.get('answer')}"
                )
        return "\n\n".join(parts)

    def generate_final_answer(
        self,
        original_question: str, 
        subtasks: List[Dict[str, Any]], 
        chat_history: List[Dict[str, Any]] | None = None
    ) -> str:
        """Generate final answer using GPT-OSS from tool outputs."""
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
                "You are a helpful AI assistant. Using ONLY the provided information below, answer the user's question(s). "
                "If multiple questions appear, answer each clearly in order. Do not mention your process or any tools. "
                "Keep the answer concise and factual.\n\n"
                f"Original Question: {original_question}\n\nContext:\n{context}\n\nChat History:\n{hist_text}\n\nFinal Answer:"
            )
        else:
            prompt = (
                "You are a helpful AI assistant. Using ONLY the provided information below, answer the user's question(s). "
                "If multiple questions appear, answer each clearly in order. Do not mention your process or any tools. "
                "Keep the answer concise and factual.\n\n"
                f"Original Question: {original_question}\n\nContext:\n{context}\n\nFinal Answer:"
            )

        resp = self._llm.invoke(prompt)
        return getattr(resp, "content", str(resp))

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

    # ==========================
    # Orchestration (đa-tool)
    # ==========================
    def orchestrate_answer(
        self,
        question: str,
        chat_history: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """Full agentic flow với planning & multi-tool orchestration."""
        # Sử dụng LLM để phân tích query thành subtasks
        candidates = self._analyze_query_with_llm(question)

        subtasks: List[Dict[str, Any]] = []
        tools_used: List[str] = []

        for sub_q in candidates:
            # Handle greeting chunks (no tools needed, but track in subtasks)
            if self._looks_like_greeting(sub_q):
                da = self.generate_direct_answer(sub_q, chat_history)
                subtasks.append({
                    "type": "greeting",
                    "question": sub_q,
                    "answer": da,
                })
                continue

            if self._weather.looks_like_weather(sub_q):
                w = self.use_weather(sub_q)
                tools_used.append("weather")
                subtasks.append({
                    "type": "weather",
                    "question": sub_q,
                    **w,
                })
            elif self._detect_calculator_intent(sub_q):
                expr = self._extract_calculation_expression(sub_q)
                calc = self.use_calculator(expr)
                tools_used.append("calculator")
                subtasks.append({
                    "type": "calculator",
                    "question": sub_q,
                    "result": calc.get("result"),
                    "error": calc.get("error"),
                })
            else:
                # Route sang RAG cho câu tri thức trực tiếp
                rag = self._private.search(sub_q, chat_history)
                tools_used.append("rag_retrieval")
                subtasks.append({
                    "type": "rag",
                    "question": sub_q,
                    "effective_query": rag.get("effective_query"),
                    "final_documents": rag.get("final_documents", []),
                    "meta": rag.get("meta", {}),
                })

        # Kiểm tra đủ điều kiện
        sufficient = True
        for st in subtasks:
            if st.get("type") == "rag" and not st.get("final_documents"):
                sufficient = False
            if st.get("type") == "calculator" and st.get("error") is not None:
                sufficient = False

        # Sinh câu trả lời cuối
        final_answer = self.generate_final_answer(question, subtasks, chat_history)

        # Chỉ gắn meta/final_documents nếu có RAG
        has_rag = any(st.get("type") == "rag" for st in subtasks)
        if has_rag:
            agg_meta: Dict[str, Any] = {
                "expanded_queries": [],
                "retrieved_docs_count": 0,
                "fused_docs_count": 0,
                "final_docs_count": 0
            }
            merged_docs: List[Dict[str, Any]] = []
            for st in subtasks:
                if st.get("type") == "rag":
                    m = st.get("meta", {})
                    agg_meta["expanded_queries"].extend(m.get("expanded_queries", []))
                    agg_meta["retrieved_docs_count"] += m.get("retrieved_docs_count", 0)
                    agg_meta["fused_docs_count"] += m.get("fused_docs_count", 0)
                    agg_meta["final_docs_count"] += m.get("final_docs_count", 0)
                    merged_docs.extend(st.get("final_documents", []))

            return {
                "tools_used": tools_used,
                "answer": final_answer,
                "subtasks": subtasks,
                "sufficient": sufficient,
                "meta": agg_meta,
                "final_documents": merged_docs[:20],
            }

        # Không có RAG -> schema gọn (giống calculator)
        return {
            "tools_used": tools_used,
            "answer": final_answer,
            "subtasks": subtasks,
            "sufficient": sufficient
        }

    # ==========================
    # Public entrypoint mới
    # ==========================
    def answer(
        self,
        question: str,
        chat_history: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """
        Entrypoint khuyến nghị.
        - Multi-intent -> orchestrate_answer()
        - Direct (greeting, general) -> generate_direct_answer() [no tools]
        - Weather câu đơn -> use_weather() + generate_final_answer()
        - Calculator câu đơn -> use_calculator() + generate_final_answer()
        - RAG câu đơn -> run_rag + generate_final_answer()
        """
        decision = self.decide_tool(question, chat_history)

        if decision["tool"] == "multi":
            return self.orchestrate_answer(question, chat_history)

        if decision["tool"] == "direct":
            # Direct LLM response (greeting, general questions)
            final = self.generate_direct_answer(question, chat_history)
            return {
                "answer": final,
                "sufficient": True
            }

        if decision["tool"] == "weather":
            w = self.use_weather(question)
            # Vẫn đi qua generate_final_answer để thống nhất giọng văn
            subtasks = [{"type": "weather", "question": question, **w}]
            final = self.generate_final_answer(question, subtasks, chat_history)
            return {
                "tools_used": ["weather"],
                "answer": final,
                "subtasks": subtasks,
                "sufficient": True
                # KHÔNG trả final_documents / meta cho weather
            }

        if decision["tool"] == "calculator":
            expr = self._extract_calculation_expression(question)
            calc = self.use_calculator(expr)
            # Vẫn đi qua generate_final_answer để thống nhất giọng văn
            subtasks = [{"type": "calculator", "question": question, **calc}]
            final = self.generate_final_answer(question, subtasks, chat_history)
            return {
                "tools_used": ["calculator"],
                "answer": final,
                "subtasks": subtasks,
                "sufficient": calc.get("error") is None
                # KHÔNG trả final_documents / meta cho calculator
            }

        # RAG câu đơn
        rag = self._private.search(question, chat_history)
        docs = rag.get("final_documents", [])
        subtasks = [{
            "type": "rag",
            "question": question,
            "final_documents": docs,
            "meta": rag.get("meta", {})
        }]
        final = self.generate_final_answer(question, subtasks, chat_history)
        return {
            "tools_used": ["rag_retrieval"],
            "answer": final,
            "subtasks": subtasks,
            "sufficient": bool(docs),
            "final_documents": docs[:20],
            "meta": rag.get("meta", {})
        }