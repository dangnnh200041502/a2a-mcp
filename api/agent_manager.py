"""Agent Manager: decide and execute tools for Agentic RAG.

Tools:
- rag_retrieval: Multi-Query → RRF → Rerank (returns final reranked documents)
- calculator: Evaluate simple arithmetic expressions when user asks to compute

Usage (khuyến nghị):
    manager = AgentManager()
    res = manager.answer(user_query, chat_history)
    print(res["answer"])
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
import os

from langchain_utils import get_advanced_rag_pipeline  # nếu bạn không dùng trực tiếp có thể giữ import
from langchain_groq import ChatGroq
from private_agent import PrivateDataAgent
from final_agent import FinalAgent


class AgentManager:
    """Selects and runs tools for queries (multi-intent aware)."""

    # ---------- class-level regex / tokens ----------
    _SAFE_PATTERN = re.compile(r"^[\s\d\+\-\*/\(\)\.]+$")
    _OP_TOKENS = ("+", "-", "*", "/")
    _NL_CALC_HINTS = (
        "tinh", "tính", "sum", "plus", "add", "calculate", "calc",
        "total", "result of",
    )
    _WEATHER_HINTS = (
        "weather", "thoi tiet", "thời tiết", "forecast", "rain", "sunny", "temperature",
    )
    _MULTI_SEP = re.compile(r"\?+|\s+(?:and|&)\s+|,\s+", re.I)

    def __init__(self):
        # Lazy pipeline creation; initialized on first RAG use (nếu cần)
        self._pipeline = None
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )
        self._private_agent = PrivateDataAgent()
        self._final_agent = FinalAgent()
        self._last_expr: str | None = None

    # ==========================
    # Helpers: splitting & routing
    # ==========================
    def _split_candidates(self, raw: str) -> List[str]:
        """Tách chuỗi người dùng thành các câu hỏi con (sub-queries) tiềm năng."""
        raw = re.sub(r"\s+", " ", (raw or "").strip())

        # 1) Ưu tiên tách theo '?'
        parts_q = [p.strip() for p in re.split(r"\?+", raw) if p.strip()]
        if len(parts_q) >= 2:
            return [p if p.endswith("?") else p + "?" for p in parts_q]

        # 2) Fallback: tách theo 'and', '&', hoặc dấu phẩy
        parts = [p.strip() for p in self._MULTI_SEP.split(raw) if p.strip()]

        # 3) Lọc nhiễu & giữ lại vế có tính chất câu hỏi/biểu thức
        keep: List[str] = []
        for p in parts:
            lp = p.lower()
            is_interrogative = any(w in lp for w in ["what", "when", "where", "who", "how", "which"])
            has_math = (any(op in lp for op in self._OP_TOKENS) and re.search(r"\d", lp)) or ("result of" in lp)
            if is_interrogative or has_math:
                keep.append(p if p.endswith("?") else p + "?")

        return keep or [raw if raw.endswith("?") else raw + "?"]

    def _has_mixed_intents(self, chunks: List[str]) -> bool:
        """Có cả câu toán và câu cần RAG trong cùng lượt không?"""
        saw_calc, saw_rag = False, False
        for c in chunks:
            if self._looks_like_calculation(c):
                saw_calc = True
            else:
                saw_rag = True
        return saw_calc and saw_rag

    # ==========================
    # Decision
    # ==========================
    def decide_tool(self, question: str) -> Dict[str, Any]:
        """
        Quyết định cho lượt hỏi hiện tại:
        - 'multi' nếu phát hiện nhiều câu hỏi hoặc ý định lẫn nhau (calc + rag).
        - 'calculator' nếu 1 câu đơn là phép tính.
        - 'weather' nếu 1 câu đơn hỏi thời tiết.
        - 'rag' cho các trường hợp còn lại.
        """
        chunks = self._split_candidates(question)
        if len(chunks) > 1 or self._has_mixed_intents(chunks):
            return {"tool": "multi", "reason": "multi-intent detected", "chunks": chunks}

        if self._looks_like_weather(question):
            return {"tool": "weather", "reason": "weather intent"}

        if self._looks_like_calculation(question):
            return {"tool": "calculator", "reason": "detected arithmetic expression"}

        return {"tool": "rag", "reason": "default retrieval path"}

    # ==========================
    # Calculator
    # ==========================
    def _extract_math_expression(self, text: str) -> str | None:
        # Loại bỏ ký tự không an toàn; giữ số, +-*/(). và khoảng trắng
        candidate = re.sub(r"[^0-9\+\-\*/\(\)\.\s]", "", text)
        candidate = candidate.strip().strip("=?")
        candidate = re.sub(r"\s+", " ", candidate)
        # Cần có ít nhất 1 chữ số và 1 toán tử
        if any(op in candidate for op in self._OP_TOKENS) and re.search(r"\d", candidate):
            if self._SAFE_PATTERN.match(candidate):
                return candidate
        return None

    def _looks_like_calculation(self, question: str) -> bool:
        text = (question or "").strip().lower()
        has_hint = any(h in text for h in self._NL_CALC_HINTS)
        has_digits_ops = any(op in text for op in self._OP_TOKENS) and re.search(r"\d", text)
        if not (has_hint or has_digits_ops):
            return False
        expr = self._extract_math_expression(text)
        if expr:
            self._last_expr = expr
            return True
        return False

    # ==========================
    # Weather (toy) detector & tool
    # ==========================
    def _looks_like_weather(self, question: str) -> bool:
        text = (question or "").strip().lower()
        return any(h in text for h in self._WEATHER_HINTS)

    def use_weather(self, question: str) -> Dict[str, Any]:
        # Toy tool: always sunny
        return {"tool": "weather", "location": None, "status": "sunny"}

    def use_calculator(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate a basic arithmetic expression (context fence)."""
        expr = (self._last_expr or "").strip()
        if not expr:
            expr = self._extract_math_expression((expression or "").strip()) or ""
        if not self._SAFE_PATTERN.match(expr):
            return {"tool": "calculator", "error": "unsupported expression", "result": None}
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return {"tool": "calculator", "result": result}
        except Exception as e:
            return {"tool": "calculator", "error": str(e), "result": None}

    # ==========================
    # Generation via GPT-OSS (giữ nguyên)
    # ==========================
    def _format_docs_as_context(self, documents: List[Any]) -> str:
        parts = []
        for doc in documents or []:
            content = getattr(doc, "content", None)
            if not content:
                continue
            parts.append(str(content))
        return "\n\n".join(parts)

    def generate_with_gpt_oss(
        self,
        question: str,
        documents: List[Any],
        chat_history: List[Dict[str, Any]] | None = None
    ) -> str:
        context = self._format_docs_as_context(documents)
        if not context:
            prompt = f"Answer concisely: {question}"
            resp = self._llm.invoke(prompt)
            return getattr(resp, "content", str(resp))

        if chat_history:
            history_lines = []
            for m in chat_history:
                role = m.get("role")
                content = m.get("content", "")
                if role == "human":
                    history_lines.append(f"User: {content}")
                elif role == "ai":
                    history_lines.append(f"Assistant: {content}")
            history_text = "\n".join(history_lines)
            prompt = (
                "You are a helpful AI assistant. Use the following retrieved context and chat "
                "history to answer the user's question as accurately and concisely as possible.\n\n"
                f"Context:\n{context}\n\nChat History:\n{history_text}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            prompt = (
                "You are a helpful AI assistant. Use the following retrieved context to answer "
                "the user's question as accurately and concisely as possible.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
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
        raw = (question or "").strip()
        raw_norm = re.sub(r"\s+", " ", raw)

        # Tách câu hỏi thành candidates
        parts_q = [p.strip() for p in re.split(r"\?+", raw_norm) if p.strip()]
        if len(parts_q) >= 2:
            candidates: List[str] = [p if p.endswith("?") else p + "?" for p in parts_q]
        else:
            temp = re.split(r"\s+(?:and|&)\s+|,\s+", raw_norm)
            temp = [t.strip() for t in temp if t.strip()]
            if len(temp) >= 2:
                candidates = []
                for t in temp:
                    lt = t.lower()
                    is_interrogative = any(w in lt for w in ["what", "when", "where", "who", "how", "which"])
                    has_math = (any(op in lt for op in self._OP_TOKENS) and re.search(r"\d", lt)) or ("result of" in lt)
                    if is_interrogative or has_math:
                        candidates.append(t if t.endswith("?") else t + "?")
                if not candidates:
                    candidates = [raw_norm]
            else:
                candidates = [raw_norm]

        subtasks: List[Dict[str, Any]] = []
        tools_used: List[str] = []

        for sub_q in candidates:
            if self._looks_like_weather(sub_q):
                w = self.use_weather(sub_q)
                tools_used.append("weather")
                subtasks.append({
                    "type": "weather",
                    "question": sub_q,
                    **w,
                })
            elif self._looks_like_calculation(sub_q):
                calc = self.use_calculator(sub_q)
                tools_used.append("calculator")
                subtasks.append({
                    "type": "calculator",
                    "question": sub_q,
                    "result": calc.get("result"),
                    "error": calc.get("error"),
                })
            else:
                # Route sang RAG cho câu tri thức
                rag = self._private_agent.run_rag_retrieval(sub_q, chat_history)
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
        final_answer = self._final_agent.generate(question, subtasks, chat_history)

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
        - Weather câu đơn -> use_weather() (schema gọn, không meta/final_docs)
        - Calculator câu đơn -> use_calculator() (schema gọn)
        - RAG câu đơn -> run_rag + final_agent
        """
        decision = self.decide_tool(question)

        if decision["tool"] == "multi":
            return self.orchestrate_answer(question, chat_history)

        if decision["tool"] == "weather":
            w = self.use_weather(question)
            # Vẫn đi qua FinalAgent để thống nhất giọng văn
            subtasks = [{"type": "weather", "question": question, **w}]
            final = self._final_agent.generate(question, subtasks, chat_history)
            return {
                "tools_used": ["weather"],
                "answer": final,
                "subtasks": subtasks,
                "sufficient": True
                # KHÔNG trả final_documents / meta cho weather
            }

        if decision["tool"] == "calculator":
            calc = self.use_calculator(question)
            text = "Result: " + (str(calc["result"]) if calc.get("error") is None else "Error")
            return {
                "tools_used": ["calculator"],
                "answer": text,
                "subtasks": [{"type": "calculator", "question": question, **calc}],
                "sufficient": calc.get("error") is None
                # KHÔNG trả final_documents / meta cho calculator
            }

        # RAG câu đơn
        rag = self._private_agent.run_rag_retrieval(question, chat_history)
        docs = rag.get("final_documents", [])
        subtasks = [{
            "type": "rag",
            "question": question,
            "final_documents": docs,
            "meta": rag.get("meta", {})
        }]
        final = self._final_agent.generate(question, subtasks, chat_history)
        return {
            "tools_used": ["rag_retrieval"],
            "answer": final,
            "subtasks": subtasks,
            "sufficient": bool(docs),
            "final_documents": docs[:20],
            "meta": rag.get("meta", {})
        }
