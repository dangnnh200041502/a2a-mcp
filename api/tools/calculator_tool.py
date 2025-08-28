from __future__ import annotations

import re
from typing import Any, Dict


class CalculatorTool:
    """Sandboxed arithmetic evaluator with simple intent detection.

    Holds minimal state (last extracted expression) to support downstream calls.
    """

    _SAFE_PATTERN = re.compile(r"^[\s\d\+\-\*/\(\)\.]+$")
    _OP_TOKENS = ("+", "-", "*", "/")
    _NL_CALC_HINTS = (
        "tinh", "tính", "sum", "plus", "add", "calculate", "calc",
        "total", "result of",
    )

    def __init__(self) -> None:
        self._last_expr: str | None = None

    def _extract_math_expression(self, text: str) -> str | None:
        # Remove leading bullet-like hyphens to avoid misreading minus
        text = re.sub(r"^\s*[\-–]\s+", "", text)
        # Handle common natural-language patterns first
        m = re.search(r"sum\s+of\s+(\d+(?:\.\d+)?)\s+(?:and|&)\s+(\d+(?:\.\d+)?)", text, re.I)
        if m:
            a, b = m.group(1), m.group(2)
            return f"{a} + {b}"
        m = re.search(r"(\d+(?:\.\d+)?)\s+(?:plus|add(?:ed)?\s+to)\s+(\d+(?:\.\d+)?)", text, re.I)
        if m:
            a, b = m.group(1), m.group(2)
            return f"{a} + {b}"
        # Keep only safe characters
        candidate = re.sub(r"[^0-9\+\-\*/\(\)\.\s]", "", text)
        candidate = candidate.strip().strip("=?")
        candidate = re.sub(r"\s+", " ", candidate)
        if any(op in candidate for op in self._OP_TOKENS) and re.search(r"\d", candidate):
            if self._SAFE_PATTERN.match(candidate):
                return candidate
        return None

    def looks_like_calculation(self, question: str) -> bool:
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

    def use_calculator(self, expression: str) -> Dict[str, Any]:
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



