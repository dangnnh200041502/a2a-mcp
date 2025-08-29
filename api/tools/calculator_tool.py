from __future__ import annotations

import re
from typing import Any, Dict


class CalculatorTool:
    """Pure arithmetic calculator - only handles +, -, *, / operations."""

    _SAFE_PATTERN = re.compile(r"^[\s\d\+\-\*/\(\)\.]+$")

    def calculate(self, expression: str) -> Dict[str, Any]:
        """Calculate arithmetic expression with only +, -, *, / operations."""
        expr = (expression or "").strip()
        
        # Validate expression contains only safe characters
        if not self._SAFE_PATTERN.match(expr):
            return {"error": "unsupported expression", "result": None}
        
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return {"result": result}
        except Exception as e:
            return {"error": str(e), "result": None}


