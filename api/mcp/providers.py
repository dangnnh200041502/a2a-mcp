from __future__ import annotations

from typing import Any, Dict

from .types import MCPTool
from tools.calculator_tool import CalculatorTool
from tools.weather_tool import WeatherTool
from agents.agent_private import AgentPrivate
from tools.expansion_tool import ExpansionTool
from tools.fusion_tool import FusionTool
from tools.rerank_tool import RerankTool


class CalculatorProvider(MCPTool):
    name = "calculator"

    def __init__(self) -> None:
        self._impl = CalculatorTool()

    def looks_like(self, question: str) -> bool:
        return self._impl.looks_like_calculation(question)

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("expression") or params.get("natural_text") or ""
        return self._impl.use_calculator(query)


class WeatherProvider(MCPTool):
    name = "weather"

    def __init__(self) -> None:
        self._impl = WeatherTool()

    def looks_like(self, question: str) -> bool:
        return self._impl.looks_like_weather(question)

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        question = params.get("question", "")
        return self._impl.use_weather(question)


class PrivateSearchProvider(MCPTool):
    name = "private_search"

    def __init__(self) -> None:
        # pass MCP client into AgentPrivate to avoid circular imports at module import time
        self._client = None
        self._impl = None

    def looks_like(self, question: str) -> bool:
        return False

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Lazy init client to break circular import
        if self._impl is None:
            from .client_factory import create_mcp_client
            self._client = create_mcp_client()
            self._impl = AgentPrivate(client=self._client)

        question = params.get("question", "")
        chat_history = params.get("chat_history")
        res = self._impl.search(question, chat_history)
        return {
            "effective_query": res.get("original_query", question),
            "meta": {
                "expanded_queries": res.get("expanded_queries", []),
                "retrieved_docs_count": res.get("retrieved_docs_count", 0),
                "fused_docs_count": res.get("fused_docs_count", 0),
                "final_docs_count": res.get("final_docs_count", 0),
            },
            "final_documents": res.get("final_documents", []),
        }


class ExpansionProvider(MCPTool):
    name = "expansion"

    def __init__(self) -> None:
        self._impl = ExpansionTool()

    def looks_like(self, question: str) -> bool:
        return False

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._impl.invoke(params)


class FusionProvider(MCPTool):
    name = "fusion"

    def __init__(self) -> None:
        self._impl = FusionTool()

    def looks_like(self, question: str) -> bool:
        return False

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._impl.invoke(params)


class RerankProvider(MCPTool):
    name = "rerank"

    def __init__(self) -> None:
        self._impl = RerankTool()

    def looks_like(self, question: str) -> bool:
        return False

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._impl.invoke(params)


