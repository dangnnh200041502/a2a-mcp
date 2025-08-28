from __future__ import annotations

from typing import Protocol, runtime_checkable, Any, Dict, Optional


@runtime_checkable
class MCPTool(Protocol):
    """Minimal MCP-like tool interface for local execution.

    This mirrors the Model Context Protocol idea: tools are discoverable,
    have a name, optional input schema (not enforced here), and an invoke method
    returning structured data.
    """

    name: str

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def looks_like(self, question: str) -> bool:
        """Optional intent detector for routing convenience."""
        return False


class MCPClient:
    """Lightweight in-process MCP client/registry.

    - register_tool(name, tool)
    - call(name, params) -> tool.invoke(params)
    - detect(name, question) -> tool.looks_like(question)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, MCPTool] = {}

    def register_tool(self, tool: MCPTool) -> None:
        self._tools[tool.name] = tool

    def call(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"MCP tool not found: {name}")
        return self._tools[name].invoke(params or {})

    def detect(self, name: str, question: str) -> bool:
        tool = self._tools.get(name)
        if tool is None:
            return False
        try:
            return bool(tool.looks_like(question))
        except Exception:
            return False


