from __future__ import annotations

from .types import MCPClient
from .providers import (
    CalculatorProvider,
    WeatherProvider,
    PrivateSearchProvider,
    ExpansionProvider,
    FusionProvider,
    RerankProvider,
)


def create_mcp_client() -> MCPClient:
    client = MCPClient()
    client.register_tool(CalculatorProvider())
    client.register_tool(WeatherProvider())
    client.register_tool(PrivateSearchProvider())
    client.register_tool(ExpansionProvider())
    client.register_tool(FusionProvider())
    client.register_tool(RerankProvider())
    return client


