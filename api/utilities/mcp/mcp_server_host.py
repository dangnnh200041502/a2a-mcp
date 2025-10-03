from __future__ import annotations

import json
from mcp.server.fastmcp import FastMCP
from uuid import uuid4
from dotenv import load_dotenv

from utilities.a2a.agent_discovery import AgentDiscovery
from utilities.a2a.agent_connect import AgentConnector

load_dotenv()

mcp = FastMCP("host-router-tools")


@mcp.tool()
async def list_agents() -> str:
    """Return discovered AgentCards (A2A child agents) as a JSON string."""
    discovery = AgentDiscovery()
    cards = await discovery.list_agent_cards()
    try:
        names = [c.name for c in cards]
    except Exception:
        names = []
    return json.dumps([c.model_dump(exclude_none=True) for c in cards], ensure_ascii=False)


@mcp.tool()
async def delegate_task(agent_name: str, message: str) -> str:
    """Delegate the original user message to an agent by name and return its response."""
    discovery = AgentDiscovery()
    cards = await discovery.list_agent_cards()
    matched = None
    for c in cards:
        try:
            if c.name.lower() == (agent_name or "").lower():
                matched = c
                break
            elif getattr(c, "id", "").lower() == (agent_name or "").lower():
                matched = c
                break
        except Exception as e:
            pass
    if matched is None:
        return "Agent not found"
    connector = AgentConnector(agent_card=matched)
    session_id = str(uuid4())
    return await connector.send_task(message=message, session_id=session_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")


