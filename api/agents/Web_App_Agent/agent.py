import asyncio
import json
from collections.abc import AsyncIterable
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq.chat_models import ChatGroq
from langchain.chat_models import init_chat_model
from langgraph.constants import END
from langgraph.graph import StateGraph
from dotenv import load_dotenv

from agents.Web_App_Agent.states import *
from utilities.common.file_loader import load_instructions_file

# Load environment variables (e.g., GROQ_API_KEY) before initializing model
load_dotenv()
# Use ChatGroq like the API source implementation
llm = ChatGroq(model="openai/gpt-oss-120b")


class WebAgent:
    """
    LangGraph-based Web Agent that compiles a pipeline (planner -> architect -> coder)
    and uses MCP tools. Implements an `invoke` method compatible with A2A Executor.
    """

    def __init__(self):
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._mcp_tools = None
        self.description = load_instructions_file("agents/Web_App_Agent/description.txt")
        self.instructions = load_instructions_file("agents/Web_App_Agent/instructions.txt")

    async def initialize(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            # Initialize MCP client for web tools
            self._mcp_client = MultiServerMCPClient({
                "web": {
                    "command": "python",
                    "args": ["-m", "utilities.mcp.mcp_server_web"],
                    "transport": "stdio",
                }
            })
            self._mcp_tools = await self._mcp_client.get_tools()
            self._initialized = True

    async def invoke(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """A2A-compatible async generator stream."""
        await self.initialize()

        if not query or not isinstance(query, str):
            yield {
                "is_task_complete": True,
                "content": "Invalid input query"
            }
            return

        # Emitting processing status
        yield {
            "is_task_complete": False,
            "updates": "Starting pipeline..."
        }

        try:
            # Run LangGraph pipeline
            initial_state = {
                "user_prompt": query.strip(),
                "mcp_tools": self._mcp_tools,
            }
            graph = _get_web_graph()
            final_state = await graph.ainvoke(initial_state, {"recursion_limit": 100})

            # Short, consistent success message
            yield {
                "is_task_complete": True,
                "content": "Web page has been created successfully! Check the generated files (index.html, styles.css, script.js) in your project directory."
            }

        except Exception as e:
            yield {
                "is_task_complete": True,
                "content": f"Error in WebAgent: {str(e)}"
            }

    async def cleanup(self) -> None:
        if hasattr(self, "_mcp_client") and self._mcp_client:
            await self._mcp_client.aclose()
        return None


# ========== LangGraph Nodes ==========

def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    base = load_instructions_file("agents/Web_App_Agent/planner_instructions.txt")
    prompt = f"{base}\n\nUser request:\n{user_prompt}\n"
    resp = llm.with_structured_output(Plan, method="json_schema").invoke(
        prompt
    )
    if resp is None:
        raise ValueError("Planner failed to produce output.")
    out = {"plan": resp}
    if "mcp_tools" in state:
        out["mcp_tools"] = state["mcp_tools"]
    return out


def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]
    base = load_instructions_file("agents/Web_App_Agent/architect_instructions.txt")
    prompt = f"{base}\n\nProject Plan:\n{plan.model_dump_json()}\n"
    resp = llm.with_structured_output(TaskPlan, method="json_schema").invoke(
        prompt
    )
    if resp is None:
        raise ValueError("Architect failed to produce output.")
    resp.plan = plan
    out = {"task_plan": resp}
    if "mcp_tools" in state:
        out["mcp_tools"] = state["mcp_tools"]
    return out


async def coder_agent(state: dict) -> dict:
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    task = steps[coder_state.current_step_idx]
    coder_tools = state.get("mcp_tools", [])
    if not coder_tools:
        return {"coder_state": coder_state, "status": "ERROR", "error": "No MCP tools available"}

    from langgraph.prebuilt import create_react_agent
    model = init_chat_model("groq:openai/gpt-oss-20b")
    react_agent = create_react_agent(model, coder_tools)

    tool_names = [getattr(t, "name", "unknown") for t in coder_tools]
    tools_info = f"AVAILABLE TOOLS: {', '.join(tool_names)}\n"

    system_prompt = load_instructions_file("agents/Web_App_Agent/coder_instructions.txt")
    user_prompt = (
        f"Task: {task.task_description}\n"
        f"File: {task.filepath}\n"
        f"{tools_info}Use write_file(path, content) to save changes."
    )

    # Chạy một bước
    await react_agent.ainvoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    })

    coder_state.current_step_idx += 1
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    return {"coder_state": coder_state, "status": "WORKING", "mcp_tools": state.get("mcp_tools")}


# ========== Graph Compiler ==========

def _get_web_graph():
    g = StateGraph(dict)
    g.add_node("planner", planner_agent)
    g.add_node("architect", architect_agent)
    g.add_node("coder", coder_agent)

    g.add_edge("planner", "architect")
    g.add_edge("architect", "coder")
    g.add_conditional_edges(
        "coder",
        lambda s: "END" if s.get("status") == "DONE" else "coder",
        {"END": END, "coder": "coder"}
    )

    g.set_entry_point("planner")
    return g.compile()
