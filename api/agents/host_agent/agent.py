from collections.abc import AsyncIterable
import json
from typing import Any
from uuid import uuid4

from utilities.common.file_loader import load_instructions_file

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from rich import print as rprint
from rich.syntax import Syntax
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
load_dotenv()
    

class HostAgent:
    """
    Host Agent orchestration using LangGraph
    - Discovers A2A agents dynamically
    - Delegates query exactly once to the correct agent
    """

    def __init__(self):
        self.system_instruction = load_instructions_file("agents/host_agent/instructions.txt")
        self.description = load_instructions_file("agents/host_agent/description.txt")
        self._checkpointer = MemorySaver()
        self._graph = None

    async def create(self):
        await self._build_graph()

    # ---------------- Build Graph ----------------

    async def _build_graph(self):
        model = init_chat_model("groq:openai/gpt-oss-120b")

        # Load MCP tools from host router server
        mcp_client = MultiServerMCPClient({
            "host_router": {
                "command": "python",
                "args": ["-m", "utilities.mcp.mcp_server_host"],
                "transport": "stdio",
            }
        })
        mcp_tools = await mcp_client.get_tools()

        model_with_tools = model.bind_tools(mcp_tools)
        tool_node = ToolNode(mcp_tools)

        async def call_model(state: MessagesState):
            msgs = state["messages"]
            sys = {"role": "system", "content": self.system_instruction}
            response = await model_with_tools.ainvoke([sys, *msgs])
            return {"messages": [response]}

        def should_continue(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        def route_after_tools(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "content") and "delegated" in str(last.content):
                return END
            return "call_model"

        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_node)
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", should_continue)
        builder.add_conditional_edges("tools", route_after_tools)

        self._graph = builder.compile(checkpointer=self._checkpointer)

    # ---------------- Invoke ----------------
    
    async def invoke(self, query: str, session_id: str) -> AsyncIterable[dict]:
        effective_session_id = session_id if session_id else uuid4().hex

        if self._graph is None:
            await self._build_graph()

        config = {"configurable": {"thread_id": effective_session_id}}

        yield {
            "is_task_complete": False,
            "updates": "Processing request...",
            "session_id": effective_session_id
        }

        try:
            messages = [{"role": "user", "content": query}]
            out = await self._graph.ainvoke({"messages": messages}, config=config)

            final = out.get("messages", [])
            answer = getattr(final[-1], "content", str(final[-1])) if final else "No response"

            # Normalize result
            parsed = None
            try:
                if isinstance(answer, str):
                    parsed = json.loads(answer)
                elif isinstance(answer, dict):
                    parsed = answer
            except Exception:
                parsed = {"result": str(answer)}

            if not parsed:
                parsed = {"result": str(answer)}
                
            yield {
                "is_task_complete": True,
                "content": parsed.get("result", parsed),
                "session_id": effective_session_id
                }
        except Exception as e:
            yield {
                "is_task_complete": True,
                "content": f"Error: {str(e)}",
                "session_id": effective_session_id
                }


def print_json_response(response: Any, title: str) -> None:
    print(f"\n=== {title} ===")
    try:
        if hasattr(response, "root"):
            data = response.root.model_dump(mode="json", exclude_none=True)
        else:
            data = response.model_dump(mode="json", exclude_none=True)

        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        rprint(syntax)
    except Exception as e:
        rprint(f"[red bold]Error printing JSON:[/red bold] {e}")
        rprint(repr(response))