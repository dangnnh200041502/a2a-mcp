from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from typing import Any, Dict, Optional, List
from collections.abc import AsyncIterable
import asyncio
from dotenv import load_dotenv
from utilities.common.file_loader import load_instructions_file

load_dotenv()


class SecretAgent:
    """Secret agent using MCP tools via MultiServerMCPClient with persistent connection."""

    def __init__(self) -> None:
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self.description = load_instructions_file("agents/Secret_Agent/description.txt")
        self.instructions = load_instructions_file("agents/Secret_Agent/instructions.txt")

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._mcp_client = MultiServerMCPClient(
                {
                    "public": {
                        "command": "python",
                        "args": ["-m", "utilities.mcp.mcp_server_public"],
                        "transport": "stdio",
                    },
                    "private": {
                        "command": "python",
                        "args": ["-m", "utilities.mcp.mcp_server_private"],
                        "transport": "stdio",
                    },
                }
            )
            self._mcp_tools = await self._mcp_client.get_tools()
            self._initialized = True

    async def invoke(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        await self.initialize()

        if not query or not isinstance(query, str):
            yield {"is_task_complete": True, "content": "Invalid input query"}
            return

        yield {"is_task_complete": False, "updates": "Processing..."}

        try:
            result = await self.process_query(query)

            yield {
                "is_task_complete": True,
                "content": result.get("result", "No result generated")
            }
        except Exception as e:
            yield {"is_task_complete": True, "content": f"Error in Secret Agent: {str(e)}"}

    async def process_query(self, question: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        await self.initialize()

        model = init_chat_model("groq:openai/gpt-oss-20b")
        model_with_tools = model.bind_tools(self._mcp_tools)
        tool_node = ToolNode(self._mcp_tools)

        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END

        async def call_model(state: MessagesState):
            messages = state["messages"]
            response = await model_with_tools.ainvoke(messages)
            return {"messages": [response]}

        # Build the graph following LangGraph MCP documentation pattern
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            should_continue,
        )
        builder.add_edge("tools", "call_model")
        graph = builder.compile()

        messages = [{"role": "user", "content": question}]
        out = await graph.ainvoke({"messages": messages})

        final = out.get("messages", [])
        answer = getattr(final[-1], "content", str(final[-1])) if final else "No response generated"

        # Always return a clean dict
        return {
            "tool_used": "secret_agent",
            "result": str(answer).strip(),
            "success": True
        }

    async def cleanup(self) -> None:
        if hasattr(self, "_mcp_client") and self._mcp_client:
            await self._mcp_client.aclose()
        return None
