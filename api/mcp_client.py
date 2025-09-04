from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
import traceback
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = []

    # connect to the MCP server
    async def connect_to_server(self, server_script_path: str = "mcp_server.py"):
        try:
            is_python = server_script_path.endswith(".py")
            if not is_python:
                raise ValueError("Server script must be a .py file")

            server_params = StdioServerParameters(
                command="python", args=[server_script_path], env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            print("Connected to MCP server")

            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]

            print(f"Available tools: {[tool['name'] for tool in self.tools]}")

            return True

        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    # get mcp tool list
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            print(f"Error getting MCP tools: {e}")
            raise

    # call a specific tool
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server and return a simple Python object.
        Minimal unwrapping only: if MCP returns a single JSON or text part, return that value.
        """
        try:
            if not self.session:
                raise RuntimeError("Not connected to MCP server")

            result = await self.session.call_tool(tool_name, args)
            content = result.content
            # Minimal unwrap of typical MCP content parts
            if isinstance(content, list) and content:
                first = content[0]
                # Handle dict-shaped content part
                if isinstance(first, dict):
                    ctype = first.get("type")
                    if ctype == "json":
                        return first.get("data")
                    if ctype == "text":
                        txt = first.get("text")
                        try:
                            return json.loads(txt)
                        except Exception:
                            return txt
                    # Unknown dict shape — return inner if present
                    inner = first.get("data") or first.get("text")
                    if isinstance(inner, str):
                        try:
                            return json.loads(inner)
                        except Exception:
                            return inner
                    return inner or first
                # Handle FastMCP content classes (e.g., TextContent/JsonContent)
                if hasattr(first, "type"):
                    ctype = getattr(first, "type", None)
                    if ctype == "json" and hasattr(first, "data"):
                        return getattr(first, "data")
                    if ctype == "text":
                        txt = None
                        if hasattr(first, "text"):
                            txt = getattr(first, "text")
                        elif hasattr(first, "value"):
                            txt = getattr(first, "value")
                        if isinstance(txt, str):
                            try:
                                return json.loads(txt)
                            except Exception:
                                return txt
                # Fallback: return as-is
                return first
            # Handle single content object (not a list)
            if hasattr(content, "type"):
                ctype = getattr(content, "type", None)
                if ctype == "json" and hasattr(content, "data"):
                    return getattr(content, "data")
                if ctype == "text":
                    txt = None
                    if hasattr(content, "text"):
                        txt = getattr(content, "text")
                    elif hasattr(content, "value"):
                        txt = getattr(content, "value")
                    if isinstance(txt, str):
                        try:
                            return json.loads(txt)
                        except Exception:
                            return txt
                    return txt
            # If plain string, try JSON else return text
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except Exception:
                    return content
            return content
        except Exception as e:
            print(f"Error calling tool {tool_name}: {e}")
            raise

    # cleanup
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            print("Disconnected from MCP server")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise
