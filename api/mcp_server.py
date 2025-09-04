from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from pinecone_utils import vectorstore

load_dotenv()

# Initialize LLM (Groq) for contextualization-only
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

# Initialize MCP server
mcp = FastMCP("rag-tools")

@mcp.tool()
async def calculator(expression: str):
    """Calculate arithmetic expressions with only +, -, *, / operations."""
    import re as _re
    safe_pattern = _re.compile(r"^[\s\d\+\-\*/\(\)\.]+$")
    if not safe_pattern.match((expression or "")):
        return {"error": "unsupported expression", "result": None}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e), "result": None}

@mcp.tool()
async def weather(question: str):
    """Get weather information (toy)."""
    hints = ("weather", "thoi tiet", "thời tiết", "forecast", "rain", "sunny", "temperature")
    text = (question or "").lower()
    if any(h in text for h in hints):
        return {"tool": "weather", "location": None, "status": "sunny"}
    return {"tool": "weather", "error": "not a weather question"}

@mcp.tool()
async def vector_search(query: str, chat_history: list | None = None, top_k: int = 2):
    """
    History-aware single-query retrieval (match notebook flow):
    1) Contextualize follow-up question into a standalone question using chat history
    2) Retrieve top_k docs from Pinecone retriever
    3) Return { "effective_query": str, "snippets": [str, ...] }
    """
    # Ensure vectorstore available
    try:
        from pinecone_utils import vectorstore as _vs
    except Exception as e:
        return {"effective_query": query, "snippets": [], "error": f"vectorstore import failed: {e}"}
    if _vs is None:
        return {"effective_query": query, "snippets": [], "error": "vectorstore_not_initialized"}

    # Build chat history as lines (support DB rows and role/content)
    lines: list[str] = []
    if chat_history and isinstance(chat_history, list):
        for m in chat_history:
            try:
                if isinstance(m, dict):
                    q = m.get("user_query") or m.get("question") or m.get("user")
                    a = m.get("ai_response") or m.get("answer") or m.get("response") or m.get("bot")
                    if q:
                        lines.append(f"User: {str(q)}")
                    if a:
                        lines.append(f"Assistant: {str(a)}")
                    role = (m.get("role") or "").lower()
                    content = m.get("content") or m.get("message") or m.get("text") or ""
                    if content:
                        if role in ("human", "user"):
                            lines.append(f"User: {content}")
                        elif role in ("ai", "assistant"):
                            lines.append(f"Assistant: {content}")
                elif isinstance(m, str):
                    lines.append(f"User: {m}")
                else:
                    lines.append(str(m))
            except Exception:
                continue
    hist_text = "\n".join(lines)

    # Contextualize prompt (exactly like notebook intent)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    prompt = (
        f"{contextualize_q_system_prompt}\n\n"
        f"Chat History:\n{hist_text}\n\n"
        f"User: {query}\n"
        f"Standalone question:"
    )
    try:
        resp = llm.invoke(prompt)
        effective_query = (getattr(resp, "content", str(resp)) or "").strip()
        # Use the first line and strip code fences if any
        effective_query = effective_query.strip('`').splitlines()[0].strip() if effective_query else (query or "")
        if not effective_query:
            effective_query = query
    except Exception:
        effective_query = query

    # Retrieve
    k = int(top_k) if isinstance(top_k, int) and top_k > 0 else 2
    try:
        retriever = _vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
        try:
            docs = retriever.invoke(effective_query)
        except Exception:
            docs = retriever.get_relevant_documents(effective_query)
    except Exception as e:
        return {"effective_query": effective_query, "snippets": [], "error": f"retriever_error: {e}"}

    # Build snippets
    snippets: list[str] = []
    for d in docs or []:
        content = getattr(d, "page_content", getattr(d, "content", str(d))) or ""
        if not isinstance(content, str):
            content = str(content)
        if len(content) > 400:
            content = content[:400] + "..."
        snippets.append(content)

    return {"effective_query": effective_query, "snippets": snippets}

if __name__ == "__main__":
    mcp.run(transport="stdio")
