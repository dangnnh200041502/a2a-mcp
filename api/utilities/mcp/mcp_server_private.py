from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
from pinecone_utils import get_vectorstore

load_dotenv()

# No LLM usage here; vector_search is a thin retrieval wrapper

# Initialize Private MCP server
mcp = FastMCP("rag-private-tools")

@mcp.tool()
async def vector_search(query: str, chat_history: list | None = None):
    """Vector search — similarity retrieval for private company documents."""
    # Force top_k=2 regardless of LLM input
    effective_query = query
    k = 2
    try:
        # Lazy initialization of vectorstore
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        try:
            docs = retriever.invoke(effective_query)
        except Exception:
            docs = retriever.get_relevant_documents(effective_query)
    except Exception as e:
        return {"effective_query": effective_query, "context": "", "error": f"retriever_error: {e}"}

    contents = []
    for d in docs:
        text = getattr(d, "page_content", getattr(d, "content", str(d)))
        if not isinstance(text, str):
            text = str(text)
        if text:
            contents.append(text.strip())
    joined = "\n\n".join(contents)
    context = joined[:8000]
    
    return {"effective_query": effective_query, "context": context}

if __name__ == "__main__":
    mcp.run(transport="stdio")

