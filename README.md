# Agentic RAG Orchestration (A2A + LangGraph + MCP)

A lightweight multi-agent system built with A2A (Agent-to-Agent), LangGraph, and MCP. It includes:
- Host Agent (orchestrator) using LangGraph StateGraph
- Secret Agent (confidential info retrieval via MCP tools)
- Web Agent (planner → architect → coder pipeline to generate static web assets)
- FastAPI app with synchronous chat and SSE streaming endpoints

![Overview](docs/image.png)

## Features
- A2A (Agent-to-Agent) discovery and delegation
- LangGraph StateGraph with tool routing
- MCP-hosted tools for both Host and Secret/Web agents
- Optional Postgres persistence helpers (logs, documents)

![Agents](docs/image1.png)

---

## Project Structure

```
.
├─ api/
│  ├─ agents/
│  │  ├─ host_agent/
│  │  │  ├─ __init__.py
│  │  │  ├─ __main__.py
│  │  │  ├─ agent.py
│  │  │  ├─ agent_executor.py
│  │  │  ├─ description.txt
│  │  │  └─ instructions.txt
│  │  ├─ Secret_Agent/
│  │  │  ├─ __init__.py
│  │  │  ├─ __main__.py
│  │  │  ├─ agent.py
│  │  │  ├─ agent_executor.py
│  │  │  ├─ description.txt
│  │  │  └─ instructions.txt
│  │  └─ Web_App_Agent/
│  │     ├─ __init__.py
│  │     ├─ __main__.py
│  │     ├─ agent.py
│  │     ├─ agent_executor.py
│  │     ├─ states.py
│  │     ├─ planner_instructions.txt
│  │     ├─ architect_instructions.txt
│  │     ├─ coder_instructions.txt
│  │     ├─ instructions.txt
│  │     └─ description.txt
│  ├─ utilities/
│  │  ├─ a2a/
│  │  │  ├─ agent_connect.py
│  │  │  ├─ agent_discovery.py
│  │  │  └─ agent_registry.json
│  │  ├─ common/
│  │  │  └─ file_loader.py
│  │  └─ mcp/
│  │     ├─ mcp_server_host.py
│  │     ├─ mcp_server_public.py
│  │     ├─ mcp_server_private.py
│  │     └─ mcp_server_web.py
│  ├─ generated_project_owner_information/
│  │  ├─ index.html
│  │  ├─ styles.css
│  │  ├─ script.js
│  │  └─ README.md
│  ├─ main.py
│  ├─ pydantic_models.py
│  ├─ pinecone_utils.py
│  ├─ db_utils.py
│  └─ app.log
├─ README.md
├─ pyproject.toml
```

---

## Prerequisites
- Python 3.10+
- Pip
- A Groq API key exported as environment variable `GROQ_API_KEY`
- (Optional) PostgreSQL if you want to persist logs/documents

Create a `.env` with:
```
GROQ_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb
```

Install dependencies:
```bash
pip install -e .
```

---

## Cách khởi động

Chạy các lệnh sau (theo đúng thứ tự hoặc riêng lẻ tùy nhu cầu):

1/ python3 -m agents.Web_App_Agent

2/ python3 -m agents.Secret_Agent

3/ python3 -m agents.host_agent

4/ uvicorn main:app --reload

---

## Streaming via SSE
Endpoint: (removed; use only the chat endpoint)

---

## Agents

### Host Agent
- Orchestrates requests using LangGraph with A2A-style routing.
- Binds MCP tools exposed by `utilities/mcp/mcp_server_host.py`.
- Decides to answer directly or delegate to a child agent.

### Secret Agent
- Uses MCP tools from `mcp_server_public.py` / `mcp_server_private.py` to retrieve owner/confidential info.
- Returns a clean JSON-like response in the final message.

### Web Agent
- Planner → Architect → Coder pipeline using LangGraph.
- Generates static files (`index.html`, `styles.css`, `script.js`).
- Output example folder: `api/generated_project_owner_information/`.

---

## Test Cases
Use these four prompts to verify behavior:

1) When was GreenGrow Innovations founded?
2) Where it is headquartered?
3) Create a simple calculator web application.
4) Please build me a web to answer this question and show on it, what is the owner name and his email ?

Suggested flows:
- (1) and (2): Direct Q&A or Secret Agent based on tool availability.
- (3): Delegates to Web Agent to generate the minimal calculator site.
- (4): First retrieve owner info via Secret Agent, then create a web page (may require two steps depending on your orchestration).

---

## Serving the Generated Web
The generated files live under `api/generated_project_owner_information/`. Open `index.html` directly in a browser, or serve the directory with a simple static server, for example:

```bash
# Option A: Python simple HTTP server
cd api/generated_project_owner_information
python -m http.server 8080
# then visit http://127.0.0.1:8080/index.html
```

If you want FastAPI to serve this folder under `/generated_project`, add a StaticFiles mount in `api/main.py`:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/generated_project", StaticFiles(directory="api/generated_project_owner_information"), name="generated")
```
Then open:
```
http://127.0.0.1:8000/generated_project/index.html
```

---

## Notes
- If tools fail with XML-like tool call output, ensure prompts instruct JSON tool calls only.
- For persistent memory, wire a LangGraph checkpointer (e.g., MemorySaver, Postgres) as needed.
- The repo includes log files for development traceability; exclude them for production.
# a2a-mcp
# a2a-mcp
