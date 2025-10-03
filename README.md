# A2A + LangGraph + MCP — Agentic Orchestration

A compact multi‑agent system with: a Host Orchestrator, a Confidential‑Info agent, and a Web App agent. Orchestration runs on LangGraph; tools are exposed via MCP. A FastAPI service provides a simple chat API.

![Overview](docs/image.png)

## Key Features
- A2A discovery and delegation (Host → child agents)
- LangGraph StateGraph with tool routing
- MCP tool servers for host and domain agents
- Optional persistence helpers (logs, documents)

![Agents](docs/image1.png)

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

## Prerequisites
- Python 3.10+
- A Groq API key in `.env`
- (Optional) PostgreSQL for persistence

Create `.env` in full path: `agentic_mcp-main/api/.env`:
```env
GROQ_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb
```

Install dependencies (either is fine):
```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Run (Windows/macOS/Linux)
Open three terminals, and in each one change directory to the full path `agentic_mcp-main/api/` first:
```bash
cd agentic_mcp-main/api
```

1) Start Web App Agent (default: http://localhost:10000)
```bash
python -m agents.Web_App_Agent --host localhost --port 10000
```

2) Start Secret Agent (default: http://localhost:10003)
```bash
python -m agents.Secret_Agent --host localhost --port 10003
```

3) Start FastAPI service
```bash
uvicorn main:app --host localhost --port 8000 --reload
```

The Host Orchestrator is embedded in the FastAPI app and delegates to child agents discovered from full path `agentic_mcp-main/api/utilities/a2a/agent_registry.json`:
```json
[
  "http://localhost:10000",
  "http://localhost:10003"
]
```

### Chat API
- POST `/chat` — single‑turn chat routed by the Host agent

Request body:
```json
{ "question": "Create a simple calculator web app", "session_id": "optional" }
```

Response body:
```json
{ "answer": "...", "session_id": "..." }
```

## Streaming via SSE
Endpoint: (removed; use only the chat endpoint)

## Serving the Generated Web
The generated files live under full path `agentic_mcp-main/api/generated_project_owner_information/`. You can open `index.html` directly, or serve the directory:

```bash
# Option A: Python simple HTTP server
cd agentic_mcp-main/api/generated_project_owner_information
python -m http.server 8080
# then visit http://127.0.0.1:8080/index.html
```

If you want FastAPI to serve this folder under `/generated_project`, add a StaticFiles mount in `agentic_mcp-main/api/main.py`:
```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Absolute path to the generated folder from this file's location
GENERATED_DIR = Path(__file__).parent / "generated_project_owner_information"
app.mount("/generated_project", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
```
Then open:
```
http://127.0.0.1:8000/generated_project/index.html
```

## Quick Test Prompts
- When was GreenGrow Innovations founded?
- Where is it headquartered?
- Create a simple calculator web application.
- Build a web page that shows the owner name and email.

## Troubleshooting (Windows)
- Use UTF‑8 in terminals (prevents tool encoding errors):
  - `chcp 65001`
  - `$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8`
  - `setx PYTHONUTF8 1` and `setx PYTHONIOENCODING utf-8`
- Ensure both child agents are running and ports match `agent_registry.json`.
- Always run commands from the `api/` directory so relative paths resolve.
