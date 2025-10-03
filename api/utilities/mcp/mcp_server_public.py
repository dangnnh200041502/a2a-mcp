from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import requests

load_dotenv()

# Initialize Public MCP server
mcp = FastMCP("rag-public-tools")


@mcp.tool()
async def get_user_email(email: str, subject: str | None = None, message: str | None = None):
    """Get user email — notify owner via Pushover when user shares email."""
    token = os.getenv("PUSHOVER_API_TOKEN")
    user_key = os.getenv("PUSHOVER_USER_KEY")
    if not token or not user_key:
        return {"ok": False, "error": "Missing Pushover credentials in env."}

    title = subject or "New contact email from chat"
    body_parts = [f"Email: {email}"]
    if message:
        body_parts.append(f"Message: {message}")
    body = "\n".join(body_parts)

    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user_key,
                "title": title,
                "message": body,
            },
            timeout=10,
        )
        ok = resp.status_code == 200 and resp.json().get("status") == 1
        return {
            "ok": ok,
            "status_code": resp.status_code,
            "response": resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else resp.text,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Converted resources: expose as tools so clients can call directly
@mcp.tool()
async def author_name() -> str:
    """Author name — returns the owner's display name."""
    return "Nguyen Ngoc Hai Dang"


@mcp.tool()
async def author_email() -> str:
    """Author email — returns the owner's public email address."""
    return "ndang15022004@gmail.com"


if __name__ == "__main__":
    mcp.run(transport="stdio")


