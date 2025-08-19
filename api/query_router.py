"""Router phân loại câu hỏi 0/1 bằng Gemini Flash 1.5.

Ý tưởng:
- Trả JSON: {"route": 0|1, "confidence": float, "reason": str}
- 0: Câu hỏi ngoài phạm vi (real-time/thời tiết, đồ ăn/đồ uống, luật/chính trị, thể thao, v.v.)
- 1: Câu hỏi còn lại → đưa vào luồng RAG.

Hàm chính: route_query(question) → Dict, được gọi ở `main.py` trước khi quyết định có chạy RAG hay không.
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

_MODEL_ID = os.getenv("GEMINI_ROUTER_MODEL", "gemini-1.5-flash")

_SYSTEM_INSTRUCTIONS = (
    """
You are a high-precision query router. Classify the user's question into one of two routes:

1 = RAG route (the question is likely about entities, facts, or content stored in our knowledge base such as company names, products, projects, dates, laws, or documents that require retrieval/citations).
0 = Direct LLM route (the question is general knowledge, chit-chat, opinions, coding help not tied to our internal docs, or otherwise doesn't need retrieval).

Return strict JSON with keys: {"route": 0|1, "confidence": float in [0,1], "reason": string <= 200 chars}.

Guidelines:
- Route 1 if the query asks about specific entities likely present in our KB (e.g., GreenGrow Innovations, EcoHarvest System, company history, founders, dates, PDFs we indexed).
- Route 0 for generic questions (e.g., weather, capitals, math, programming tasks) or when evidence suggests it's outside domain.
- Be conservative but prefer 1 if uncertain and the query mentions a proper noun from a company/product-like phrase.
"""
).strip()


def _safe_parse_json(text: str) -> Dict[str, Any]:
    """Cố gắng parse JSON từ đầu ra của Gemini.

    - Trường hợp model trả chuỗi kèm phần thừa, hàm sẽ cố gắng trích JSON giữa cặp ngoặc nhọn.
    - Luôn trả về dict hợp lệ (fallback route=1) để không làm vỡ luồng API.
    """
    try:
        return json.loads(text)
    except Exception:
        # Try to extract JSON block if wrapped
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return {"route": 1, "confidence": 0.5, "reason": "fallback_parse"}


def route_query(question: str) -> Dict[str, Any]:
    """Phân loại câu hỏi sang RAG (1) hoặc Direct (0) bằng Gemini.

    Trả về dict: {"route": 0|1, "confidence": float, "reason": str}
    - Được dùng bởi `/chat` và `/chat-advanced` trong `main.py`.
    - Nếu route=0: API trả câu xin lỗi lịch sự và không lưu session/history.
    """
    prompt = (
        f"{_SYSTEM_INSTRUCTIONS}\n\nUSER_QUESTION:\n{question}\n\n"
        "Return JSON only."
    )

    try:
        model = genai.GenerativeModel(_MODEL_ID)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        data = _safe_parse_json(text)
        # Validate
        route = int(data.get("route", 1))
        if route not in (0, 1):
            route = 1
        confidence = float(data.get("confidence", 0.5))
        reason = str(data.get("reason", ""))[:200]
        return {"route": route, "confidence": confidence, "reason": reason}
    except Exception as e:
        return {"route": 1, "confidence": 0.5, "reason": f"router_error: {e}"}

