## RAG System (FastAPI · Pinecone · Gemini · Groq)

README dành cho GitHub: mô tả tổng quát hệ thống RAG, cách cài đặt/chạy Backend (FastAPI) và App (Streamlit), cùng cấu trúc thư mục.

![Architecture](system.png)

## Tính năng chính (tóm tắt)
- Semantic Chunking (LangChain SemanticChunker)
- Embeddings (all-MiniLM-L6-v2) → Pinecone
- Query Expansion (Gemini 1.5 Flash)
- RRF Fusion + Cross‑Encoder Rerank
- History‑aware (contextualize trước khi truy hồi)
- Query Router 0/1 (Gemini + guardrails)

## Cấu trúc thư mục
```
rag/
├─ api/
│  ├─ main.py                # FastAPI endpoints & orchestration
│  ├─ query_router.py        # Router 0/1 (Gemini JSON + guardrails)
│  ├─ langchain_utils.py     # RAG chains & AdvancedRAGPipeline
│  ├─ pinecone_utils.py      # Load/Clean/Semantic chunking/Index to Pinecone
│  ├─ query_expansion.py     # Expand query (3 biến thể với Gemini)
│  ├─ fusion_utils.py        # DocumentScore + RRF
│  ├─ rerank_utils.py        # Cross‑Encoder reranking
│  ├─ db_utils.py            # PostgreSQL helpers (history, docs)
│  ├─ pydantic_models.py     # Pydantic models
│  └─ requirements.txt
├─ app/
│  ├─ streamlit_app.py       # UI chính
│  ├─ sidebar.py             # Upload/List/Delete tài liệu
│  ├─ chat_interface.py      # Chat UI + hiển thị kết quả
│  └─ api_utils.py           # Gọi API /chat, /upload-doc, ...
├─ docs/                     # Tài liệu mẫu (pdf/docx)
├─ system.png                # Sơ đồ kiến trúc
└─ README.md
```

## Cài đặt nhanh
```
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
```
Tạo file `.env` tại thư mục `rag/`:
```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
GEMINI_ROUTER_MODEL=gemini-1.5-flash
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=rag
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

## Chạy Backend (FastAPI)
```
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Swagger UI: `http://localhost:8000/docs`

## Chạy App (Streamlit)
Mặc định `app/api_utils.py` trỏ tới `http://localhost:8000`. Khởi chạy UI:
```
streamlit run app/streamlit_app.py
```

## Endpoints chính
- POST `/chat` – RAG cơ bản (history‑aware → router → retriever + QA)
- POST `/chat-advanced` – Advanced pipeline (expand → multi‑retrieval → RRF → rerank → LLM)
- POST `/upload-doc`, GET `/list-docs`, POST `/delete-doc`

## Ghi chú
- Route 0: trả lời lịch sự và KHÔNG lưu session/log.
- Có thể tinh chỉnh tham số retrieval/rerank trong advanced pipeline.
# rag_system
# rag_system
