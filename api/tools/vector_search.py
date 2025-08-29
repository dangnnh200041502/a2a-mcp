from __future__ import annotations

from typing import Any, Dict, List
import os
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import torch

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
_model = genai.GenerativeModel('gemini-1.5-flash')


class VectorSearchTool:
    name = "vector_search"

    def __init__(self) -> None:
        pass

    def _expand_query(self, original_query: str, chat_history: List[Dict[str, Any]] | None) -> List[str]:
        """Expand query using Gemini model."""
        hist_lines: List[str] = []
        for m in (chat_history or []):
            role = m.get("role")
            content = m.get("content", "")
            if role == "human":
                hist_lines.append(f"User: {content}")
            elif role == "ai":
                hist_lines.append(f"Assistant: {content}")
        hist_text = "\n".join(hist_lines)

        prompt = f"""
You are a query planner.

Task:
- Rewrite the user's latest question into a minimal set of standalone search queries.
- Resolve pronouns using the chat history if present.
- If the question contains multiple distinct asks (e.g., founded year and headquarters), output one query per ask.
- If it is a single ask, output exactly one query.
- Keep each query concise and unambiguous (include the entity name explicitly).

Chat History (optional):
{hist_text}

Latest Question: {original_query}

Output rules:
- Write one query per line.
- Do not add numbering or extra text.
"""
        try:
            response = _model.generate_content(prompt)
            lines = [l.strip() for l in (response.text or "").split("\n") if l.strip()]
            return lines if lines else [original_query]
        except Exception as e:
            print(f"Expansion error: {e}")
            return [original_query]

    def _fuse_results(self, query_results: List[List[Dict[str, Any]]], rrf_k: float = 60.0) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion for combining multiple query results."""
        document_scores: Dict[str, Dict[str, Any]] = {}
        for query_idx, results in enumerate(query_results):
            for rank, doc_score in enumerate(results):
                content_str = doc_score.get('content', '') or ''
                doc_key = content_str[:100] if content_str else f"doc_{query_idx}_{rank}"
                if doc_key not in document_scores:
                    document_scores[doc_key] = {
                        'content': content_str,
                        'metadata': doc_score.get('metadata', {}),
                        'rrf_score': 0.0,
                        'appearances': 0,
                        'source_queries': []
                    }
                rrf_score = 1.0 / (rrf_k + rank)
                document_scores[doc_key]['rrf_score'] += rrf_score
                document_scores[doc_key]['appearances'] += 1
                document_scores[doc_key]['source_queries'].append(doc_score.get('source_query', f'query_{query_idx}'))

        fused: List[Dict[str, Any]] = []
        for _, info in document_scores.items():
            content = info['content'] or f"Document from {info['appearances']} queries"
            fused.append({
                'content': content,
                'metadata': info['metadata'],
                'score': info['rrf_score'],
                'source_query': "combined",
                'rank': 0
            })
        fused.sort(key=lambda x: x['score'], reverse=True)
        for i, d in enumerate(fused):
            d['rank'] = i
        return fused

    def _rerank_documents(self, query: str, docs: List[dict], threshold: float, top_k: int) -> List[dict]:
        """Rerank documents using cross-encoder model."""
        if not docs:
            return docs[:top_k]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        except Exception:
            model = None

        if model is None:
            return docs[:top_k]

        pairs = [(query, d.get("content", "")) for d in docs]
        scores = model.predict(pairs)
        for d, s in zip(docs, scores):
            d["score"] = float(s)
        docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        reranked = [d for d in docs if d.get("score", 0.0) >= threshold] or docs[:top_k]
        return reranked[:top_k]

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        question: str = params.get("query", "")
        chat_history = params.get("chat_history")

        # 1) Expansion
        queries = self._expand_query(question, chat_history)

        # 2) Retrieve per query
        from pinecone_utils import vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        all_results: List[List[Dict[str, Any]]] = []
        for q in queries:
            try:
                docs = retriever.invoke(q)
            except Exception:
                docs = retriever.get_relevant_documents(q)
            rows: List[Dict[str, Any]] = []
            for i, doc in enumerate(docs):
                content = getattr(doc, 'page_content', getattr(doc, 'content', str(doc))) or ""
                rows.append({
                    "content": content,
                    "metadata": getattr(doc, 'metadata', {}),
                    "score": 1.0 - (i * 0.1),
                    "source_query": q,
                    "rank": i,
                })
            all_results.append(rows)

        # 3) Fusion
        fused = self._fuse_results(all_results, float(params.get("rrf_k", 60.0)))

        # 4) Rerank
        reranked = self._rerank_documents(
            question,
            fused,
            float(params.get("threshold", 0.5)),
            int(params.get("top_k", 10))
        )

        meta: Dict[str, Any] = {
            "expanded_queries": queries[1:] if len(queries) > 1 else [],
            "retrieved_docs_count": sum(len(r) for r in all_results),
            "fused_docs_count": len(fused),
            "final_docs_count": len(reranked),
        }

        return {
            "effective_query": question,
            "final_documents": reranked,
            "meta": meta,
        }


