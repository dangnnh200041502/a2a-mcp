from __future__ import annotations

from typing import Dict, Any, List, Dict as TDict


class DocumentScore:
    def __init__(self, content: str, metadata: TDict[str, Any], score: float, source_query: str, rank: int):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.source_query = source_query
        self.rank = rank


class ReciprocalRankFusion:
    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse_results(self, query_results: List[List[DocumentScore]]) -> List[DocumentScore]:
        document_scores: TDict[str, TDict[str, Any]] = {}
        for query_idx, results in enumerate(query_results):
            for rank, doc_score in enumerate(results):
                content_str = getattr(doc_score, 'content', '') or ''
                doc_key = content_str[:100] if content_str else f"doc_{query_idx}_{rank}"
                if doc_key not in document_scores:
                    document_scores[doc_key] = {
                        'content': content_str,
                        'metadata': getattr(doc_score, 'metadata', {}),
                        'rrf_score': 0.0,
                        'appearances': 0,
                        'source_queries': []
                    }
                rrf_score = 1.0 / (self.k + rank)
                document_scores[doc_key]['rrf_score'] += rrf_score
                document_scores[doc_key]['appearances'] += 1
                document_scores[doc_key]['source_queries'].append(getattr(doc_score, 'source_query', f'query_{query_idx}'))
        fused: List[DocumentScore] = []
        for _, info in document_scores.items():
            content = info['content'] or f"Document from {info['appearances']} queries"
            fused.append(DocumentScore(content, info['metadata'], info['rrf_score'], "combined", 0))
        fused.sort(key=lambda x: x.score, reverse=True)
        for i, d in enumerate(fused):
            d.rank = i
        return fused


class FusionTool:
    name = "fusion"

    def looks_like(self, question: str) -> bool:
        return False

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        queries: List[str] = params.get("queries", [])
        # normalize dicts into DocumentScore objects
        raw: List[List[Any]] = params.get("retrieved", [])
        retrieved: List[List[DocumentScore]] = []
        for lst in raw:
            norm: List[DocumentScore] = []
            for d in lst:
                if isinstance(d, DocumentScore):
                    norm.append(d)
                else:
                    norm.append(DocumentScore(
                        content=d.get("content", ""),
                        metadata=d.get("metadata", {}),
                        score=float(d.get("score", 0.0)),
                        source_query=d.get("source_query", ""),
                        rank=int(d.get("rank", 0)),
                    ))
            retrieved.append(norm)
        rrf_k: float = float(params.get("rrf_k", 60.0))
        rrf = ReciprocalRankFusion(k=rrf_k)
        fused = rrf.fuse_results(retrieved)
        # return as dicts
        return {"fused": [
            {"content": d.content, "metadata": d.metadata, "score": d.score, "source_query": d.source_query, "rank": d.rank}
            for d in fused
        ]}


