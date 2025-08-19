from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DocumentScore:
    """Class để lưu thông tin document và score"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source_query: str
    rank: int

class ReciprocalRankFusion:
    """
    Implement Reciprocal Rank Fusion (RRF) để kết hợp kết quả từ nhiều queries
    """
    
    def __init__(self, k: float = 60.0):
        """
        Args:
            k: Constant để điều chỉnh ảnh hưởng của rank (mặc định 60.0)
        """
        self.k = k
    
    def fuse_results(self, query_results: List[List[DocumentScore]]) -> List[DocumentScore]:
        """
        Kết hợp kết quả từ nhiều queries sử dụng RRF
        
        Args:
            query_results: List các list DocumentScore từ mỗi query
            
        Returns:
            List DocumentScore đã được sắp xếp theo score RRF
        """
        # Dictionary để lưu tổng RRF score cho mỗi document
        document_scores = {}
        
        # Tính RRF score cho mỗi document
        for query_idx, results in enumerate(query_results):
            for rank, doc_score in enumerate(results):
                # Tạo key duy nhất cho document (dựa trên content)
                # Xử lý an toàn content
                if hasattr(doc_score, 'content') and doc_score.content:
                    content_str = str(doc_score.content)
                    doc_key = content_str[:100] if len(content_str) > 100 else content_str
                else:
                    # Fallback nếu không có content
                    doc_key = f"doc_{query_idx}_{rank}"
                
                if doc_key not in document_scores:
                    document_scores[doc_key] = {
                        'content': getattr(doc_score, 'content', ''),
                        'metadata': getattr(doc_score, 'metadata', {}),
                        'rrf_score': 0.0,
                        'appearances': 0,
                        'source_queries': []
                    }
                
                # Tính RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                document_scores[doc_key]['rrf_score'] += rrf_score
                document_scores[doc_key]['appearances'] += 1
                document_scores[doc_key]['source_queries'].append(getattr(doc_score, 'source_query', f'query_{query_idx}'))
        
        # Chuyển đổi thành list DocumentScore và sắp xếp theo RRF score
        fused_results = []
        for doc_key, doc_info in document_scores.items():
            # Đảm bảo content không rỗng
            content = doc_info['content']
            if not content:
                content = f"Document from {doc_info['appearances']} queries"
            
            fused_doc = DocumentScore(
                content=content,
                metadata=doc_info['metadata'],
                score=doc_info['rrf_score'],
                source_query=f"Combined from {doc_info['appearances']} queries",
                rank=0  # Sẽ được cập nhật sau
            )
            fused_results.append(fused_doc)
        
        # Sắp xếp theo RRF score giảm dần
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # Cập nhật rank
        for i, doc in enumerate(fused_results):
            doc.rank = i
        
        return fused_results
    
    def fuse_with_weights(self, query_results: List[List[DocumentScore]], 
                         weights: List[float] = None) -> List[DocumentScore]:
        """
        RRF với weights khác nhau cho mỗi query
        
        Args:
            query_results: List các list DocumentScore từ mỗi query
            weights: List weights cho mỗi query (mặc định: query gốc có weight cao hơn)
            
        Returns:
            List DocumentScore đã được sắp xếp theo weighted RRF score
        """
        if weights is None:
            # Mặc định: query gốc có weight cao hơn
            weights = [2.0] + [1.0] * (len(query_results) - 1)
        
        if len(weights) != len(query_results):
            raise ValueError("Số lượng weights phải bằng số lượng queries")
        
        # Dictionary để lưu tổng weighted RRF score
        document_scores = {}
        
        # Tính weighted RRF score
        for query_idx, (results, weight) in enumerate(zip(query_results, weights)):
            for rank, doc_score in enumerate(results):
                # Xử lý an toàn content
                if hasattr(doc_score, 'content') and doc_score.content:
                    content_str = str(doc_score.content)
                    doc_key = content_str[:100] if len(content_str) > 100 else content_str
                else:
                    # Fallback nếu không có content
                    doc_key = f"doc_{query_idx}_{rank}"
                
                if doc_key not in document_scores:
                    document_scores[doc_key] = {
                        'content': getattr(doc_score, 'content', ''),
                        'metadata': getattr(doc_score, 'metadata', {}),
                        'rrf_score': 0.0,
                        'appearances': 0,
                        'source_queries': []
                    }
                
                # Weighted RRF score
                rrf_score = weight / (self.k + rank)
                document_scores[doc_key]['rrf_score'] += rrf_score
                document_scores[doc_key]['appearances'] += 1
                document_scores[doc_key]['source_queries'].append(getattr(doc_score, 'source_query', f'query_{query_idx}'))
        
        # Chuyển đổi và sắp xếp
        fused_results = []
        for doc_key, doc_info in document_scores.items():
            # Đảm bảo content không rỗng
            content = doc_info['content']
            if not content:
                content = f"Document from {doc_info['appearances']} queries"
            
            fused_doc = DocumentScore(
                content=content,
                metadata=doc_info['metadata'],
                score=doc_info['rrf_score'],
                source_query=f"Weighted RRF from {doc_info['appearances']} queries",
                rank=0
            )
            fused_results.append(fused_doc)
        
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        for i, doc in enumerate(fused_results):
            doc.rank = i
        
        return fused_results
