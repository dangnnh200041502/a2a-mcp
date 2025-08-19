from typing import List
from sentence_transformers import CrossEncoder
from fusion_utils import DocumentScore
import torch

class CrossEncoderReranker:
    """
    Sử dụng Cross-Encoder để rerank kết quả từ RRF
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Tên model Cross-Encoder (mặc định: ms-marco-MiniLM-L-6-v2)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.cross_encoder = CrossEncoder(model_name, device=self.device)
            print(f"Cross-Encoder loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading Cross-Encoder: {e}")
            # Fallback to default model
            try:
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device)
                print("Fallback Cross-Encoder loaded successfully")
            except Exception as e2:
                print(f"Failed to load fallback Cross-Encoder: {e2}")
                self.cross_encoder = None
    
    def rerank_documents(self, query: str, documents: List[DocumentScore], 
                        top_k: int = 10) -> List[DocumentScore]:
        """
        Rerank documents sử dụng Cross-Encoder
        
        Args:
            query: Query gốc của người dùng
            documents: List documents từ RRF
            top_k: Số lượng documents trả về sau reranking
            
        Returns:
            List documents đã được rerank
        """
        if self.cross_encoder is None:
            print("Cross-Encoder not available, returning original documents")
            return documents[:top_k]
        
        if not documents:
            return []
        
        try:
            # Tạo pairs (query, document) cho Cross-Encoder
            pairs = [(query, doc.content) for doc in documents]
            
            # Tính scores sử dụng Cross-Encoder
            scores = self.cross_encoder.predict(pairs)
            
            # Kết hợp scores với documents
            scored_docs = []
            for doc, score in zip(documents, scores):
                # Cập nhật score mới từ Cross-Encoder
                doc.score = float(score)
                scored_docs.append(doc)
            
            # Sắp xếp theo score mới
            scored_docs.sort(key=lambda x: x.score, reverse=True)
            
            # Trả về top_k documents
            return scored_docs[:top_k]
            
        except Exception as e:
            print(f"Error in Cross-Encoder reranking: {e}")
            return documents[:top_k]
    
    def rerank_with_threshold(self, query: str, documents: List[DocumentScore], 
                             threshold: float = 0.5, top_k: int = 10) -> List[DocumentScore]:
        """
        Rerank documents với threshold để lọc documents có score thấp
        
        Args:
            query: Query gốc của người dùng
            documents: List documents từ RRF
            threshold: Ngưỡng score tối thiểu
            top_k: Số lượng documents trả về sau reranking
            
        Returns:
            List documents đã được rerank và lọc theo threshold
        """
        reranked_docs = self.rerank_documents(query, documents, top_k)
        
        # Lọc documents theo threshold
        filtered_docs = [doc for doc in reranked_docs if doc.score >= threshold]
        
        if not filtered_docs:
            print(f"No documents meet threshold {threshold}, returning top documents")
            return reranked_docs[:top_k]
        
        return filtered_docs
    
    def batch_rerank(self, queries: List[str], all_documents: List[List[DocumentScore]], 
                     top_k: int = 10) -> List[List[DocumentScore]]:
        """
        Rerank documents cho nhiều queries cùng lúc
        
        Args:
            queries: List các queries
            all_documents: List các list documents tương ứng với mỗi query
            top_k: Số lượng documents trả về cho mỗi query
            
        Returns:
            List các list documents đã được rerank
        """
        if len(queries) != len(all_documents):
            raise ValueError("Số lượng queries phải bằng số lượng document lists")
        
        reranked_results = []
        
        for query, documents in zip(queries, all_documents):
            reranked_docs = self.rerank_documents(query, documents, top_k)
            reranked_results.append(reranked_docs)
        
        return reranked_results
    
    def get_model_info(self) -> dict:
        """
        Lấy thông tin về model Cross-Encoder
        """
        if self.cross_encoder is None:
            return {"status": "not_loaded", "device": self.device}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "status": "loaded",
            "max_length": getattr(self.cross_encoder, 'max_length', 'unknown')
        }
