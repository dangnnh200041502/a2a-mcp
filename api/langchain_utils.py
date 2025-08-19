"""Tiện ích RAG và Advanced RAG Pipeline.

Nội dung chính:
- get_rag_chain(): Tạo chain RAG cơ bản có history-aware retriever.
- AdvancedRAGPipeline: Pipeline nâng cao gồm: contextualize → (multi-query) retrieve → RRF → rerank → LLM.
- context hóa câu hỏi (formulate) được dùng cả trong routing ở `main.py` (để tránh hiểu sai đại từ như "it").
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List, Dict, Any, Optional
import os
from pinecone_utils import vectorstore
from query_expansion import get_all_queries
from fusion_utils import DocumentScore, ReciprocalRankFusion
from rerank_utils import CrossEncoderReranker

# Legacy retriever (giữ lại để tương thích ngược)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain():
    """Tạo RAG chain cơ bản.

    - Sử dụng history-aware retriever để viết lại câu hỏi theo history.
    - Dùng "stuff" chain để nhét context vào prompt và trả lời.
    """
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b"
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain

 

def get_advanced_rag_pipeline(
    retrieval_k: int = 5,
    rrf_k: float = 60.0,
    rerank_top_k: int = 10,
    rerank_threshold: float = 0.5
):
    """
    Tạo Advanced RAG Pipeline với toàn bộ flow: Multiple Query → RRF → Rerank → LLM
    
    Args:
        retrieval_k: Số documents retrieval cho mỗi query
        rrf_k: Constant cho RRF
        rerank_top_k: Số documents sau reranking
        rerank_threshold: Ngưỡng score tối thiểu sau reranking
        
    Returns:
        Advanced RAG Pipeline instance
    """
    return AdvancedRAGPipeline(
        retrieval_k=retrieval_k,
        rrf_k=rrf_k,
        rerank_top_k=rerank_top_k,
        rerank_threshold=rerank_threshold
    )

class AdvancedRAGPipeline:
    """Pipeline RAG nâng cao: Multi-Query → RRF → Rerank → LLM.

    Dùng khi cần khả năng retrieval tốt hơn (mở rộng truy vấn) và chất lượng trả lời cao hơn
    (kết hợp nhiều nguồn qua RRF và sàng lọc bằng Cross-Encoder).
    """
    
    def __init__(self, retrieval_k: int = 5, rrf_k: float = 60.0, rerank_top_k: int = 10, rerank_threshold: float = 0.5):
        """
        Args:
            retrieval_k: Số documents retrieval cho mỗi query
            rrf_k: Constant cho RRF
            rerank_top_k: Số documents sau reranking
            rerank_threshold: Ngưỡng score tối thiểu sau reranking
        """
        self.retrieval_k = retrieval_k
        self.rrf_k = rrf_k
        self.rerank_top_k = rerank_top_k
        self.rerank_threshold = rerank_threshold
        
        # Khởi tạo các components
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.reranker = CrossEncoderReranker()
        
        # Khởi tạo LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b"
        )
        
        # Prompt QA dùng trực tiếp bằng f-string ở generate_answer; giữ cấu trúc hiện tại
        
        # Lưu truy vấn hiệu lực dùng cho rerank (nguyên gốc hoặc đã contextualize)
        self._effective_query: Optional[str] = None
    
    def contextualize_query(self, original_query: str, chat_history: List) -> str:
        """Viết lại câu hỏi thành dạng độc lập dựa trên history.

        - Xử lý tham chiếu đại từ (it, they, this, ...)
        - Giảm rủi ro routing sai và retrieval sai ngữ cảnh.
        - Thất bại → trả nguyên câu gốc.
        """
        try:
            # Chuyển đổi history sang định dạng LangChain messages
            langchain_history = []
            for message in (chat_history or []):
                if message.get("role") == "human":
                    langchain_history.append(("human", message.get("content", "")))
                elif message.get("role") == "ai":
                    langchain_history.append(("ai", message.get("content", "")))

            # Tạo messages theo prompt contextualize
            messages = contextualize_q_prompt.format_messages(
                chat_history=langchain_history,
                input=original_query,
            )
            response = self.llm.invoke(messages)
            text = (getattr(response, "content", None) or str(response)).strip()
            return text if text else original_query
        except Exception as e:
            print(f"Error contextualizing query: {e}")
            return original_query
    
    def retrieve_documents(self, query: str) -> List[DocumentScore]:
        """Lấy top-k tài liệu liên quan từ vectorstore cho 1 query."""
        try:
            # Sử dụng vectorstore để retrieve
            retriever = vectorstore.as_retriever(search_kwargs={"k": self.retrieval_k})
            
            # Sử dụng invoke thay vì get_relevant_documents (deprecated)
            try:
                docs = retriever.invoke(query)
            except Exception:
                # Fallback cho phiên bản cũ
                docs = retriever.get_relevant_documents(query)
            
            # Chuyển đổi thành DocumentScore objects
            document_scores = []
            for i, doc in enumerate(docs):
                # Kiểm tra xem doc có page_content không
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif hasattr(doc, 'content'):
                    content = doc.content
                else:
                    content = str(doc)
                
                doc_score = DocumentScore(
                    content=content,
                    metadata=getattr(doc, 'metadata', {}),
                    score=1.0 - (i * 0.1),  # Score giảm dần theo rank
                    source_query=query,
                    rank=i
                )
                document_scores.append(doc_score)
            
            return document_scores
            
        except Exception as e:
            print(f"Error in document retrieval: {e}")
            return []
    
    def multi_query_retrieval(self, original_query: str, chat_history: List = None) -> List[List[DocumentScore]]:
        """Retrieve cho danh sách queries (gốc + expanded).

        - Có history: reformulate trước rồi expand.
        - Không history: expand trên câu gốc.
        """
        if chat_history and len(chat_history) > 0:
            standalone_query = self.contextualize_query(original_query, chat_history)
            self._effective_query = standalone_query
            base_query = standalone_query
        else:
            self._effective_query = original_query
            base_query = original_query

        # Expand queries dựa trên base_query
        all_queries = get_all_queries(base_query)

        # Retrieve documents cho mỗi query
        all_results: List[List[DocumentScore]] = []
        for q in all_queries:
            docs = self.retrieve_documents(q)
            all_results.append(docs)

        return all_results
    
    def apply_rrf(self, query_results: List[List[DocumentScore]]) -> List[DocumentScore]:
        """Hợp nhất kết quả multi-query bằng RRF (ưu tiên doc xuất hiện nhiều/đứng hạng cao)."""
        if not query_results:
            return []
        
        # Áp dụng RRF trực tiếp với query_results (đã là List[List[DocumentScore]])
        fused_docs = self.rrf.fuse_results(query_results)
        return fused_docs
    
    def apply_reranking(self, query: str, documents: List[DocumentScore]) -> List[DocumentScore]:
        """Rerank bằng Cross-Encoder, lọc theo threshold và trả về top_k."""
        try:
            reranked_docs = self.reranker.rerank_with_threshold(
                query, documents, 
                threshold=self.rerank_threshold, 
                top_k=self.rerank_top_k
            )
            
            print(f"Reranking completed: {len(reranked_docs)} documents after threshold filtering")
            return reranked_docs
            
        except Exception as e:
            print(f"Error in reranking: {e}")
            return documents[:self.rerank_top_k]
    
    def generate_answer(self, query: str, documents: List[DocumentScore], 
                       chat_history: List = None) -> Dict[str, Any]:
        """Gọi LLM để sinh câu trả lời dựa trên context + history."""
        if not documents:
            return {"answer": "I'm sorry, but I couldn't find relevant information to answer this question. You can try asking about other topics or contact us for additional support."}
        
        try:
            # Chuẩn bị context từ documents
            context_parts = []
            for doc in documents:
                if hasattr(doc, 'content') and doc.content:
                    context_parts.append(str(doc.content))
                else:
                    print(f"Warning: Document missing content attribute: {type(doc)}")
            
            if not context_parts:
                return {"answer": "I'm sorry, but I couldn't extract content from the documents to answer this question. You can try asking about other topics or contact us for additional support."}
            
            context = "\n\n".join(context_parts)
            print(f"Debug: Context length: {len(context)} characters")
            
            # Chuẩn bị chat history cho LangChain format
            if chat_history is None:
                chat_history = []
            
            # Chuyển đổi chat_history format cho LangChain
            langchain_history = []
            for message in chat_history:
                if message["role"] == "human":
                    langchain_history.append(("human", message["content"]))
                elif message["role"] == "ai":
                    langchain_history.append(("ai", message["content"]))
            
            # Tạo prompt với context và chat history
            if chat_history and len(chat_history) > 0:
                # Nếu có chat history, sử dụng prompt với context và history
                prompt_with_history = f"""You are a helpful AI assistant. Use the following context and chat history to answer the user's question.

Context: {context}

Chat History:
{self._format_chat_history(chat_history)}

Current Question: {query}

Answer:"""
                response = self.llm.invoke(prompt_with_history)
            else:
                # Nếu không có chat history, sử dụng prompt đơn giản
                response = self.llm.invoke(
                    f"""You are a helpful AI assistant. Use the following context to answer the user's question.

Context: {context}

Question: {query}

Answer:"""
                )
            
            return {"answer": response.content}
            
        except Exception as e:
            print(f"Error in answer generation: {e}")
            import traceback
            traceback.print_exc()
            return {"answer": f"Error occurred while generating answer: {str(e)}"}
    
    def _format_chat_history(self, chat_history: List) -> str:
        """Format chat history thành string để sử dụng trong prompt"""
        if not chat_history:
            return ""
        
        formatted_history = []
        for message in chat_history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if role == "human":
                formatted_history.append(f"User: {content}")
            elif role == "ai":
                formatted_history.append(f"Assistant: {content}")
        
        return "\n".join(formatted_history)
    
    def run_pipeline(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """Chạy toàn bộ Advanced RAG và trả kết quả + metadata để debug/giám sát."""
        print(f"Starting Advanced RAG Pipeline for query: '{query}'")
        
        # 1. Multi-query retrieval
        print("\n1. Multi-Query Retrieval...")
        query_results = self.multi_query_retrieval(query, chat_history)
        
        # 2. RRF Fusion
        print("\n2. RRF Fusion...")
        fused_docs = self.apply_rrf(query_results)
        
        # 3. Cross-Encoder Reranking
        print("\n3. Cross-Encoder Reranking...")
        effective_query = getattr(self, "_effective_query", None) or query
        reranked_docs = self.apply_reranking(effective_query, fused_docs)
        
        # 4. LLM Generation
        print("\n4. LLM Generation...")
        answer_result = self.generate_answer(query, reranked_docs, chat_history)
        answer = answer_result["answer"]
        
        # 5. Tổng hợp kết quả
        result = {
            "answer": answer,
            "original_query": query,
            "expanded_queries": [qr[0].source_query for qr in query_results if qr],
            "retrieved_docs_count": sum(len(qr) for qr in query_results),
            "fused_docs_count": len(fused_docs),
            "final_docs_count": len(reranked_docs),
            "final_documents": [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "score": doc.score,
                    "metadata": doc.metadata
                }
                for doc in reranked_docs
            ],
            "pipeline_status": "completed"
        }
        
        print(f"\nPipeline completed successfully!")
        print(f"Final answer length: {len(answer)} characters")
        
        return result
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về pipeline
        """
        return {
            "retrieval_k": self.retrieval_k,
            "rrf_k": self.rrf_k,
            "rerank_top_k": self.rerank_top_k,
            "rerank_threshold": self.rerank_threshold,
            "reranker_info": self.reranker.get_model_info(),
            "llm_model": "openai/gpt-oss-20b",
            "vectorstore": "Pinecone"
        }