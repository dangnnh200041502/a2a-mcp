"""Tiện ích Pinecone & semantic chunking.

Chức năng chính:
- Khởi tạo embeddings + vectorstore Pinecone.
- Tải & chia nhỏ tài liệu theo nghĩa (SemanticChunker) → Document.
- Chuẩn hóa metadata trước khi index để truy vết dễ dàng.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    PyMuPDFLoader,
    PDFPlumberLoader,
)
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
import os
from pinecone import Pinecone, ServerlessSpec
import re
import unicodedata

# 1) ENV & PINECONE SETUP
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Khởi tạo embeddings và vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = os.getenv("PINECONE_INDEX_NAME", "rag")

# Kiểm tra và tạo index nếu chưa tồn tại
try:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension cho all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' đã được tạo.")
    else:
        print(f"Index '{index_name}' đã tồn tại.")
except Exception as e:
    print(f"Error khi tạo index: {e}")

# Khởi tạo vectorstore sau khi đảm bảo index đã tồn tại
try:
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    print(f"Vectorstore đã được khởi tạo với index '{index_name}'")
except Exception as e:
    print(f"Error khi khởi tạo vectorstore: {e}")
    vectorstore = None

# 2) SEMANTIC CHUNKING
from langchain_experimental.text_splitter import SemanticChunker

def build_semantic_chunker():
    """Khởi tạo SemanticChunker với tham số an toàn.

    - Nhánh 1: phiên bản mới (keyword-only args)
    - Nhánh 2: phiên bản cũ (positional args) để tránh lỗi tương thích.
    """
    try:
        return SemanticChunker(
            embeddings=embeddings,
            buffer_size=5,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=0.75,
            sentence_split_regex=r"(?<=[.!?])\s+(?!\d+\.\s)",
            min_chunk_size=512,
        )
    except TypeError:
        return SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            buffer_size=1,
        )

# Tạo splitter
semantic_chunker = build_semantic_chunker()

# 3) LOAD & SPLIT TÀI LIỆU (dùng semantic)
def load_and_split_document(file_path: str) -> List[Document]:
    """Load 1 file rồi tách thành các chunk ngữ nghĩa (Document)."""
    try:
        if file_path.endswith('.pdf'):
            # Ưu tiên PyMuPDF (tốt về layout/spacing), fallback PDFPlumber, cuối cùng PyPDF
            loader = None
            try:
                loader = PyMuPDFLoader(file_path)
            except Exception:
                try:
                    loader = PDFPlumberLoader(file_path)
                except Exception:
                    loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.html'):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Loại tệp không hỗ trợ: {file_path}")

        documents = loader.load()

        # Sử dụng semantic chunking
        splits: List[Document] = []
        for doc in documents:
            # Làm sạch text trước khi chunk để tránh dính chữ khi trích xuất từ PDF
            cleaned_text = _clean_text(doc.page_content)
            text_chunks = semantic_chunker.split_text(cleaned_text)
            for i, chunk in enumerate(text_chunks):
                new_doc = Document(
                    page_content=chunk.strip(),
                    metadata={**(doc.metadata or {}), "chunk_index": i, "chunk_method": "semantic"}
                )
                splits.append(new_doc)

        # Loại bỏ chunk rỗng nếu có
        splits = [d for d in splits if d.page_content]
        return splits

    except Exception as e:
        print(f"Error khi tải và chia tài liệu: {e}")
        return []

def _clean_text(text: str) -> str:
    """Chuẩn hóa và làm sạch text từ PDF để hạn chế lỗi dính chữ.

    - Chuẩn hóa unicode (NFKC)
    - Gộp từ bị gạch nối khi xuống dòng: "exam-\nple" -> "example"
    - Thay \r, \n liên tiếp bằng khoảng trắng đơn
    - Gộp nhiều khoảng trắng về một khoảng trắng
    - Đảm bảo có khoảng trắng sau dấu chấm/câu nếu thiếu
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove hyphen + newline word breaks
    text = re.sub(r"(\w)-(\n|\r\n)(\w)", r"\1\3", text)

    # Replace newlines with spaces
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"[\t ]+", " ", text)

    # Ensure a space after sentence punctuation if missing (basic heuristic)
    text = re.sub(r"([\.!?])(\w)", r"\1 \2", text)

    return text.strip()

# 4) INDEX / DELETE VỚI PINECONE
def index_document_to_pinecone(file_path: str, file_id: int) -> bool:
    """Đọc → chunk → chuẩn hóa metadata → thêm vào Pinecone."""
    if vectorstore is None:
        print("Vectorstore chưa được khởi tạo.")
        return False

    try:
        splits = load_and_split_document(file_path)
        if not splits:
            print(f"Không có tài liệu nào để chỉ mục cho file {file_path}.")
            return False

        # Chuẩn hóa metadata cho mỗi chunk trước khi index
        for j, split in enumerate(splits):
            split.metadata = _normalize_metadata(
                raw_meta=(split.metadata or {}),
                file_path=os.path.abspath(file_path),
                file_id=file_id,
                chunk_index=j,
            )

        # Thêm tài liệu vào vectorstore (Pinecone)
        vectorstore.add_documents(splits)
        print(f"Đã thêm tài liệu từ {file_path} vào Pinecone (chunks: {len(splits)}, method: semantic).")
        return True
    except Exception as e:
        print(f"Error khi chỉ mục tài liệu {file_path}: {e}")
        return False

def delete_doc_from_pinecone(file_id: int) -> bool:
    """
    Xóa tài liệu từ Pinecone theo file_id.
    """
    if vectorstore is None:
        print("Vectorstore chưa được khởi tạo.")
        return False

    try:
        vectorstore.delete(filter={"file_id": file_id})
        print(f"Đã xóa tất cả tài liệu với file_id {file_id} từ Pinecone.")
        return True
    except Exception as e:
        print(f"Error khi xóa tài liệu với file_id {file_id} từ Pinecone: {str(e)}")
        return False

def _normalize_metadata(raw_meta: dict, file_path: str, file_id: int, chunk_index: int) -> dict:
    """Chuẩn hóa metadata: chỉ giữ trường hữu ích và bổ sung trường chuẩn hệ thống."""
    whitelist_keys = {
        'page',        # int
        'page_label',  # optional str
        'title',       # optional str
        'total_pages', # optional int
    }

    meta = {}
    # Copy các key trong whitelist nếu tồn tại và hợp lệ
    for k in whitelist_keys:
        if k in raw_meta and raw_meta[k] not in (None, ""):
            meta[k] = raw_meta[k]

    # Chuẩn hóa kiểu dữ liệu
    try:
        if 'page' in meta:
            meta['page'] = int(meta['page'])
    except Exception:
        meta.pop('page', None)

    try:
        if 'total_pages' in meta:
            meta['total_pages'] = int(meta['total_pages'])
    except Exception:
        meta.pop('total_pages', None)

    # Bổ sung các trường chuẩn
    meta['source'] = os.path.basename(file_path)
    meta['source_path'] = file_path
    meta['file_id'] = file_id
    meta['chunk_index'] = chunk_index
    meta['chunk_id'] = f"{file_id}-{chunk_index}"
    meta['chunk_method'] = 'semantic'

    return meta