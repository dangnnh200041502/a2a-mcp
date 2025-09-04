from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from agents.agent_manager_mcp import AgentManager
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from pinecone_utils import index_document_to_pinecone, delete_doc_from_pinecone
import os
import uuid
import logging
from langchain_groq import ChatGroq
from fastapi import UploadFile, File, HTTPException
import os
import shutil
from contextlib import asynccontextmanager

logging.basicConfig(filename='app.log', level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize MCP-based AgentManager
    app.state.agent_manager = AgentManager()
    yield
    # Cleanup MCP connection
    await app.state.agent_manager.cleanup()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat_with_agentic_rag(query_input: QueryInput):
    """Agentic RAG endpoint using MCP tools"""
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")

    # Get chat history if available
    chat_history_existing = get_chat_history(session_id) if session_id else []
    effective_question = query_input.question

    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Use MCP-based AgentManager
        agent = app.state.agent_manager
        result = await agent.answer(effective_question, chat_history_existing)
        
        # Save conversation to database
        answer_text = result.get("answer") or ""
        insert_application_logs(session_id, query_input.question, answer_text)
        logging.info(f"MCP pipeline completed - Session: {session_id}, Question: '{query_input.question}'")
        
        tools_used = result.get("tools_used") or []
        response = {
            "answer": answer_text,
            "session_id": session_id,
        }
        # Include tool_used when non-empty
        if tools_used:
            response["tool_used"] = tools_used
        # If RAG used, attach top-1 retrieval snippet to represent retrieval
        if any(t == "rag_retrieval" for t in tools_used):
            snippets = result.get("snippets") or []
            if snippets:
                response["retrieval"] = snippets[0]
        return response
        
    except Exception as e:
        logging.error(f"MCP pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"MCP pipeline error: {e}")

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = insert_document_record(file.filename)
        success = index_document_to_pinecone(temp_file_path, file_id)
        
        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Pinecone
    pinecone_delete_success = delete_doc_from_pinecone(request.file_id)

    if pinecone_delete_success:
        # If successfully deleted from Pinecone, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Pinecone but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Pinecone."}

 