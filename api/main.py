from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain, get_advanced_rag_pipeline
from agent_manager import AgentManager
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from pinecone_utils import index_document_to_pinecone, delete_doc_from_pinecone
import os
import uuid
import logging
from langchain_groq import ChatGroq
logging.basicConfig(filename='app.log', level=logging.INFO)
app = FastAPI()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    """RAG endpoint với semantic chunking và history (giữ nguyên workflow hiện tại)"""
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")

    # Lấy history nếu có để contextualize trước khi route
    chat_history = get_chat_history(session_id) if session_id else []
    effective_question = query_input.question
    if chat_history:
        try:
            adv = get_advanced_rag_pipeline()
            effective_question = adv.contextualize_query(query_input.question, chat_history)
        except Exception as e:
            logging.warning(f"Contextualize failed, fallback original question. Error: {e}")

    # Router: nếu out-of-scope (route=0) -> trả câu lịch sự, KHÔNG lưu session/log
    try:
        agent = AgentManager()
        # Prefer checking the ORIGINAL question for calculator detection to avoid
        # contextualization removing numbers/operators.
        orig_decision = agent.decide_tool(query_input.question)
        if orig_decision.get("tool") == "calculator":
            decision = orig_decision
        else:
            decision = agent.decide_tool(effective_question)
        tool = decision.get("tool", "rag")
        reason = decision.get("reason", "")
    except Exception as e:
        logging.error(f"Agent manager error: {e}")
        tool, reason = "rag", "agent_error"

    if tool == "calculator":
        # Generate a polite no-info response via LLM (no logging/no session persistence)
        try:
            calc_res = agent.use_calculator(effective_question)
            if calc_res.get("error") is None:
                answer = f"Result: {calc_res.get('result')}"
            else:
                answer = "I couldn't evaluate that expression."
        except Exception:
            answer = "I couldn't evaluate that expression."
        # Ensure session id and save logs like other paths
        if not session_id:
            session_id = str(uuid.uuid4())
        insert_application_logs(session_id, query_input.question, answer)
        logging.info(f"Calculator used ({reason}), session saved.")
        return QueryResponse(answer=answer, session_id=session_id, tool_used="calculator")

    if not session_id:
        session_id = str(uuid.uuid4())

    # For basic chat, keep existing RAG chain behavior
    rag_chain = get_rag_chain()
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']
    insert_application_logs(session_id, query_input.question, answer)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, tool_used="rag_basic")

 

@app.post("/chat-advanced")
def chat_with_advanced_pipeline(query_input: QueryInput):
    """RAG endpoint với Advanced Pipeline: Multiple Query → RRF → Rerank → LLM"""
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question} (advanced pipeline)")

    # Lấy history nếu có để contextualize trước khi route
    chat_history_existing = get_chat_history(session_id) if session_id else []
    effective_question = query_input.question
    if chat_history_existing:
        try:
            adv_tmp = get_advanced_rag_pipeline()
            effective_question = adv_tmp.contextualize_query(query_input.question, chat_history_existing)
        except Exception as e:
            logging.warning(f"Contextualize failed (advanced), fallback original. Error: {e}")

    # Router: nếu out-of-scope (route=0) -> trả câu lịch sự, KHÔNG lưu session/log
    try:
        agent = AgentManager()
        # Prefer checking original question first for calculator path
        orig_decision = agent.decide_tool(query_input.question)
        if orig_decision.get("tool") == "calculator":
            decision = orig_decision
        else:
            decision = agent.decide_tool(effective_question)
        tool = decision.get("tool", "rag")
        reason = decision.get("reason", "")
    except Exception as e:
        logging.error(f"Agent manager error: {e}")
        tool, reason = "rag", "agent_error"

    if tool == "calculator":
        try:
            calc_res = agent.use_calculator(effective_question)
            if calc_res.get("error") is None:
                answer = f"Result: {calc_res.get('result')}"
            else:
                answer = "I couldn't evaluate that expression."
        except Exception:
            answer = "I couldn't evaluate that expression."
        if not session_id:
            session_id = str(uuid.uuid4())
        insert_application_logs(session_id, query_input.question, answer)
        logging.info(f"Calculator used ({reason}), session saved.")
        return {"answer": answer, "session_id": session_id, "tool_used": "calculator"}

    if not session_id:
        session_id = str(uuid.uuid4())

    question = query_input.question
    chat_history = get_chat_history(session_id)

    try:
        # Full orchestration with Agent Manager: retrieve docs and generate via GPT-OSS
        agent = AgentManager()
        orchestration = agent.orchestrate_answer(
            question,
            chat_history,
        )
        # Lưu conversation vào database
        insert_application_logs(session_id, question, orchestration.get("answer", ""))
        logging.info(f"Advanced pipeline completed - Session: {session_id}, Question: '{question}'")
        
        meta = orchestration.get("meta", {})
        tools_used = orchestration.get("tools_used") or []
        response = {
            "answer": orchestration.get("answer", ""),
            "session_id": session_id,
            "tool_used": tools_used,
            "subtasks": orchestration.get("subtasks", []),
            "sufficient": orchestration.get("sufficient", True)
        }

        # Chỉ bổ sung thông tin retrieval khi có dùng RAG
        if any(t == "rag_retrieval" for t in tools_used):
            response["pipeline_info"] = {
                "retrieval_k": meta.get("retrieved_docs_count", 0),
                "fused_docs": meta.get("fused_docs_count", 0),
                "final_docs": meta.get("final_docs_count", 0),
                "expanded_queries": meta.get("expanded_queries", []),
                "pipeline_status": "completed"
            }
            response["final_documents"] = orchestration.get("final_documents", [])

        return response
        
    except Exception as e:
        logging.error(f"Advanced pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced pipeline error: {e}")

 

from fastapi import UploadFile, File, HTTPException
import os
import shutil

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

 