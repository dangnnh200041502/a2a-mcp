from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    tool_used: str | None = None

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int