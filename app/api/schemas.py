from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str

class SourceResponse(BaseModel):
    source_file: str
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]

class UploadResponse(BaseModel):
    message: str
