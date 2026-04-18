# models.py
from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    filename: str
    chunks: int
    status: str = "indexed"

class AskRequest(BaseModel):
    query: str
    docs: Optional[List[str]] = None  # None = search all uploaded docs
    top_k: int = 5
    model: str = "groq-llama"  # or "groq-llama"

class SourceChunk(BaseModel):
    doc: str
    text: str
    score: float
    page: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    confidence: float  # avg score of top chunks, 0-1

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    size_bytes: int
    status: str