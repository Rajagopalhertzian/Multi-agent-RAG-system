"""
core/models.py
Pydantic schemas for documents, chunks, queries, and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import uuid


class DocumentSource(str, Enum):
    PDF = "pdf"
    URL = "url"
    TEXT = "text"


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str
    metadata: dict = {}
    embedding: Optional[List[float]] = None
    chunk_index: int = 0
    has_image: bool = False
    image_description: Optional[str] = None  # from Vision Agent


class IngestedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: DocumentSource
    source_path: str
    title: str = ""
    num_chunks: int = 0
    has_images: bool = False
    status: str = "pending"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    doc_ids: Optional[List[str]] = None  # filter to specific docs


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    content_snippet: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    agent_trace: List[str] = []  # which agents were invoked
    evaluation: Optional[dict] = None  # RAGAS scores if enabled
    latency_ms: float = 0.0


class EvaluationResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float] = None
