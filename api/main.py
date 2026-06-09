"""
api/main.py
FastAPI application — production-ready REST API for the document intelligence platform.
Endpoints:
  POST /ingest/pdf       — upload and ingest a PDF
  POST /ingest/text      — ingest raw text
  POST /query            — run multi-agent Q&A pipeline
  GET  /documents        — list all ingested documents
  DELETE /documents/{id} — remove a document
  GET  /health           — health check
"""
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from core.config import settings
from core.models import QueryRequest, QueryResponse, IngestedDocument
from agents.ingestion_agent import IngestionAgent
from agents.orchestrator import get_orchestrator


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up all agent singletons at startup."""
    logger.info("Starting Multi-Agent Document Intelligence Platform...")
    _ = get_orchestrator()  # pre-load all agents
    logger.info("All agents ready. API is live.")
    yield
    logger.info("Shutting down...")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Document Intelligence Platform",
    description="""
## Multi-Agent RAG System with LangGraph

Built by Raja Gopal S — features:
- **LangGraph orchestration** with Router, Retrieval, Synthesis, and Evaluation agents
- **Hybrid retrieval**: Dense (ChromaDB) + Sparse (BM25) + Cross-encoder reranking
- **Vision Agent**: CLIP-based image/chart understanding (unique differentiator)
- **RAGAS evaluation**: Automated faithfulness, relevancy, and precision scoring
- **Structured output**: All responses include citations and agent traces

### Tech Stack
LangGraph · LangChain · OpenAI · ChromaDB · FAISS · CLIP · RAGAS · FastAPI · Docker
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document registry (in production: use a database)
_document_registry: dict[str, IngestedDocument] = {}


# ─── Dependencies ─────────────────────────────────────────────────────────────

def get_ingestion_agent() -> IngestionAgent:
    return IngestionAgent()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for Docker and monitoring."""
    from core.vector_store import get_vector_store
    vs = get_vector_store()
    return {
        "status": "healthy",
        "chunks_indexed": vs.collection.count(),
        "documents_loaded": len(_document_registry),
        "llm_provider": "Groq (free)",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ingest/pdf", response_model=IngestedDocument, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent),
):
    """
    Upload and ingest a PDF document.
    - Extracts text with pdfplumber
    - Describes images/charts using CLIP Vision Agent
    - Chunks, embeds, and stores in hybrid vector store
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc = ingestion_agent.ingest_pdf(tmp_path)
        doc.source_path = file.filename  # show original name
        doc.title = Path(file.filename).stem
        _document_registry[doc.doc_id] = doc
        logger.info(f"Ingested PDF: {file.filename} → {doc.doc_id}")
        return doc
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class TextIngestionRequest(BaseModel):
    text: str
    title: str = "Untitled Document"


@app.post("/ingest/text", response_model=IngestedDocument, tags=["Ingestion"])
async def ingest_text(
    request: TextIngestionRequest,
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent),
):
    """Ingest raw text content directly."""
    if len(request.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short (min 50 characters)")

    doc = ingestion_agent.ingest_text(request.text, title=request.title)
    _document_registry[doc.doc_id] = doc
    return doc


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Run the full multi-agent pipeline on a query.

    **Pipeline:**
    1. **Router Agent** — classifies intent (factual/summary/visual/comparison)
    2. **Retrieval Agent** — hybrid dense+sparse search + cross-encoder reranking
    3. **Synthesis Agent** — generates grounded answer with citations
    4. **Evaluation Agent** — RAGAS scores (faithfulness, relevancy, precision)

    Returns structured response with answer, citations, agent trace, and evaluation scores.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    orchestrator = get_orchestrator()

    try:
        response = orchestrator.run(request)
        return response
    except Exception as e:
        logger.error(f"Query pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[IngestedDocument], tags=["Documents"])
def list_documents():
    """List all ingested documents with their metadata."""
    return list(_document_registry.values())


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    """Remove a document and all its chunks from the vector store."""
    if doc_id not in _document_registry:
        raise HTTPException(status_code=404, detail="Document not found")

    from core.vector_store import get_vector_store
    get_vector_store().delete_document(doc_id)
    del _document_registry[doc_id]

    return {"message": f"Document {doc_id} deleted successfully"}


@app.get("/", tags=["System"])
def root():
    return {
        "name": "Multi-Agent Document Intelligence Platform",
        "docs": "/docs",
        "health": "/health",
        "author": "Raja Gopal S",
        "github": "github.com/Rajagopalhertzian",
    }
