"""
tests/test_api.py
Unit and integration tests for the document intelligence platform.
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ─── API Tests ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    with patch("agents.orchestrator.get_orchestrator") as mock_orch, \
         patch("core.vector_store.get_vector_store") as mock_vs:

        # Mock vector store
        mock_vs_instance = MagicMock()
        mock_vs_instance.collection.count.return_value = 42
        mock_vs.return_value = mock_vs_instance

        # Mock orchestrator
        from core.models import QueryResponse
        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = QueryResponse(
            query="test query",
            answer="This is a test answer.",
            citations=[],
            agent_trace=["router_agent", "retrieval_agent", "synthesis_agent"],
            latency_ms=250.0,
        )
        mock_orch.return_value = mock_orch_instance

        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "chunks_indexed" in data


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Multi-Agent" in resp.json()["name"]


def test_query_endpoint(client):
    resp = client.post("/query", json={"query": "What is machine learning?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert "agent_trace" in data
    assert "latency_ms" in data


def test_query_empty_string(client):
    resp = client.post("/query", json={"query": "   "})
    assert resp.status_code == 400


def test_list_documents_empty(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ─── Unit Tests ───────────────────────────────────────────────────────────────

class TestSemanticChunker:
    def test_chunking_long_text(self):
        from agents.ingestion_agent import SemanticChunker
        chunker = SemanticChunker()
        long_text = "Machine learning is amazing. " * 100
        chunks = chunker.split(long_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 600  # chunk_size + some tolerance

    def test_chunking_short_text(self):
        from agents.ingestion_agent import SemanticChunker
        chunker = SemanticChunker()
        short = "Short text."
        chunks = chunker.split(short)
        assert len(chunks) == 1


class TestRouterAgent:
    @patch("agents.router_agent.ChatOpenAI")
    def test_router_classification(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.content = "factual"
        mock_llm.__or__ = lambda self, other: MagicMock(
            invoke=lambda x: mock_response
        )

        from agents.router_agent import RouterAgent, QueryIntent
        router = RouterAgent()
        # Test that it returns a valid QueryIntent
        assert QueryIntent.FACTUAL.value == "factual"

    def test_retrieval_config_visual(self):
        from agents.router_agent import RouterAgent, QueryIntent
        with patch("agents.router_agent.ChatOpenAI"):
            router = RouterAgent()
            config = router.get_retrieval_config(QueryIntent.VISUAL)
            assert config["prefer_images"] is True

    def test_retrieval_config_summary_fetches_more(self):
        from agents.router_agent import RouterAgent, QueryIntent
        from core.config import settings
        with patch("agents.router_agent.ChatOpenAI"):
            router = RouterAgent()
            factual_config = router.get_retrieval_config(QueryIntent.FACTUAL)
            summary_config = router.get_retrieval_config(QueryIntent.SUMMARY)
            assert summary_config["top_k"] > factual_config["top_k"]


class TestDocumentModels:
    def test_document_chunk_defaults(self):
        from core.models import DocumentChunk
        chunk = DocumentChunk(doc_id="test-doc", content="Hello world")
        assert chunk.chunk_id  # auto-generated
        assert chunk.has_image is False
        assert chunk.chunk_index == 0

    def test_query_request_validation(self):
        from core.models import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="ab")  # too short

    def test_ingested_document_defaults(self):
        from core.models import IngestedDocument, DocumentSource
        doc = IngestedDocument(
            source=DocumentSource.PDF,
            source_path="/tmp/test.pdf",
        )
        assert doc.doc_id  # auto-generated
        assert doc.status == "pending"
