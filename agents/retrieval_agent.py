"""
agents/retrieval_agent.py
Retrieves and reranks chunks for a given query.
  - Hybrid search (dense + sparse) from vector store
  - Cross-encoder reranking for precision
  - Returns top-K DocumentChunks with scores
"""
from typing import List, Tuple, Optional
from loguru import logger

from core.config import settings
from core.models import DocumentChunk
from core.vector_store import get_vector_store


class RerankerAgent:
    """
    Cross-encoder reranker using sentence-transformers.
    Much more accurate than bi-encoder similarity for final ranking.
    """

    def __init__(self):
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            self._available = True
            logger.info("Cross-encoder reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")
            self._available = False

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[DocumentChunk, float]],
        top_k: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Rerank retrieved chunks using cross-encoder scores."""
        if not self._available or not chunks:
            return chunks[:top_k]

        try:
            pairs = [[query, chunk.content] for chunk, _ in chunks]
            scores = self._model.predict(pairs)

            reranked = sorted(
                zip([c for c, _ in chunks], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            logger.debug(f"Reranked {len(chunks)} → top {top_k}")
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return chunks[:top_k]


class RetrievalAgent:
    """
    Full retrieval pipeline:
    1. Hybrid search (dense + BM25)
    2. Cross-encoder reranking
    3. Return top-K with scores
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.reranker = RerankerAgent()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        doc_ids: Optional[List[str]] = None,
        use_reranker: bool = True,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Main retrieval entry point.
        Returns list of (DocumentChunk, relevance_score) tuples.
        """
        top_k = top_k or settings.top_k_retrieval
        reranker_k = settings.reranker_top_k

        logger.info(f"Retrieving for: '{query[:80]}...' (top_k={top_k})")

        # Step 1: Hybrid search (gets more candidates than needed)
        candidates = self.vector_store.hybrid_search(
            query=query,
            top_k=top_k * 2,
            doc_ids=doc_ids,
        )

        if not candidates:
            logger.warning("No candidates found in vector store")
            return []

        # Step 2: Rerank for precision
        if use_reranker and len(candidates) > reranker_k:
            results = self.reranker.rerank(query, candidates, top_k=reranker_k)
        else:
            results = candidates[:top_k]

        logger.info(f"Retrieved {len(results)} chunks after reranking")
        return results
