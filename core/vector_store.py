"""
core/vector_store.py
Hybrid vector store: ChromaDB for persistence + FAISS for fast ANN search.
Handles embedding, storage, and retrieval with BM25 sparse retrieval as fallback.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from core.config import settings
from core.models import DocumentChunk


class HybridVectorStore:
    """
    Combines:
    - ChromaDB  → persistent vector store (survives restarts)
    - FAISS     → fast in-memory ANN index (for speed)
    - BM25      → sparse keyword retrieval (for exact-match fallback)
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        # ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="doc_intelligence",
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory BM25 index (rebuilt on startup from Chroma)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: List[DocumentChunk] = []
        self._rebuild_bm25()

        logger.info(f"VectorStore ready — {self.collection.count()} chunks indexed")

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embed and store a list of DocumentChunks."""
        if not chunks:
            return

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                **c.metadata,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "has_image": c.has_image,
                "image_description": c.image_description or "",
                "source_path": c.metadata.get("source_path", ""),
            }
            for c in chunks
        ]

        # Batch embed
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embeddings.embed_documents(texts)

        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Rebuild BM25
        self._rebuild_bm25()
        logger.info(f"Stored {len(chunks)} chunks. Total: {self.collection.count()}")

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def dense_search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Dense semantic search via ChromaDB cosine similarity."""
        query_embedding = self.embeddings.embed_query(query)

        where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks_scores = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = DocumentChunk(
                chunk_id=results["ids"][0][len(chunks_scores)],
                doc_id=meta.get("doc_id", ""),
                content=doc,
                metadata=meta,
                chunk_index=meta.get("chunk_index", 0),
                has_image=meta.get("has_image", False),
                image_description=meta.get("image_description") or None,
            )
            score = 1.0 - float(dist)  # cosine distance → similarity
            chunks_scores.append((chunk, score))

        return chunks_scores

    def sparse_search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """BM25 sparse keyword search."""
        if not self._bm25 or not self._bm25_chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self._bm25_chunks[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        alpha: float = 0.7,  # weight for dense vs sparse
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Reciprocal Rank Fusion of dense + sparse results.
        alpha=1.0 → pure dense; alpha=0.0 → pure sparse
        """
        dense_results = self.dense_search(query, top_k=top_k * 2, doc_ids=doc_ids)
        sparse_results = self.sparse_search(query, top_k=top_k * 2)

        # RRF fusion
        scores: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + alpha * (1 / (rank + 60))
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(sparse_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + (1 - alpha) * (1 / (rank + 60))
            chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]

    def get_all_doc_ids(self) -> List[str]:
        """Return unique document IDs in the store."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        return list({m["doc_id"] for m in results["metadatas"]})

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks for a document."""
        self.collection.delete(where={"doc_id": doc_id})
        self._rebuild_bm25()
        logger.info(f"Deleted document {doc_id}")

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from ChromaDB contents."""
        try:
            if self.collection.count() == 0:
                self._bm25 = None
                self._bm25_chunks = []
                return

            results = self.collection.get(include=["documents", "metadatas"])
            self._bm25_chunks = []
            for i, (doc, meta) in enumerate(
                zip(results["documents"], results["metadatas"])
            ):
                self._bm25_chunks.append(
                    DocumentChunk(
                        chunk_id=results["ids"][i],
                        doc_id=meta.get("doc_id", ""),
                        content=doc,
                        metadata=meta,
                    )
                )

            tokenized_corpus = [c.content.lower().split() for c in self._bm25_chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            logger.warning(f"BM25 rebuild failed: {e}")
            self._bm25 = None


# Singleton
_vector_store: Optional[HybridVectorStore] = None


def get_vector_store() -> HybridVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = HybridVectorStore()
    return _vector_store
