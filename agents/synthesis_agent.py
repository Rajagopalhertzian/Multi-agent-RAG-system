"""
agents/synthesis_agent.py
Synthesizes a grounded, cited answer from retrieved chunks.
Uses structured output (Pydantic) to ensure citations are always returned.
"""
from typing import List, Tuple
from loguru import logger

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from core.config import settings
from core.models import DocumentChunk, Citation, QueryResponse


SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert document analyst. Answer the user's question using ONLY the provided context chunks.

Rules:
1. Base your answer strictly on the provided context. Do not hallucinate.
2. If the context doesn't contain enough information, say so clearly.
3. Always cite which chunks you used (by chunk_id).
4. Be concise but complete. Use markdown formatting.
5. If visual content (charts, images) is relevant, mention it explicitly.

Return a JSON object with this exact structure:
{{
  "answer": "your detailed answer in markdown",
  "chunk_ids_used": ["chunk_id_1", "chunk_id_2", ...],
  "confidence": 0.0-1.0,
  "has_visual_evidence": true/false
}}"""
    ),
    (
        "human",
        """Question: {query}

Context chunks:
{context}

Answer in the JSON format specified."""
    ),
])


class SynthesisAgent:
    """
    Generates a grounded answer with citations from retrieved chunks.
    Uses structured JSON output to ensure reliable parsing.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.1,
            groq_api_key=settings.groq_api_key,
        )
        self.parser = JsonOutputParser()
        self.chain = SYNTHESIS_PROMPT | self.llm | self.parser

    def synthesize(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> dict:
        """
        Generate a grounded answer from retrieved chunks.
        Returns dict with answer, citations, and metadata.
        """
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "citations": [],
                "agent_trace": ["synthesis_agent"],
            }

        # Format context for the prompt
        context_parts = []
        chunk_map = {}

        for chunk, score in retrieved_chunks:
            chunk_map[chunk.chunk_id] = (chunk, score)
            content = chunk.content

            # If chunk has image description, highlight it
            if chunk.has_image:
                content = f"[IMAGE CONTENT] {content}"

            context_parts.append(
                f"chunk_id: {chunk.chunk_id}\n"
                f"source: {chunk.metadata.get('source_path', 'unknown')} (page {chunk.metadata.get('page', '?')})\n"
                f"relevance_score: {score:.3f}\n"
                f"content: {content}\n"
            )

        context_str = "\n---\n".join(context_parts)

        logger.info(f"Synthesizing answer from {len(retrieved_chunks)} chunks")

        try:
            result = self.chain.invoke({
                "query": query,
                "context": context_str,
            })
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "answer": "An error occurred while generating the answer. Please try again.",
                "citations": [],
                "agent_trace": ["synthesis_agent", "error"],
            }

        # Build citation objects
        citations = []
        for chunk_id in result.get("chunk_ids_used", []):
            if chunk_id in chunk_map:
                chunk, score = chunk_map[chunk_id]
                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        doc_id=chunk.doc_id,
                        source_path=chunk.metadata.get("source_path", ""),
                        content_snippet=chunk.content[:200] + "...",
                        relevance_score=round(score, 4),
                    )
                )

        return {
            "answer": result.get("answer", "No answer generated."),
            "citations": citations,
            "confidence": result.get("confidence", 0.0),
            "has_visual_evidence": result.get("has_visual_evidence", False),
            "agent_trace": ["retrieval_agent", "synthesis_agent"],
        }
