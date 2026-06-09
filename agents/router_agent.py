"""
agents/router_agent.py
Classifies the query intent and routes to the right sub-pipeline.
Uses LLM classification for smart routing decisions.
"""
from enum import Enum
from typing import Optional
from loguru import logger

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings


class QueryIntent(str, Enum):
    FACTUAL = "factual"          # Direct question answerable from docs
    SUMMARY = "summary"          # Summarise a document or section
    COMPARISON = "comparison"    # Compare multiple items/sections
    VISUAL = "visual"            # Question about charts/images/tables
    OUT_OF_SCOPE = "out_of_scope"  # Cannot be answered from docs


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query router for a document intelligence system.
Classify the user's query into exactly one of these intents:
- factual: a direct factual question answerable from documents
- summary: asks for a summary or overview
- comparison: asks to compare, contrast, or list differences
- visual: asks about charts, graphs, images, tables, or visual data
- out_of_scope: cannot be answered from documents (e.g. general knowledge, opinions)

Reply with ONLY the intent word. Nothing else."""
    ),
    ("human", "Query: {query}"),
])


class RouterAgent:
    """
    Classifies query intent to enable smarter retrieval strategies.
    - visual intent → prioritise image chunks
    - summary intent → fetch more chunks
    - comparison intent → multi-query retrieval
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.0,
            groq_api_key=settings.groq_api_key,
        )
        self.chain = ROUTER_PROMPT | self.llm

    def classify(self, query: str) -> QueryIntent:
        """Classify query intent. Falls back to FACTUAL on error."""
        try:
            result = self.chain.invoke({"query": query})
            intent_str = result.content.strip().lower()
            intent = QueryIntent(intent_str)
            logger.info(f"Query intent: {intent.value}")
            return intent
        except Exception as e:
            logger.warning(f"Router classification failed ({e}), defaulting to factual")
            return QueryIntent.FACTUAL

    def get_retrieval_config(self, intent: QueryIntent) -> dict:
        """Return retrieval configuration based on query intent."""
        configs = {
            QueryIntent.FACTUAL: {
                "top_k": settings.top_k_retrieval,
                "use_reranker": True,
                "prefer_images": False,
            },
            QueryIntent.SUMMARY: {
                "top_k": settings.top_k_retrieval * 2,  # fetch more for summaries
                "use_reranker": False,
                "prefer_images": False,
            },
            QueryIntent.COMPARISON: {
                "top_k": settings.top_k_retrieval,
                "use_reranker": True,
                "prefer_images": False,
            },
            QueryIntent.VISUAL: {
                "top_k": settings.top_k_retrieval,
                "use_reranker": True,
                "prefer_images": True,  # boost image chunks
            },
            QueryIntent.OUT_OF_SCOPE: {
                "top_k": 0,
                "use_reranker": False,
                "prefer_images": False,
            },
        }
        return configs.get(intent, configs[QueryIntent.FACTUAL])
