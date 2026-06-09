"""
agents/orchestrator.py
LangGraph-based multi-agent orchestrator.
Defines the agent graph: Router → Retrieval → Synthesis → Evaluation
This is the CORE of the project — shows LangGraph skills prominently.
"""
import time
from typing import TypedDict, List, Optional, Annotated
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from core.models import DocumentChunk, Citation, QueryRequest, QueryResponse
from agents.router_agent import RouterAgent, QueryIntent
from agents.retrieval_agent import RetrievalAgent
from agents.synthesis_agent import SynthesisAgent
from evaluation.ragas_evaluator import RAGASEvaluator


# ─── Graph State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state passed between all agents in the graph."""
    query: str
    doc_ids: Optional[List[str]]
    use_reranker: bool
    top_k: int

    # Router outputs
    intent: str
    retrieval_config: dict

    # Retrieval outputs
    retrieved_chunks: list   # List of (DocumentChunk, float)

    # Synthesis outputs
    answer: str
    citations: List[Citation]
    confidence: float
    has_visual_evidence: bool

    # Evaluation outputs
    evaluation: Optional[dict]

    # Trace
    agent_trace: List[str]
    latency_ms: float
    start_time: float


# ─── Agent Node Functions ──────────────────────────────────────────────────────

def router_node(state: AgentState, router: RouterAgent) -> AgentState:
    """Node 1: Classify query intent and determine retrieval strategy."""
    logger.info("--- Router Agent ---")
    intent = router.classify(state["query"])
    config = router.get_retrieval_config(intent)

    return {
        **state,
        "intent": intent.value,
        "retrieval_config": config,
        "agent_trace": state["agent_trace"] + ["router_agent"],
    }


def retrieval_node(state: AgentState, retriever: RetrievalAgent) -> AgentState:
    """Node 2: Retrieve relevant chunks using hybrid search + reranking."""
    logger.info("--- Retrieval Agent ---")

    if state["intent"] == QueryIntent.OUT_OF_SCOPE.value:
        return {**state, "retrieved_chunks": [], "agent_trace": state["agent_trace"] + ["retrieval_agent_skipped"]}

    config = state.get("retrieval_config", {})
    top_k = config.get("top_k", state["top_k"])
    use_reranker = config.get("use_reranker", state["use_reranker"])

    chunks_with_scores = retriever.retrieve(
        query=state["query"],
        top_k=top_k,
        doc_ids=state.get("doc_ids"),
        use_reranker=use_reranker,
    )

    # If visual intent, boost image chunks to the front
    if config.get("prefer_images"):
        image_chunks = [(c, s) for c, s in chunks_with_scores if c.has_image]
        text_chunks = [(c, s) for c, s in chunks_with_scores if not c.has_image]
        chunks_with_scores = image_chunks + text_chunks

    return {
        **state,
        "retrieved_chunks": chunks_with_scores,
        "agent_trace": state["agent_trace"] + ["retrieval_agent"],
    }


def synthesis_node(state: AgentState, synthesizer: SynthesisAgent) -> AgentState:
    """Node 3: Generate grounded answer with citations."""
    logger.info("--- Synthesis Agent ---")

    result = synthesizer.synthesize(
        query=state["query"],
        retrieved_chunks=state["retrieved_chunks"],
    )

    return {
        **state,
        "answer": result["answer"],
        "citations": result["citations"],
        "confidence": result.get("confidence", 0.0),
        "has_visual_evidence": result.get("has_visual_evidence", False),
        "agent_trace": state["agent_trace"] + ["synthesis_agent"],
    }


def evaluation_node(state: AgentState, evaluator: RAGASEvaluator) -> AgentState:
    """Node 4: Evaluate answer quality using RAGAS metrics."""
    logger.info("--- Evaluation Agent ---")

    if not state["retrieved_chunks"] or not state.get("answer"):
        return {**state, "evaluation": None}

    try:
        contexts = [chunk.content for chunk, _ in state["retrieved_chunks"]]
        eval_result = evaluator.evaluate(
            query=state["query"],
            answer=state["answer"],
            contexts=contexts,
        )
        return {
            **state,
            "evaluation": eval_result,
            "agent_trace": state["agent_trace"] + ["evaluation_agent"],
        }
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return {**state, "evaluation": None}


def should_evaluate(state: AgentState) -> str:
    """Conditional edge: only evaluate if answer was generated."""
    from core.config import settings
    if settings.ragas_eval_enabled and state.get("answer") and state["retrieved_chunks"]:
        return "evaluate"
    return "end"


# ─── Graph Builder ─────────────────────────────────────────────────────────────

class DocumentIntelligenceOrchestrator:
    """
    LangGraph multi-agent orchestrator.
    Graph topology:
      router → retrieval → synthesis → [evaluate?] → end
    """

    def __init__(self):
        self.router = RouterAgent()
        self.retriever = RetrievalAgent()
        self.synthesizer = SynthesisAgent()
        self.evaluator = RAGASEvaluator()
        self.graph = self._build_graph()
        logger.info("Orchestrator ready — LangGraph graph compiled")

    def _build_graph(self) -> any:
        """Build and compile the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes (bind agent instances to node functions)
        workflow.add_node("router", lambda s: router_node(s, self.router))
        workflow.add_node("retrieval", lambda s: retrieval_node(s, self.retriever))
        workflow.add_node("synthesis", lambda s: synthesis_node(s, self.synthesizer))
        workflow.add_node("evaluate", lambda s: evaluation_node(s, self.evaluator))

        # Define edges
        workflow.set_entry_point("router")
        workflow.add_edge("router", "retrieval")
        workflow.add_edge("retrieval", "synthesis")

        # Conditional: evaluate only if enabled and answer exists
        workflow.add_conditional_edges(
            "synthesis",
            should_evaluate,
            {"evaluate": "evaluate", "end": END},
        )
        workflow.add_edge("evaluate", END)

        return workflow.compile()

    def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the full agent pipeline for a query."""
        start = time.time()

        initial_state: AgentState = {
            "query": request.query,
            "doc_ids": request.doc_ids,
            "use_reranker": request.use_reranker,
            "top_k": request.top_k,
            "intent": "factual",
            "retrieval_config": {},
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "has_visual_evidence": False,
            "evaluation": None,
            "agent_trace": [],
            "latency_ms": 0.0,
            "start_time": start,
        }

        logger.info(f"Running pipeline for: '{request.query[:80]}'")
        final_state = self.graph.invoke(initial_state)

        latency = (time.time() - start) * 1000

        return QueryResponse(
            query=request.query,
            answer=final_state.get("answer", "No answer generated."),
            citations=final_state.get("citations", []),
            agent_trace=final_state.get("agent_trace", []),
            evaluation=final_state.get("evaluation"),
            latency_ms=round(latency, 2),
        )


# Singleton
_orchestrator: Optional[DocumentIntelligenceOrchestrator] = None


def get_orchestrator() -> DocumentIntelligenceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DocumentIntelligenceOrchestrator()
    return _orchestrator
