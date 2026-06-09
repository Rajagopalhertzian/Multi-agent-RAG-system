"""
evaluation/ragas_evaluator.py
Evaluates RAG pipeline quality using RAGAS metrics:
  - Faithfulness: is the answer grounded in the context?
  - Answer Relevancy: does the answer address the question?
  - Context Precision: are retrieved chunks relevant?
"""
from typing import List, Optional, Dict
from loguru import logger


class RAGASEvaluator:
    """
    Wraps RAGAS evaluation metrics.
    Designed to run asynchronously so it doesn't block API responses.
    """

    def __init__(self):
        self._available = False
        self._load_ragas()

    def _load_ragas(self):
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
            )
            self._evaluate = evaluate
            self._metrics = [faithfulness, answer_relevancy, context_precision]
            self._available = True
            logger.info("RAGAS evaluator loaded")
        except Exception as e:
            logger.warning(f"RAGAS not available: {e}")
            self._available = False

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Run RAGAS evaluation. Returns dict of metric scores (0.0-1.0).
        Returns None if evaluation is unavailable.
        """
        if not self._available:
            return None

        try:
            from datasets import Dataset

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            result = self._evaluate(dataset, metrics=self._metrics)

            scores = result.to_pandas().iloc[0].to_dict()

            # Round scores for cleaner output
            cleaned = {
                k: round(float(v), 4)
                for k, v in scores.items()
                if isinstance(v, (int, float)) and not k.startswith("Unnamed")
            }

            logger.info(f"RAGAS scores: {cleaned}")
            return cleaned

        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            return None

    def batch_evaluate(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
    ) -> Optional[Dict[str, float]]:
        """Evaluate a batch of query-answer pairs. Returns averaged scores."""
        if not self._available:
            return None

        try:
            from datasets import Dataset

            dataset = Dataset.from_dict({
                "question": queries,
                "answer": answers,
                "contexts": contexts_list,
            })

            result = self._evaluate(dataset, metrics=self._metrics)
            scores = result.to_pandas().mean(numeric_only=True).to_dict()

            return {
                k: round(float(v), 4)
                for k, v in scores.items()
                if isinstance(v, (int, float))
            }
        except Exception as e:
            logger.warning(f"Batch RAGAS evaluation failed: {e}")
            return None
