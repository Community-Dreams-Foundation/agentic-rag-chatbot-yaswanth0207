"""RAG evaluation using RAGAS metrics.

Measures faithfulness and answer relevancy to provide an objective
quality signal for the retrieval-augmented generation pipeline.
"""

from __future__ import annotations

import os

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.2"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class RAGEvaluator:
    """Evaluates RAG pipeline quality using RAGAS faithfulness & relevancy."""

    def __init__(self) -> None:
        logger.info("Initialising RAGEvaluator")
        self._initialised = False
        try:
            from ragas.metrics import answer_relevancy, faithfulness  # noqa: F401

            self._faithfulness = faithfulness
            self._answer_relevancy = answer_relevancy
            self._initialised = True
            logger.info("RAGAS metrics loaded")
        except Exception:
            logger.warning("RAGAS metrics could not be loaded – evaluation disabled")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> dict[str, float]:
        """Run RAGAS evaluation and return a dict of metric scores.

        Returns an empty dict on failure (evaluation is non-critical).
        """
        if not self._initialised:
            return {}

        try:
            from datasets import Dataset
            from langchain_ollama import ChatOllama
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness

            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.0,
            )

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=llm,
            )

            scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            logger.info("RAGAS scores: {}", scores)
            return scores
        except Exception:
            logger.exception("RAGAS evaluation failed")
            return {}
