"""RAG evaluation using RAGAS faithfulness metric."""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager

from loguru import logger

OLLAMA_MODEL = "llama3.2"
EVAL_TIMEOUT_SECONDS = 90

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@contextmanager
def _timeout(seconds: int):
    """Raise TimeoutError if the block exceeds *seconds* (Unix only; no-op on Windows)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    def _handler(signum, frame):
        raise TimeoutError(f"RAGAS evaluation timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class RAGEvaluator:
    """Evaluates RAG pipeline quality using RAGAS faithfulness."""

    def __init__(self) -> None:
        logger.info("Initialising RAGEvaluator")
        self._initialised = False
        try:
            from ragas.metrics import faithfulness  # noqa: F401
            self._faithfulness = faithfulness
            self._initialised = True
            logger.info("RAGAS metrics loaded")
        except Exception:
            logger.warning("RAGAS metrics could not be loaded – evaluation disabled")

    def evaluate(
        self, question: str, answer: str, contexts: list[str], ground_truth: str = "",
    ) -> dict[str, float]:
        """Run RAGAS evaluation. Returns empty dict on failure (non-critical)."""
        if not self._initialised:
            return {}
        try:
            from datasets import Dataset
            from langchain_ollama import ChatOllama
            from ragas import evaluate
            from ragas.metrics import faithfulness

            def _is_finite_scalar(v):
                try:
                    f = float(v)
                    return f == f  # exclude NaN
                except (TypeError, ValueError):
                    return False

            llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
            data: dict = {
                "user_input": [question],
                "response": [answer],
                "retrieved_contexts": [contexts],
            }
            if ground_truth:
                data["reference"] = [ground_truth]

            with _timeout(EVAL_TIMEOUT_SECONDS):
                result = evaluate(
                    Dataset.from_dict(data),
                    metrics=[faithfulness],
                    llm=llm,
                    show_progress=False,
                )

            if hasattr(result, "_repr_dict"):
                scores = {
                    k: float(v) for k, v in result._repr_dict.items()
                    if _is_finite_scalar(v)
                }
            elif hasattr(result, "scores") and result.scores:
                scores = {
                    k: float(v) for k, v in result.scores[0].items()
                    if _is_finite_scalar(v)
                }
            else:
                scores = {}
            logger.info("RAGAS scores: {}", scores)
            return scores
        except TimeoutError:
            logger.warning("RAGAS evaluation timed out after {}s", EVAL_TIMEOUT_SECONDS)
            return {}
        except Exception:
            logger.exception("RAGAS evaluation failed")
            return {}
