"""FlashRank cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from flashrank import Ranker, RerankRequest
from loguru import logger

from models.schemas import Chunk

FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


class FlashRankReranker:
    """Reranks retrieved chunks using a FlashRank cross-encoder."""

    def __init__(self) -> None:
        logger.info("Initialising FlashRankReranker (model={})", FLASHRANK_MODEL)
        self._ranker = Ranker(model_name=FLASHRANK_MODEL)
        logger.info("FlashRankReranker ready")

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        if not chunks:
            return []
        try:
            passages = [
                {"id": idx, "text": chunk.text, "meta": {"chunk": chunk}}
                for idx, chunk in enumerate(chunks)
            ]
            results = self._ranker.rerank(RerankRequest(query=query, passages=passages))
            reranked: list[Chunk] = []
            for item in sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]:
                chunk: Chunk = item["meta"]["chunk"]
                chunk.score = float(item["score"])
                reranked.append(chunk)
            logger.info(
                "Reranked {} → {} chunks (top score {:.3f})",
                len(chunks), len(reranked), reranked[0].score if reranked else 0.0,
            )
            return reranked
        except Exception:
            logger.exception("Reranking failed – returning original order")
            return chunks[:top_k]
