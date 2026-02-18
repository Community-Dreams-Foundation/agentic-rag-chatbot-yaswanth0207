"""Hybrid BM25 + semantic retrieval using LangChain EnsembleRetriever.

Combines keyword-based BM25 scoring with ChromaDB dense-vector search,
fusing results with configurable weights for best-of-both-worlds retrieval.
"""

from __future__ import annotations

import os
import uuid

import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from loguru import logger

from models.schemas import Chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "rag_documents"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_SEMANTIC_WEIGHT = 0.5

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class _FastEmbedLangChain:
    """Minimal LangChain-compatible embedding wrapper around FastEmbed."""

    def __init__(self) -> None:
        self._model = FastEmbedEmbedding(model_name=EMBED_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._model.get_text_embedding(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._model.get_query_embedding(text)


class HybridRetriever:
    """Hybrid BM25 + semantic retriever backed by ChromaDB."""

    def __init__(self, bm25_weight: float = DEFAULT_BM25_WEIGHT) -> None:
        logger.info("Initialising HybridRetriever …")

        self._chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
        )

        self._embed_fn = _FastEmbedLangChain()
        self._lc_vectorstore = Chroma(
            client=self._chroma_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
        )

        self._bm25_weight = bm25_weight
        self._semantic_weight = 1.0 - bm25_weight
        self._bm25_retriever: BM25Retriever | None = None
        self._ensemble: EnsembleRetriever | None = None

        self.build_bm25_index()
        logger.info("HybridRetriever ready")

    # ------------------------------------------------------------------
    # BM25 index construction
    # ------------------------------------------------------------------

    def build_bm25_index(self) -> None:
        """Fetch all docs from ChromaDB and build / rebuild the BM25 index."""
        try:
            results = self._collection.get(
                include=["documents", "metadatas"],
            )
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []
            ids = results.get("ids") or []

            if not documents:
                logger.warning("ChromaDB collection is empty – BM25 not built")
                self._bm25_retriever = None
                self._ensemble = None
                return

            lc_docs = [
                LCDocument(
                    page_content=doc,
                    metadata={**(metadatas[i] if i < len(metadatas) else {}), "id": ids[i]},
                )
                for i, doc in enumerate(documents)
            ]

            self._bm25_retriever = BM25Retriever.from_documents(lc_docs, k=10)
            semantic_retriever = self._lc_vectorstore.as_retriever(
                search_kwargs={"k": 10},
            )
            self._ensemble = EnsembleRetriever(
                retrievers=[self._bm25_retriever, semantic_retriever],
                weights=[self._bm25_weight, self._semantic_weight],
            )
            logger.info("BM25 index built with {} documents", len(lc_docs))
        except Exception:
            logger.exception("Failed to build BM25 index")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Run hybrid BM25 + semantic search and return Chunk objects."""
        if self._ensemble is None:
            logger.warning("Ensemble retriever not ready – returning empty")
            return []

        try:
            results = self._ensemble.invoke(query)
            chunks = self._docs_to_chunks(results[:top_k])
            logger.info("Hybrid search '{}' → {} chunks", query[:60], len(chunks))
            return chunks
        except Exception as exc:
            if "does not exist" in str(exc):
                logger.warning("Stale collection reference — resetting retriever")
                self.reset()
                return []
            logger.exception("Hybrid search failed for '{}'", query[:60])
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _docs_to_chunks(docs: list[LCDocument]) -> list[Chunk]:
        """Convert LangChain Documents to our Chunk schema."""
        chunks: list[Chunk] = []
        for doc in docs:
            meta = doc.metadata or {}
            chunks.append(
                Chunk(
                    text=doc.page_content,
                    source=meta.get("source", "unknown"),
                    chunk_id=meta.get("chunk_id", meta.get("id", str(uuid.uuid4())[:8])),
                    page_number=int(meta.get("page_number", 0)),
                )
            )
        return chunks

    def reset(self) -> None:
        """Re-bind to the (possibly recreated) ChromaDB collection and clear indexes."""
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
        )
        self._lc_vectorstore = Chroma(
            client=self._chroma_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
        )
        self._bm25_retriever = None
        self._ensemble = None
        self.build_bm25_index()
        logger.info("HybridRetriever reset — re-bound to collection")

    def set_weights(self, bm25_weight: float) -> None:
        """Update fusion weights and rebuild the ensemble retriever."""
        self._bm25_weight = bm25_weight
        self._semantic_weight = 1.0 - bm25_weight
        self.build_bm25_index()
