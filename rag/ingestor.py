"""Document ingestion pipeline using LlamaIndex, ChromaDB, and FastEmbed.

Handles PDF / TXT / MD parsing, sentence-level chunking, embedding via
FastEmbed (all-MiniLM-L6-v2), and persistent vector storage in ChromaDB.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import chromadb
import pdfplumber
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "rag_documents"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class DocumentIngestor:
    """Parses, chunks, embeds, and stores documents in ChromaDB."""

    def __init__(self) -> None:
        logger.info("Initialising DocumentIngestor …")

        self._chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
        )

        self._embed_model = FastEmbedEmbedding(model_name=EMBED_MODEL)
        Settings.embed_model = self._embed_model

        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )
        self._splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info("DocumentIngestor ready (ChromaDB → {})", CHROMA_PERSIST_DIR)

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> list[Document]:
        """Parse a single file into LlamaIndex Document objects."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        filename = path.name

        if suffix == ".pdf":
            return self._parse_pdf(file_path, filename)
        return self._parse_text(file_path, filename)

    def _parse_pdf(self, file_path: str, filename: str) -> list[Document]:
        """Extract text and tables from a PDF using pdfplumber."""
        docs: list[Document] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    tables = page.extract_tables() or []
                    for table in tables:
                        rows = ["\t".join(str(c) for c in row if c) for row in table]
                        text += "\n\n[TABLE]\n" + "\n".join(rows)
                    if text.strip():
                        docs.append(
                            Document(
                                text=text,
                                metadata={
                                    "source": filename,
                                    "page_number": page_num,
                                    "file_type": "pdf",
                                },
                            )
                        )
            logger.info("Parsed PDF '{}': {} page(s)", filename, len(docs))
        except Exception:
            logger.exception("Failed to parse PDF '{}'", filename)
        return docs

    def _parse_text(self, file_path: str, filename: str) -> list[Document]:
        """Read a plain-text or Markdown file."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            logger.info("Parsed text file '{}': {} chars", filename, len(content))
            return [
                Document(
                    text=content,
                    metadata={
                        "source": filename,
                        "page_number": 0,
                        "file_type": Path(file_path).suffix.lstrip("."),
                    },
                )
            ]
        except Exception:
            logger.exception("Failed to read '{}'", filename)
            return []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str, original_filename: str | None = None) -> int:
        """Parse, chunk, embed, and store a file. Returns chunk count."""
        documents = self.parse_file(file_path)

        if original_filename:
            for doc in documents:
                doc.metadata["source"] = original_filename
        if not documents:
            logger.warning("No content extracted from '{}'", file_path)
            return 0

        nodes = self._splitter.get_nodes_from_documents(documents)
        for node in nodes:
            node.metadata["chunk_id"] = str(uuid.uuid4())[:8]

        VectorStoreIndex(
            nodes=nodes,
            storage_context=self._storage_context,
            show_progress=False,
        )

        chunk_count = self._collection.count()
        logger.info("Ingested '{}' → {} total chunks in collection", file_path, chunk_count)
        return chunk_count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_indexed_sources(self) -> list[str]:
        """Return sorted list of unique source filenames in ChromaDB."""
        try:
            results = self._collection.get(include=["metadatas"])
            sources: set[str] = set()
            for meta in (results.get("metadatas") or []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            logger.exception("Failed to query indexed sources")
            return []

    def clear(self) -> None:
        """Delete the entire collection and recreate it."""
        self._chroma_client.delete_collection(CHROMA_COLLECTION)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )
        logger.info("Cleared all documents from ChromaDB")
