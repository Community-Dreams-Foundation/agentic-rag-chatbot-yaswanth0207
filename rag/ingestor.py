"""Document ingestion: parse, chunk, embed, and store in ChromaDB."""

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
        self._collection = self._chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
        self._embed_model = FastEmbedEmbedding(model_name=EMBED_MODEL)
        Settings.embed_model = self._embed_model
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        logger.info("DocumentIngestor ready (ChromaDB → {})", CHROMA_PERSIST_DIR)

    def parse_file(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(file_path, path.name)
        return self._parse_text(file_path, path.name)

    def _parse_pdf(self, file_path: str, filename: str) -> list[Document]:
        docs: list[Document] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    for table in page.extract_tables() or []:
                        rows = ["\t".join(str(c) for c in row if c) for row in table]
                        text += "\n\n[TABLE]\n" + "\n".join(rows)
                    if text.strip():
                        docs.append(Document(
                            text=text,
                            metadata={"source": filename, "page_number": page_num, "file_type": "pdf"},
                        ))
            logger.info("Parsed PDF '{}': {} page(s)", filename, len(docs))
        except Exception:
            logger.exception("Failed to parse PDF '{}'", filename)
        return docs

    def _parse_text(self, file_path: str, filename: str) -> list[Document]:
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            logger.info("Parsed text file '{}': {} chars", filename, len(content))
            return [Document(
                text=content,
                metadata={"source": filename, "page_number": 0, "file_type": Path(file_path).suffix.lstrip(".")},
            )]
        except Exception:
            logger.exception("Failed to read '{}'", filename)
            return []

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

        VectorStoreIndex(nodes=nodes, storage_context=self._storage_context, show_progress=False)
        file_chunk_count = len(nodes)
        logger.info("Ingested '{}' → {} chunks (collection total: {})", file_path, file_chunk_count, self._collection.count())
        return file_chunk_count

    def get_indexed_sources(self) -> list[str]:
        try:
            results = self._collection.get(include=["metadatas"])
            sources: set[str] = set()
            for meta in results.get("metadatas") or []:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            logger.exception("Failed to query indexed sources")
            return []

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks for the given source. Returns number of chunks removed."""
        try:
            results = self._collection.get(
                where={"source": {"$eq": source}},
                include=["metadatas"],
            )
            ids = results.get("ids") or []  # ids always returned by Chroma
            if not ids:
                return 0
            self._collection.delete(ids=ids)
            logger.info("Deleted {} chunks for source '{}'", len(ids), source)
            return len(ids)
        except Exception:
            logger.exception("Failed to delete by source '{}'", source)
            return 0

    def get_chunks_for_source(self, source: str) -> list[dict]:
        """Return list of chunk dicts (id, text, source, chunk_id, page_number) for inspection."""
        try:
            results = self._collection.get(
                where={"source": {"$eq": source}},
                include=["documents", "metadatas"],
            )
            ids = results.get("ids") or []
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            out: list[dict] = []
            for i, doc_id in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                out.append({
                    "id": doc_id,
                    "text": docs[i] if i < len(docs) else "",
                    "source": meta.get("source", source),
                    "chunk_id": meta.get("chunk_id", "?"),
                    "page_number": meta.get("page_number", 0),
                })
            return out
        except Exception:
            logger.exception("Failed to get chunks for source '{}'", source)
            return []

    def clear(self) -> None:
        self._chroma_client.delete_collection(CHROMA_COLLECTION)
        self._collection = self._chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        logger.info("Cleared all documents from ChromaDB")
