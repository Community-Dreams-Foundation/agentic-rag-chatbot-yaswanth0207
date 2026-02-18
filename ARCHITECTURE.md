# Architecture Overview

## Goal

A Streamlit-based RAG chatbot that demonstrates three AI-first product features:
file-grounded Q&A with inline citations, durable selective memory written to
markdown, and safe Open-Meteo time-series analysis with computed analytics.

---

## High-Level Flow

```
Upload ──▶ Parse ──▶ Chunk ──▶ Embed ──▶ ChromaDB
                                              │
Query ──▶ BM25 + Semantic Hybrid ──▶ Rerank ──▶ Ollama/Llama 3.2 (grounded answer + citations)
                                                       │
                                              LangGraph Memory Decision
                                              (analyze → dedup → write)
```

---

## 1) Ingestion (Upload → Parse → Chunk)

**Supported inputs:** PDF, TXT, Markdown — uploaded via Streamlit file uploader.

**Parsing approach:**
- **PDF**: `pdfplumber` extracts text page-by-page with layout awareness. Tables
  are detected and formatted as tab-separated text, then appended to the page
  content so tabular data is queryable.
- **TXT / MD**: Direct file read into a single LlamaIndex `Document`.

**Chunking strategy:**
- LlamaIndex `SentenceSplitter` with **400-token chunks** and **50-token overlap**.
- Splits on sentence boundaries to preserve semantic coherence within each chunk.

**Metadata captured per chunk:**
| Field | Source | Example |
|---|---|---|
| `source` | Original filename | `report.pdf` |
| `page_number` | PDF page (0 for TXT/MD) | `3` |
| `chunk_id` | UUID-8 assigned after splitting | `a3f1c92b` |
| `file_type` | File extension | `pdf` |

---

## 2) Indexing / Storage

**Vector store:** ChromaDB (persistent client at `./chroma_db`).
- Chosen for zero-setup: no cloud account, no external service.
- Judges can `git clone && run` immediately.

**Persistence:** ChromaDB `PersistentClient` writes to disk automatically.
Volumes are mounted in Docker for container persistence.

**Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`) — 384-dimensional vectors,
3× faster cold start than `sentence-transformers`.

**Lexical index (BM25):** Yes — LangChain `BM25Retriever` is built from all
ChromaDB documents at startup and after each new ingestion. This catches exact
entity names and technical terms that dense embeddings may miss.

---

## 3) Retrieval + Grounded Answering

**Retrieval method:**

```
Query ─┬──▶ BM25Retriever (keyword, top-10) ─┐
       │                                       ├──▶ EnsembleRetriever (weights 0.5 / 0.5)
       └──▶ ChromaDB semantic (top-10)       ─┘            │
                                                            ▼
                                                   FlashRank Reranker
                                                   (ms-marco-MiniLM-L-12-v2)
                                                            │
                                                       Top-K Chunks (default 5)
```

- Hybrid fusion via LangChain `EnsembleRetriever` with UI-configurable BM25 ↔
  semantic weight slider.
- FlashRank cross-encoder reranks fused candidates — lightweight, no GPU, <50 ms.

**How citations are built:**
- The system prompt instructs the LLM to cite every claim inline as
  `[source: <filename>, chunk: <chunk_id>]`.
- After generation, a regex extracts all citation markers and matches them back
  to the original chunks to build structured `Citation` objects containing:
  - **source**: original filename
  - **locator**: `chunk_id` (UUID-8)
  - **snippet**: first 200 chars of the matched chunk text
  - **page_number**: from chunk metadata

**Failure behavior:**
- **No chunks retrieved** (empty collection): Returns a canned refusal —
  *"I don't have enough information in the uploaded documents to answer this
  question."* — with zero citations.
- **Chunks retrieved but irrelevant**: The system prompt says "Do NOT guess,
  speculate, or use general knowledge" and "If the context chunks do NOT contain
  the answer, respond EXACTLY with the refusal phrase." No fake citations are
  produced (Rule 4: "Do NOT produce citations pointing to information not in
  the chunks").
- **LLM error**: Caught by try/except — returns a graceful error message.

---

## 4) Memory System (Selective)

**What counts as "high-signal" memory:**
- User job title or role (e.g. "Project Finance Analyst")
- Stated preferences (e.g. "prefers weekly summaries on Mondays")
- Company-wide facts (e.g. "migrated from AWS to GCP")
- Recurring workflows (e.g. "runs batch ETL every Monday morning")

**What we explicitly do NOT store:**
- Raw conversation text or transcript dumps
- Secrets, passwords, API keys, or sensitive credentials
- Vague or low-value statements
- Anything the user didn't clearly state themselves

**How we decide when to write — LangGraph state machine:**

```
          ┌───────────┐
START ──▶ │  analyze   │──▶ should_write=False ──▶ END
          └─────┬─────┘
                │ should_write=True
                ▼
          ┌──────────────┐
          │ deduplicate   │──▶ DUPLICATE ──▶ END
          └──────┬───────┘
                 │ NEW fact
                 ▼
          ┌───────────┐
          │   write    │──▶ END
          └───────────┘
```

1. **Analyze node**: Ollama (Llama 3.2) + robust JSON extraction produces a structured
   `MemoryDecision` with fields: `should_write`, `target` (user/company/none),
   `summary` (concise standalone fact), `confidence` (0.0–1.0).
2. **Deduplicate node**: Reads the existing target memory file and uses keyword-overlap
   similarity (≥60% threshold) to detect duplicates. If duplicate → skip.
3. **Write node**: Appends only if `should_write=True` AND `confidence ≥ 0.7`.

**Format written to:**
- `USER_MEMORY.md` — personal facts (job, preferences, habits)
- `COMPANY_MEMORY.md` — organizational learnings (tools, migrations, processes)
- Format: `- <concise standalone fact>` (one bullet per entry)
- Both files are injected into the RAG system prompt so future answers are
  personalized.

---

## 5) Optional: Safe Tooling (Open-Meteo)

**Tool interface shape:**
- `WeatherTool.run(city_name, days)` → `WeatherAnalysis`
  (Pydantic model with daily summaries, overall stats, and LLM explanation).
- Auto-geocodes city name to coordinates via Open-Meteo Geocoding API.
- Exposed in the Streamlit sidebar with city name input and day slider.

**Safety boundaries:**

| Boundary | Implementation |
|---|---|
| **Network timeout** | 15-second `requests.get()` timeout on Open-Meteo API |
| **No API key required** | Open-Meteo is free — no credential exposure risk |
| **Error isolation** | All API calls wrapped in try/except; failures return a safe fallback `WeatherAnalysis` with empty data and an error explanation |
| **No arbitrary code exec** | Analytics use pure Python + `math` module only — no `exec()`, `eval()`, or dynamic imports |
| **Restricted scope** | Tool only calls `api.open-meteo.com`; no user-controlled URLs |
| **Data validation** | All outputs are Pydantic-validated before display |

**Analytics computed:**
- Daily aggregates: avg/max/min temperature, total precipitation, avg windspeed
- Rolling 3-day average of daily temperatures
- Volatility: population standard deviation of daily averages
- Anomaly detection: days where avg temp deviates > 1.5σ from mean
- Ollama (Llama 3.2) generates a friendly 3–4 paragraph explanation of findings

---

## Tradeoffs & Next Steps

### Why this design?

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Vector DB | ChromaDB (local) | Pinecone / Weaviate | Zero-setup for judges — no cloud accounts needed |
| LLM | Ollama (Llama 3.2, local) | GPT-4o / Gemini / Claude | Fully local — zero cost, no API keys, runs offline |
| Embeddings | FastEmbed (bge-small-en-v1.5) | sentence-transformers | 3× faster cold start, smaller binary |
| Reranker | FlashRank | cross-encoder/ms-marco | Lighter, no GPU, <50 ms per batch |
| Memory | LangGraph state machine | Simple if/else | Explicit graph is auditable, testable, extensible |
| Retrieval | Hybrid BM25 + semantic | Semantic only | BM25 catches exact entity names embedders miss |
| Injection defense | System prompt rules | Input sanitization | Defence-in-depth at the prompt layer |

### What we would improve with more time:

- **pgvector** for production-grade vector storage with SQL filtering
- **Streaming responses** via Streamlit's `st.write_stream` for better UX latency
- **Multi-user isolation** with session-scoped ChromaDB collections
- **Ground-truth evaluation harness** with gold answers for automated scoring
- **Tool-use agent loop** using LangGraph to auto-detect weather queries from chat
- **Input sanitization layer** to complement prompt-level injection defense
- **Chunk-level confidence scoring** to suppress low-relevance chunks before LLM
