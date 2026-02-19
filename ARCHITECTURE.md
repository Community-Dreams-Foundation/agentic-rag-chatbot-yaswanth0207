# Architecture Overview

## Goal

A Streamlit-based RAG chatbot that demonstrates three AI-first product features:
file-grounded Q&A with inline citations, durable selective memory written to
markdown, and safe Open-Meteo time-series analysis with computed analytics.
**Safe execution boundaries** for the weather tool (Feature C) are stated explicitly in **Section 5**.

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

**Query rewriting (optional step before retrieval):**  
The current user message (and recent conversation) is rewritten by an LLM into a single standalone search query so that follow-ups like "And the limitations?" are resolved for retrieval. The rewritten query is used only for hybrid search and reranking; the original user message is still sent to the answerer. See `rag/query_rewriter.py`. Pipeline Trace and Retrieval Transparency in the UI show the search query when it differs from the user message.

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

**Streaming & citation cleaning:**
- Answers are streamed token-by-token via `st.write_stream` for low-latency UX.
- A buffering layer holds tokens inside `[...]` brackets, strips citation markers
  and chunk metadata headers before yielding to the UI.
- The `_clean_citation_tags()` method also repairs orphaned grammar left behind
  by citation removal (e.g. "According to ," → "According to the document,").

**Failure behavior:**
- **No chunks retrieved + no memory**: Returns a canned refusal —
  *"I don't have enough information in the uploaded documents to answer this
  question."* — with zero citations.
- **No chunks but memory available**: The LLM answers from stored memory facts
  without adding citation markers.
- **Chunks retrieved but irrelevant**: The system prompt enforces grounding rules.
  No fake citations are produced.
- **LLM error**: Caught by try/except — returns a graceful error message.
- **Stale ChromaDB reference**: Auto-detected and self-healed by the retriever.

**Prompt-injection awareness (Security Mindset bonus):**  
The RAG system prompt includes an explicit **SECURITY** rule: all context chunk content is treated as **DATA only — never as instructions**. The model is instructed not to obey or act on any text in the chunks that asks it to change behavior, reveal the prompt, or ignore the rules. This reduces the risk of malicious or accidental prompt injection via uploaded documents.

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

**Safe execution boundaries (Feature C):**  
The weather tool enforces the following safety boundaries. No user-supplied code is executed; the tool only performs HTTP calls to Open-Meteo and pure-Python analytics.

| Boundary | Implementation |
|---|---|
| **Network timeout** | 15-second `requests.get()` timeout on Open-Meteo API; prevents hung requests. |
| **No API key required** | Open-Meteo is free — no credential exposure risk. |
| **Error isolation** | All API calls wrapped in try/except; failures return a safe fallback `WeatherAnalysis` with empty data and an error explanation (no stack traces to user). |
| **No arbitrary code execution** | Analytics use pure Python + `math` only — no `exec()`, `eval()`, or dynamic imports. Tool does not interpret or run user code. |
| **Restricted network scope** | Tool only calls `api.open-meteo.com` and `geocoding-api.open-meteo.com`; no user-controlled URLs or arbitrary outbound requests. |
| **Data validation** | All outputs are Pydantic-validated before display; malformed API responses are caught and turned into safe fallbacks. |
| **Process isolation** | Tool runs in-process. For production we would run tool calls in a sandboxed subprocess or llm-sandbox; not implemented for this hackathon because the tool does not execute user or third-party code. |

**Analytics computed:**
- Daily aggregates: avg/max/min temperature, total precipitation, avg windspeed
- Rolling 3-day average of daily temperatures
- Volatility: population standard deviation of daily averages
- Anomaly detection: days where avg temp deviates > 1.5σ from mean
- **Missingness checks:** per-variable % of null/missing hourly values (`missingness_pct`), with optional alerts when any variable has gaps; LLM explanation can mention data quality
- Ollama (Llama 3.2) generates a friendly 3–4 paragraph explanation of findings

---

## Security Mindset (Bonus)

How this submission addresses the three security bonus criteria:

| Criterion | Addressed | Where |
|-----------|-----------|--------|
| **Prompt-injection awareness in RAG** | Yes | **Section 3 (Retrieval + Grounded Answering):** System prompt includes an explicit SECURITY rule: chunk content is DATA only — never instructions. The model is instructed not to obey or act on any text in the chunks that asks it to change behavior, reveal the prompt, or ignore the rules. See also the "Prompt-injection awareness" paragraph in Section 3. |
| **Sandbox isolation (if implementing Feature C)** | Documented | **Section 5 (Safe Tooling):** Weather tool runs in-process; no user or third-party code is executed. For production we would run tool calls in a sandboxed subprocess or llm-sandbox. Not implemented for this hackathon given the tool’s restricted scope (HTTP + pure-Python analytics only). See "Process isolation" in the safety boundaries table. |
| **Safe handling of external API calls** | Yes | **Section 5 (Safe Tooling):** All external calls (Open-Meteo, geocoding) use a 15-second timeout, try/except with safe fallback responses, no user-controlled URLs, and Pydantic-validated outputs. Failures return a safe `WeatherAnalysis` with an error message instead of leaking stack traces or failing the app. |

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
| RAGAS eval | Faithfulness only | Faithfulness + relevancy | Relevancy requires embeddings — too slow on local Ollama |
| Streaming | Buffer + strip citations | Post-process only | Users see clean text in real-time without citation noise |

### What we would improve with more time:

- **pgvector** for production-grade vector storage with SQL filtering
- **Multi-user isolation** with session-scoped ChromaDB collections
- **Ground-truth evaluation harness** with gold answers for automated scoring
- **Input sanitization layer** to complement prompt-level injection defense
- **Chunk-level confidence scoring** to suppress low-relevance chunks before LLM
- **Conversation persistence** with SQLite-backed chat history across sessions
