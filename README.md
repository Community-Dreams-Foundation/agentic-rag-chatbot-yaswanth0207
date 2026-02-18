# 🤖 Agentic RAG Chatbot

A production-quality Retrieval-Augmented Generation chatbot with **agentic memory**, **hybrid search**, **cross-encoder reranking**, **streaming responses**, and **external tool integration** — fully local, no API keys required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Upload & │  │   Chat    │  │ Weather  │  │   Settings    │  │
│  │  Index   │  │ Interface │  │ Analysis │  │  & Eval       │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └───────────────┘  │
└───────┼──────────────┼──────────────┼────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌───────────────────────────────┐ ┌──────────────┐
│  Ingestor    │ │       RAG Pipeline            │ │ Weather Tool │
│              │ │                               │ │              │
│ pdfplumber   │ │ HybridRetriever               │ │ Open-Meteo   │
│ LlamaIndex   │ │  ├─ BM25 (keyword)            │ │ Pure-Python  │
│ SentenceSplit│ │  └─ Semantic (ChromaDB)        │ │  analytics   │
│ FastEmbed    │ │         │                      │ │ Ollama       │
│ ChromaDB     │ │  EnsembleRetriever (0.5/0.5)  │ │  explanation │
│              │ │         │                      │ └──────────────┘
└──────────────┘ │  FlashRank Reranker            │
                 │         │                      │
                 │  RAGAnswerer (Ollama Llama 3.2) │
                 │  + streaming + citation strip   │
                 │  + memory injection             │
                 └───────────────┬─────────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │    Agentic Memory (LangGraph) │
                 │                               │
                 │  analyze → deduplicate → write │
                 │       MemoryDecision          │
                 │  USER_MEMORY.md               │
                 │  COMPANY_MEMORY.md            │
                 └───────────────────────────────┘
```

---

## Participant Info

| Field | Value |
|---|---|
| **Full Name** | |
| **Email** | |
| **GitHub Username** | |

---

## Video Walkthrough

PASTE YOUR LINK HERE

---

## Quick Start

### Local Development

```bash
# Clone and enter the repo
git clone <repo-url>
cd agentic-rag-chatbot

# Create a Python 3.11 or 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies (all versions pinned)
pip install -r requirements.txt

# Install & start Ollama
# macOS: brew install ollama
ollama serve &          # start the Ollama server
ollama pull llama3.2    # download the model (~2 GB)

# Run the app
streamlit run app.py
# Open http://localhost:8501
```

### Docker

```bash
# Make sure Ollama is running on the host: ollama serve
docker-compose up --build
# Open http://localhost:8501
```

### Sanity Check

```bash
source .venv/bin/activate
make sanity
bash scripts/sanity_check.sh
```

---

## Features

### Feature A — RAG Pipeline with Grounded Citations

- **Document ingestion**: PDF (with table extraction), TXT, and Markdown via pdfplumber + LlamaIndex
- **Hybrid retrieval**: BM25 keyword search + semantic vector search fused by EnsembleRetriever
- **Cross-encoder reranking**: FlashRank re-scores candidates for precision
- **Streaming responses**: Real-time token streaming with `st.write_stream` for low-latency UX
- **Grounded answering**: Ollama (Llama 3.2) with enforced inline citations
- **Clean citation display**: All `[source: ...]` markers stripped from streamed output; clickable citation expanders show full chunk text
- **Duplicate-safe indexing**: Re-uploading the same file skips re-indexing

### Feature B — Agentic Memory System

- **LangGraph state machine**: `analyze → deduplicate → write` decision flow
- **Structured decisions**: Ollama + robust JSON parsing produces typed `MemoryDecision`
- **Deduplication**: Keyword-overlap deduplication prevents redundant writes
- **Confidence threshold**: Only facts with ≥0.7 confidence are persisted
- **Dual targets**: USER_MEMORY.md (personal) and COMPANY_MEMORY.md (organisational)
- **Memory-aware answers**: LLM answers from memory even when no documents are indexed
- **Live memory viewer**: Sidebar panel shows stored facts in real-time

### Feature C — Weather Analysis Tool

- **Open-Meteo API**: Free, no-key-required weather data with 15s timeout
- **Auto-routing**: Weather queries detected from chat input (e.g. "weather in Tokyo")
- **Pure-Python analytics**: Daily aggregates, rolling 3-day averages, anomaly detection
- **LLM explanation**: Friendly weather narrative via Ollama
- **Interactive charts**: Temperature line charts and precipitation bar charts

### Additional Features

- **Conversation-aware follow-ups**: Last 3 turns injected as context for multi-turn dialogue
- **Pipeline trace panel**: Timing breakdown for Retrieval, Rerank, Generation, and Memory stages
- **Retrieval transparency panel**: Shows all retrieved chunks, rerank scores, and which were cited
- **RAGAS evaluation**: Toggle-activated faithfulness scoring with progress bar and quality labels
- **Smart query suggestions**: Auto-generated question buttons after document indexing
- **Export chat**: Download full conversation as Markdown
- **Welcoming empty state**: Clean onboarding UI with example queries

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Interactive web interface with streaming |
| RAG Framework | LlamaIndex | Document ingestion, chunking, indexing |
| Chains & Tools | LangChain | Prompt templates, output parsing, retriever fusion |
| Agent Framework | LangGraph | Agentic memory decision state machine |
| Vector Store | ChromaDB | Local persistent vector storage |
| Embeddings | FastEmbed (bge-small-en-v1.5) | Fast, lightweight text embeddings |
| Reranker | FlashRank (ms-marco-MiniLM) | Cross-encoder reranking |
| LLM | Ollama (Llama 3.2, local) | Answer generation, memory analysis, weather explanation |
| PDF Parsing | pdfplumber | Layout-aware PDF text + table extraction |
| Evaluation | RAGAS | Faithfulness metrics (local, no OpenAI) |
| Data Models | Pydantic v2 | Type-safe inter-module data contracts |
| Logging | Loguru | Structured, colourful logging |
| Containerisation | Docker + Compose | Reproducible deployment |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **ChromaDB over Pinecone** | Zero setup — judges can `git clone && run` without cloud accounts |
| **Ollama (local) over cloud APIs** | Zero cost, no API keys, fully offline |
| **FastEmbed over sentence-transformers** | 3× faster cold start, smaller dependency footprint |
| **FlashRank over full cross-encoder** | Lightweight, no GPU needed, <50 ms per batch |
| **LangGraph for memory** | Explicit state machine is auditable, testable, and extensible |
| **Hybrid BM25 + semantic** | BM25 catches exact entity names that embeddings may miss |
| **RAGAS faithfulness only** | Avoids slow embedding calls; runs fully local via Ollama |
| **Streaming + buffer** | Citation markers buffered and stripped so users see clean text |
| **Pydantic everywhere** | Type safety catches bugs at module boundaries, not in production |

---

## Project Structure

```
├── app.py                    # Streamlit UI — all features orchestrated here
├── rag/
│   ├── __init__.py
│   ├── ingestor.py           # Document parsing, chunking, embedding
│   ├── retriever.py          # Hybrid BM25 + semantic retrieval
│   ├── reranker.py           # FlashRank cross-encoder reranking
│   ├── answerer.py           # Streaming answer generation + citation cleaning
│   └── evaluator.py          # RAGAS faithfulness evaluation
├── memory/
│   ├── __init__.py
│   ├── memory_graph.py       # LangGraph agentic memory state machine
│   └── memory_writer.py      # Memory writer facade
├── tools/
│   ├── __init__.py
│   └── weather_tool.py       # Open-Meteo weather analysis + Ollama explanation
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic v2 data models
├── scripts/
│   ├── run_sanity.py         # End-to-end sanity check runner
│   ├── sanity_check.sh       # Shell wrapper for sanity check
│   └── verify_output.py      # Sanity output JSON validator
├── sample_docs/
│   ├── sample.txt            # NovaTech Solutions company profile
│   └── research_summary.txt  # RAG paper summary (Lewis et al.)
├── USER_MEMORY.md            # Persistent user memory
├── COMPANY_MEMORY.md         # Persistent company memory
├── ARCHITECTURE.md           # Detailed system architecture
├── EVAL_QUESTIONS.md         # 20 evaluation questions
├── Makefile                  # Build & run targets
├── Dockerfile                # Container image
├── docker-compose.yml        # Container orchestration
├── requirements.txt          # Pinned Python dependencies
├── .env.example              # Environment template
└── .gitignore                # Git ignore rules
```
