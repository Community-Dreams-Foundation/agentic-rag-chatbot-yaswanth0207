# 🤖 Agentic RAG Chatbot

A production-quality Retrieval-Augmented Generation chatbot with **agentic memory**, **hybrid search**, **cross-encoder reranking**, and **external tool integration** — built for hackathon judges who appreciate clean architecture and real engineering.

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
                 │  + inline citations             │
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

# Create a Python 3.11 or 3.12 virtual environment (3.13+ not supported by some deps)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install & start Ollama (if not already)
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
source .venv/bin/activate   # if not already active
make sanity
bash scripts/sanity_check.sh
```

---

## Features

### Feature A — RAG Pipeline with Grounded Citations

- **Document ingestion**: PDF (with table extraction), TXT, and Markdown via pdfplumber + LlamaIndex
- **Hybrid retrieval**: BM25 keyword search + semantic vector search fused by EnsembleRetriever
- **Cross-encoder reranking**: FlashRank re-scores candidates for precision
- **Grounded answering**: Ollama (Llama 3.2) with enforced inline citations `[source: X, chunk: Y]`
- **Citation extraction**: Regex-based parsing matched back to original chunks for provenance

### Feature B — Agentic Memory System

- **LangGraph state machine**: `analyze → deduplicate → write` decision flow
- **Structured decisions**: Ollama (Llama 3.2) + robust JSON parsing produces typed `MemoryDecision`
- **Deduplication**: Keyword-overlap deduplication prevents redundant writes
- **Confidence threshold**: Only facts with ≥0.7 confidence are persisted
- **Dual targets**: USER_MEMORY.md (personal) and COMPANY_MEMORY.md (organisational)

### Feature C — Weather Analysis Tool

- **Open-Meteo API**: Free, no-key-required weather data with 15s timeout
- **Pure-Python analytics**: Daily aggregates, rolling 3-day averages, standard deviation, anomaly detection
- **LLM explanation**: Friendly 3–4 paragraph weather narrative via Ollama
- **Interactive charts**: Temperature line charts and precipitation bar charts in the UI

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Interactive web interface |
| RAG Framework | LlamaIndex | Document ingestion, chunking, indexing |
| Chains & Tools | LangChain | Prompt templates, output parsing, retriever fusion |
| Agent Framework | LangGraph | Agentic memory decision state machine |
| Vector Store | ChromaDB | Local persistent vector storage |
| Embeddings | FastEmbed (bge-small-en-v1.5) | Fast, lightweight text embeddings |
| Reranker | FlashRank (ms-marco-MiniLM) | Cross-encoder reranking |
| LLM | Ollama (Llama 3.2, local) | Answer generation, memory analysis, weather explanation |
| PDF Parsing | pdfplumber | Layout-aware PDF text + table extraction |
| Evaluation | RAGAS | Faithfulness & answer relevancy metrics |
| Data Models | Pydantic v2 | Type-safe inter-module data contracts |
| Logging | Loguru | Structured, colourful logging |
| Containerisation | Docker + Compose | Reproducible deployment |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **ChromaDB over Pinecone** | Zero setup — judges can `git clone && run` without cloud accounts |
| **Ollama (local) over cloud APIs** | Zero cost, no API keys, fully offline — judges can run without cloud accounts |
| **FastEmbed over sentence-transformers** | 3× faster cold start, smaller dependency footprint |
| **FlashRank over full cross-encoder** | Lightweight, no GPU needed, <50 ms per batch |
| **LangGraph for memory** | Explicit state machine is auditable, testable, and extensible |
| **Hybrid BM25 + semantic** | BM25 catches exact entity names that embeddings may miss |
| **Pydantic everywhere** | Type safety catches bugs at module boundaries, not in production |
| **Loguru over stdlib logging** | Better formatting, zero config, rotation built in |

---

## Project Structure

```
├── app.py                    # Streamlit UI
├── rag/
│   ├── __init__.py
│   ├── ingestor.py           # Document parsing, chunking, embedding
│   ├── retriever.py          # Hybrid BM25 + semantic retrieval
│   ├── reranker.py           # FlashRank cross-encoder reranking
│   ├── answerer.py           # Grounded answer generation + citations
│   └── evaluator.py          # RAGAS evaluation metrics
├── memory/
│   ├── __init__.py
│   ├── memory_graph.py       # LangGraph agentic memory state machine
│   └── memory_writer.py      # Memory writer facade
├── tools/
│   ├── __init__.py
│   └── weather_tool.py       # Open-Meteo weather analysis + Gemini
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic v2 data models
├── scripts/
│   ├── sanity_check.sh       # End-to-end sanity test runner
│   └── verify_output.py      # Sanity output validator
├── artifacts/                # Generated outputs (sanity_output.json)
├── sample_docs/
│   └── sample.txt            # NovaTech Solutions company profile
├── USER_MEMORY.md            # Persistent user memory
├── COMPANY_MEMORY.md         # Persistent company memory
├── ARCHITECTURE.md           # System architecture document
├── EVAL_QUESTIONS.md         # 20 evaluation questions
├── Makefile                  # Build & run targets
├── Dockerfile                # Container image
├── docker-compose.yml        # Container orchestration
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
└── .gitignore                # Git ignore rules
```
