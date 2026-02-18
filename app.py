"""Streamlit UI for the Agentic RAG Chatbot.

Provides document upload & indexing, chat with grounded citations,
agentic memory persistence, weather analysis, and optional RAGAS evaluation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Environment & page config
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ollama check
# ---------------------------------------------------------------------------

try:
    import requests as _req
    _ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    _req.get(f"{_ollama_host}/api/tags", timeout=3)
except Exception:
    st.error("🦙 Ollama is not running. Please start it with `ollama serve` and restart.")
    st.stop()

# ---------------------------------------------------------------------------
# Lazy component initialisation (cached in session state)
# ---------------------------------------------------------------------------


def _init_components() -> None:
    """Lazily initialise heavy components once per session."""
    if "components_ready" in st.session_state:
        return

    with st.spinner("🔧 Loading RAG components …"):
        from memory.memory_writer import MemoryWriter
        from rag.answerer import RAGAnswerer
        from rag.evaluator import RAGEvaluator
        from rag.ingestor import DocumentIngestor
        from rag.reranker import FlashRankReranker
        from rag.retriever import HybridRetriever
        from tools.weather_tool import WeatherTool

        st.session_state["ingestor"] = DocumentIngestor()
        st.session_state["retriever"] = HybridRetriever()
        st.session_state["reranker"] = FlashRankReranker()
        st.session_state["answerer"] = RAGAnswerer()
        st.session_state["evaluator"] = RAGEvaluator()
        st.session_state["memory_writer"] = MemoryWriter()
        st.session_state["weather_tool"] = WeatherTool()
        st.session_state["components_ready"] = True


_init_components()

# Shorthand accessors
ingestor = st.session_state["ingestor"]
retriever = st.session_state["retriever"]
reranker = st.session_state["reranker"]
answerer = st.session_state["answerer"]
evaluator = st.session_state["evaluator"]
memory_writer = st.session_state["memory_writer"]
weather_tool = st.session_state["weather_tool"]

# Session defaults
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_sources", {})

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # ── Document Manager ──────────────────────────────────────────────
    st.header("📁 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("📥 Index Files", disabled=not uploaded_files, use_container_width=True):
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            with st.spinner(f"Indexing **{uf.name}** …"):
                chunks = ingestor.ingest_file(tmp_path, original_filename=uf.name)
                retriever.build_bm25_index()
                st.session_state["indexed_sources"][uf.name] = chunks
                st.success(f"✅ Indexed **{chunks}** chunks from **{uf.name}**")

            os.unlink(tmp_path)

    with st.expander("📂 Indexed Documents"):
        sources = st.session_state["indexed_sources"]
        if sources:
            for name, count in sources.items():
                st.markdown(f"- **{name}** — `{count}` chunks")
        else:
            st.caption("No documents indexed yet.")

    if st.button("🗑️ Clear All Documents", use_container_width=True):
        ingestor.clear()
        st.session_state["indexed_sources"] = {}
        st.session_state["messages"] = []
        st.toast("Cleared all documents and chat history.")

    st.divider()

    # ── Weather Analysis ──────────────────────────────────────────────
    st.header("🌤️ Weather Analysis")
    city = st.text_input("City Name", placeholder="e.g. Tokyo, Austin, London")
    days = st.slider("Forecast Days", min_value=1, max_value=16, value=7)

    if st.button("🔍 Analyze Weather", disabled=not city, use_container_width=True):
        with st.spinner(f"Fetching weather for **{city}** …"):
            result = weather_tool.run(city, days=days)

        st.caption(f"📍 Detected: {result.location}")
        st.info(result.explanation)

        if result.daily_summary:
            df = pd.DataFrame([s.model_dump() for s in result.daily_summary])

            st.subheader("🌡️ Daily Average Temperature")
            st.line_chart(df.set_index("date")["avg_temp"])

            st.subheader("🌧️ Daily Precipitation")
            st.bar_chart(df.set_index("date")["total_precipitation"])

            anomalies = [s.date for s in result.daily_summary if s.is_anomaly]
            if anomalies:
                st.warning(f"⚠️ Anomaly days detected: {', '.join(anomalies)}")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────
    st.header("⚙️ Settings")
    alpha = st.slider(
        "Hybrid Search Balance (BM25 ↔ Semantic)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )
    top_k = st.slider("Retrieved Chunks", min_value=3, max_value=10, value=5)
    show_eval = st.toggle("Show RAGAS Evaluation Scores", value=False)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🤖 Agentic RAG Chatbot")
st.caption("Upload documents and ask questions with source citations")

# Chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} Citation(s)"):
                for cit in msg["citations"]:
                    st.info(
                        f"**{cit['source']}** (page {cit['page_number']})  \n"
                        f"chunk: `{cit['chunk_id']}`  \n"
                        f"_{cit['snippet'][:200]}_"
                    )
        if msg.get("eval_scores"):
            cols = st.columns(len(msg["eval_scores"]))
            for col, (metric, score) in zip(cols, msg["eval_scores"].items()):
                col.metric(metric, f"{score:.2f}")

# User input
if prompt := st.chat_input("Ask a question about your documents …"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Update hybrid weights if changed
    retriever.set_weights(alpha)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving and reasoning …"):
            chunks = retriever.hybrid_search(prompt, top_k=top_k)
            reranked = reranker.rerank(prompt, chunks, top_k=top_k)
            response = answerer.answer(prompt, reranked)

        # Memory (run before display so we can adjust the answer)
        with st.spinner("💾 Checking memory …"):
            written, target = memory_writer.process(prompt, response.answer)
        if written:
            answerer.reload_memory()

        no_doc_answer = "I don't have enough information" in response.answer
        display_answer = response.answer
        if written and no_doc_answer:
            display_answer = (
                f"Thanks for sharing! I've noted that in my **{target}** memory "
                f"and will remember it for future conversations.\n\n"
                f"*(That said, this isn't related to your uploaded documents — "
                f"feel free to ask me anything about them!)*"
            )

        st.markdown(display_answer)

        if written:
            st.success(f"💾 Saved to **{target}** memory")

        citation_dicts = [c.model_dump() for c in response.citations]
        if response.citations:
            with st.expander(f"📎 {len(response.citations)} Citation(s)"):
                for cit in response.citations:
                    st.info(
                        f"**{cit.source}** (page {cit.page_number})  \n"
                        f"chunk: `{cit.chunk_id}`  \n"
                        f"_{cit.snippet[:200]}_"
                    )

        # RAGAS evaluation
        eval_scores: dict = {}
        if show_eval and response.citations:
            with st.spinner("📊 Running RAGAS evaluation …"):
                contexts = [c.text for c in reranked]
                eval_scores = evaluator.evaluate(prompt, response.answer, contexts)
            if eval_scores:
                cols = st.columns(len(eval_scores))
                for col, (metric, score) in zip(cols, eval_scores.items()):
                    col.metric(metric, f"{score:.2f}")

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": display_answer,
                "citations": citation_dicts,
                "eval_scores": eval_scores,
            }
        )
