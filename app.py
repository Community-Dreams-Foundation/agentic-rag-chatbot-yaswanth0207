"""Streamlit UI for the Agentic RAG Chatbot."""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from models.schemas import RAGResponse

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CITATION_PATTERN = re.compile(r"\[source:\s*(.+?),\s*chunk(?:_id)?[=:]\s*(.+?)(?:,\s*page[=:]\s*\d+)?\]")
ALL_CITATION_TAGS = re.compile(r"\[source:[^\]]*\]", re.DOTALL)
CHUNK_HEADER_PATTERN = re.compile(r"\[\d+\]\s*source=.+?chunk_id=\S+\s*page=\d+\s*")
WEATHER_PATTERN = re.compile(
    r"\b(?:weather|temperature|forecast|rain|snow|wind|humid|climate)\b.*\b(?:in|for|at|of)\s+([A-Z][a-zA-Z\s]{2,30})\b",
    re.IGNORECASE,
)
NO_INFO_PHRASE = "I don't have enough information in the uploaded documents to answer this question."

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

# Custom CSS for polish
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] { background: rgba(28,131,225,0.05); border-radius: 8px; padding: 8px 12px; }
    div[data-testid="stExpander"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

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
# Lazy component initialisation
# ---------------------------------------------------------------------------


def _init_components() -> None:
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

ingestor = st.session_state["ingestor"]
retriever = st.session_state["retriever"]
reranker = st.session_state["reranker"]
answerer = st.session_state["answerer"]
evaluator = st.session_state["evaluator"]
memory_writer = st.session_state["memory_writer"]
weather_tool = st.session_state["weather_tool"]

st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_sources", {})
st.session_state.setdefault("suggestions", [])
st.session_state.setdefault("eval_history", [])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_weather_query(text: str) -> str | None:
    m = WEATHER_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _strip_citation_markers(text: str) -> str:
    text = ALL_CITATION_TAGS.sub("", text)
    text = CHUNK_HEADER_PATTERN.sub("", text)
    text = re.sub(r"According to\s*,", "According to the document,", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip()


def _strip_trailing_no_info(text: str) -> str:
    lines = text.strip().split("\n")
    if len(lines) > 1 and NO_INFO_PHRASE in lines[-1]:
        return "\n".join(lines[:-1]).strip()
    return text


def _build_conversation_context(messages: list[dict], max_turns: int = 3) -> str:
    recent = [m for m in messages if m["role"] in ("user", "assistant")][-max_turns * 2:]
    if not recent:
        return ""
    parts = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        content = m["content"][:300]
        parts.append(f"{role}: {content}")
    return (
        "--- PRIOR CONVERSATION (for context only — do NOT cite this section) ---\n"
        + "\n".join(parts)
    )


def _generate_suggestions(answerer_obj, chunks_summary: str) -> list[str]:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            "Given this document summary, suggest exactly 4 short questions a user might ask. "
            "Return ONLY the questions, one per line, no numbering.\n\n{summary}"
        )
        chain = prompt | answerer_obj._llm
        result = chain.invoke({"summary": chunks_summary[:1500]})
        raw = result.content or ""
        questions = [q.strip().lstrip("0123456789.-) ") for q in raw.strip().split("\n") if q.strip()]
        return questions[:4]
    except Exception:
        return []


def _render_citations(citations: list[dict], full_chunks: list[dict] | None = None) -> None:
    chunk_map = {}
    if full_chunks:
        chunk_map = {c["chunk_id"]: c["text"] for c in full_chunks}

    for cit in citations:
        src = cit.get("source", "unknown")
        page = cit.get("page_number", 0)
        cid = cit.get("chunk_id", "?")
        label = f"📄 {src} · page {page}" if page else f"📄 {src}"
        with st.expander(label, expanded=False):
            full_text = chunk_map.get(cid, cit.get("snippet", ""))
            st.markdown(full_text if full_text else "_No text available_")
            st.caption(f"chunk: `{cid}`")


def _render_pipeline_trace(trace: dict) -> None:
    cols = st.columns(len(trace))
    for col, (stage, ms) in zip(cols, trace.items()):
        col.metric(stage, f"{ms:.0f} ms")


def _export_chat_markdown(messages: list[dict]) -> str:
    lines = [f"# Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for msg in messages:
        role = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"### {role}\n")
        lines.append(msg["content"] + "\n")
        if msg.get("citations"):
            lines.append("**Citations:**\n")
            for c in msg["citations"]:
                lines.append(f"- {c['source']} (page {c['page_number']}, chunk `{c['chunk_id']}`)")
            lines.append("")
        lines.append("---\n")
    return "\n".join(lines)


def _read_memory_facts(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
        return [l.strip() for l in text.splitlines() if l.strip().startswith("- ")]
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("📥 Index Files", disabled=not uploaded_files, use_container_width=True):
        already_indexed = set(st.session_state["indexed_sources"].keys())
        new_files = [uf for uf in uploaded_files if uf.name not in already_indexed]

        if not new_files:
            st.info("All files already indexed.")
        else:
            sample_texts: list[str] = []
            for uf in new_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name

                with st.spinner(f"Indexing **{uf.name}** …"):
                    chunks = ingestor.ingest_file(tmp_path, original_filename=uf.name)
                    retriever.build_bm25_index()
                    st.session_state["indexed_sources"][uf.name] = chunks
                    st.success(f"✅ Indexed **{chunks}** chunks from **{uf.name}**")

                    docs = ingestor.parse_file(tmp_path)
                    for d in docs[:2]:
                        sample_texts.append(d.text[:500])

                os.unlink(tmp_path)

            if sample_texts:
                with st.spinner("Generating suggested questions …"):
                    suggestions = _generate_suggestions(answerer, "\n".join(sample_texts))
                    st.session_state["suggestions"] = suggestions

    with st.expander("📂 Indexed Documents"):
        sources = st.session_state["indexed_sources"]
        if sources:
            for name, count in sources.items():
                st.markdown(f"- **{name}** — `{count}` chunks")
        else:
            st.caption("No documents indexed yet.")

    if st.button("🗑️ Clear All Documents", use_container_width=True):
        ingestor.clear()
        retriever.reset()
        st.session_state["indexed_sources"] = {}
        st.session_state["messages"] = []
        st.session_state["suggestions"] = []
        st.toast("Cleared all documents and chat history.")

    st.divider()

    # ── Weather Analysis (sidebar) ───────────────────────────────────
    st.header("🌤️ Weather Analysis")
    city_input = st.text_input("City Name", placeholder="e.g. Tokyo, Austin, London")
    forecast_days = st.slider("Forecast Days", min_value=1, max_value=16, value=7)

    if st.button("🔍 Analyze Weather", disabled=not city_input, use_container_width=True):
        with st.spinner(f"Fetching weather for **{city_input}** …"):
            w_result = weather_tool.run(city_input, days=forecast_days)

        st.caption(f"📍 Detected: {w_result.location}")
        st.info(w_result.explanation)

        if w_result.daily_summary:
            wdf = pd.DataFrame([s.model_dump() for s in w_result.daily_summary])

            st.subheader("🌡️ Daily Average Temperature")
            st.line_chart(wdf.set_index("date")["avg_temp"])

            st.subheader("🌧️ Daily Precipitation")
            st.bar_chart(wdf.set_index("date")["total_precipitation"])

            anomalies = [s.date for s in w_result.daily_summary if s.is_anomaly]
            if anomalies:
                st.warning(f"⚠️ Anomaly days: {', '.join(anomalies)}")

    st.divider()

    # ── Memory Viewer (reads fresh from disk every render) ────────────
    st.header("🧠 Memory Viewer")
    mem_tab_user, mem_tab_company = st.tabs(["User", "Company"])
    with mem_tab_user:
        user_facts = _read_memory_facts(Path("USER_MEMORY.md"))
        if user_facts:
            for f in user_facts:
                st.markdown(f)
        else:
            st.caption("No user memories yet.")
    with mem_tab_company:
        co_facts = _read_memory_facts(Path("COMPANY_MEMORY.md"))
        if co_facts:
            for f in co_facts:
                st.markdown(f)
        else:
            st.caption("No company memories yet.")

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
    show_eval = st.toggle(
        "Show RAGAS Evaluation Scores",
        value=st.session_state.get("show_ragas", False),
        key="show_ragas_toggle",
    )
    st.session_state["show_ragas"] = show_eval

    st.divider()

    if st.session_state["messages"]:
        md_export = _export_chat_markdown(st.session_state["messages"])
        st.download_button(
            "📥 Export Chat",
            data=md_export,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🤖 Agentic RAG Chatbot")
st.caption("Upload documents and ask questions · Weather queries auto-detected · Memory persists across sessions")

# Suggested questions
if st.session_state.get("suggestions"):
    cols = st.columns(len(st.session_state["suggestions"]))
    for i, (col, q) in enumerate(zip(cols, st.session_state["suggestions"])):
        if col.button(f"💡 {q}", key=f"suggestion_{i}", use_container_width=True):
            st.session_state["_prefill_query"] = q
            st.rerun()

# Empty state — welcoming UI when no messages yet
if not st.session_state["messages"]:
    st.markdown("")
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 1rem 2rem 1rem; opacity: 0.7;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📄 🔍 🤖</div>
                <h3 style="margin-bottom: 0.5rem;">Welcome!</h3>
                <p>Upload a document in the sidebar, then ask a question below.</p>
                <p style="font-size: 0.85rem; margin-top: 1rem;">
                    Try: <em>"Summarize the main findings"</em> · <em>"What are the key metrics?"</em> · <em>"Weather in Tokyo"</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} Citation(s)"):
                _render_citations(msg["citations"], msg.get("full_chunks"))
        if msg.get("pipeline_trace"):
            with st.expander("⚡ Pipeline Trace"):
                _render_pipeline_trace(msg["pipeline_trace"])
        if msg.get("retrieval_debug"):
            with st.expander("🔍 Retrieval Transparency"):
                df = pd.DataFrame(msg["retrieval_debug"])
                if "rerank_score" in df.columns:
                    df = df.sort_values("rerank_score", ascending=False)
                st.dataframe(df, use_container_width=True)
        if msg.get("eval_scores"):
            with st.expander("📊 RAG Quality Scores"):
                scores = msg["eval_scores"]
                score_cols = st.columns(len(scores))
                for sc, (metric, val) in zip(score_cols, scores.items()):
                    sc.metric(metric, f"{val:.0%}")
                overall = sum(scores.values()) / max(len(scores), 1)
                st.progress(overall, text=f"Overall: {overall:.0%}")
        if msg.get("weather"):
            w = msg["weather"]
            st.info(w.get("explanation", ""))
            if w.get("daily"):
                wdf = pd.DataFrame(w["daily"])
                st.subheader("🌡️ Temperature Trend")
                st.line_chart(wdf.set_index("date")["avg_temp"])
                st.subheader("🌧️ Precipitation")
                st.bar_chart(wdf.set_index("date")["total_precipitation"])

# User input
prefill = st.session_state.pop("_prefill_query", None)
if prompt := st.chat_input("Ask about your documents, or try: 'weather in London' …"):
    pass
elif prefill:
    prompt = prefill

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    city = _detect_weather_query(prompt)

    if city:
        # ── Weather path ──────────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.spinner(f"🌤️ Fetching weather for **{city}** …"):
                result = weather_tool.run(city, days=7)

            st.markdown(f"**Weather for {result.location}:**")
            st.info(result.explanation)

            weather_data: dict = {"explanation": result.explanation, "daily": []}
            if result.daily_summary:
                wdf = pd.DataFrame([s.model_dump() for s in result.daily_summary])
                weather_data["daily"] = [s.model_dump() for s in result.daily_summary]

                st.subheader("🌡️ Temperature Trend")
                st.line_chart(wdf.set_index("date")["avg_temp"])

                st.subheader("🌧️ Precipitation")
                st.bar_chart(wdf.set_index("date")["total_precipitation"])

                anomalies = [s.date for s in result.daily_summary if s.is_anomaly]
                if anomalies:
                    st.warning(f"⚠️ Anomaly days: {', '.join(anomalies)}")

            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"**Weather for {result.location}:** {result.explanation[:200]}…",
                "weather": weather_data,
                "citations": [],
                "eval_scores": {},
            })
    else:
        # ── RAG path ─────────────────────────────────────────────────
        retriever.set_weights(alpha)
        pipeline_trace: dict[str, float] = {}

        with st.chat_message("assistant"):
            # Retrieval
            with st.spinner("🔍 Searching documents …"):
                t0 = time.perf_counter()
                chunks = retriever.hybrid_search(prompt, top_k=top_k)
                pipeline_trace["Retrieval"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                reranked = reranker.rerank(prompt, chunks, top_k=top_k)
                pipeline_trace["Rerank"] = (time.perf_counter() - t0) * 1000

            # Ensure memory is fresh before answering
            answerer.reload_memory()

            # Conversation context
            conv_context = _build_conversation_context(st.session_state["messages"])
            answerer._conversation_context = conv_context or ""

            # Stream the answer with a status indicator
            t0 = time.perf_counter()
            result_container: dict[str, RAGResponse] = {}
            stream_generator = answerer.stream_answer(prompt, reranked, result_container)

            with st.status("🧠 Thinking …", expanded=True) as status:
                raw_display = st.write_stream(stream_generator)
                status.update(label="✅ Done", state="complete", expanded=True)

            pipeline_trace["Generation"] = (time.perf_counter() - t0) * 1000

            response = result_container.get("response")
            if not response:
                response = answerer.answer(prompt, reranked)

            clean_answer = _strip_trailing_no_info(
                _strip_citation_markers(response.answer)
            )

            # Memory
            t0 = time.perf_counter()
            with st.spinner("💾 Checking memory …"):
                written, target = memory_writer.process(prompt, clean_answer)
            pipeline_trace["Memory"] = (time.perf_counter() - t0) * 1000

            if written:
                answerer.reload_memory()

            no_doc_answer = NO_INFO_PHRASE in response.answer and not response.citations
            if written and no_doc_answer:
                st.info(
                    f"💾 Thanks for sharing! I've noted that in my **{target}** memory "
                    f"and will remember it for future conversations."
                )
            elif written:
                st.success(f"💾 Saved to **{target}** memory")

            # Citations
            citation_dicts = [c.model_dump() for c in response.citations]
            full_chunks = [{"chunk_id": c.chunk_id, "text": c.text} for c in reranked]
            if response.citations:
                with st.expander(f"📎 {len(response.citations)} Citation(s)"):
                    _render_citations(citation_dicts, full_chunks)

            # Pipeline trace
            with st.expander("⚡ Pipeline Trace"):
                _render_pipeline_trace(pipeline_trace)

            # Retrieval transparency
            used_chunk_ids = {c.chunk_id for c in response.citations}
            retrieval_debug = [
                {
                    "source": c.source,
                    "chunk_id": c.chunk_id,
                    "cited": "✅" if c.chunk_id in used_chunk_ids else "",
                    "rerank_score": round(c.score, 4),
                    "snippet": c.text[:120],
                }
                for c in reranked
            ]
            if retrieval_debug:
                with st.expander("🔍 Retrieval Transparency"):
                    rdf = pd.DataFrame(retrieval_debug)
                    st.dataframe(rdf, use_container_width=True)

            # RAGAS evaluation
            eval_scores: dict = {}
            if show_eval and reranked:
                with st.spinner("📊 Evaluating faithfulness (up to 90s) …"):
                    try:
                        contexts = [c.text for c in reranked]
                        eval_scores = evaluator.evaluate(prompt, clean_answer, contexts)
                    except Exception:
                        logger.exception("RAGAS evaluation failed")
                        eval_scores = {}

                if eval_scores:
                    st.session_state["eval_history"].append({
                        "question": prompt[:50] + ("…" if len(prompt) > 50 else ""),
                        **eval_scores,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })

                    with st.expander("📊 RAG Quality Scores", expanded=True):
                        score_cols = st.columns(len(eval_scores))
                        for sc, (metric, val) in zip(score_cols, eval_scores.items()):
                            label = metric.replace("_", " ").title()
                            sc.metric(label, f"{val:.0%}")
                        overall = sum(eval_scores.values()) / max(len(eval_scores), 1)
                        st.progress(overall, text=f"Overall Quality: {overall:.0%}")
                        if overall >= 0.8:
                            st.success("Excellent — answer is well-grounded in sources")
                        elif overall >= 0.6:
                            st.warning("Good — mostly grounded in sources")
                        else:
                            st.error("Low — try rephrasing or uploading more relevant docs")
                else:
                    st.caption("⚠️ Evaluation returned no scores — try a different question")

            st.session_state["messages"].append({
                "role": "assistant",
                "content": clean_answer,
                "citations": citation_dicts,
                "full_chunks": full_chunks,
                "eval_scores": eval_scores,
                "retrieval_debug": retrieval_debug,
                "pipeline_trace": pipeline_trace,
            })

    # Force a rerun so sidebar memory viewer refreshes immediately
    if prompt:
        st.rerun()
