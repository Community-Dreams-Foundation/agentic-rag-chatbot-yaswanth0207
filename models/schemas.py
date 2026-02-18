"""Pydantic v2 data models for the Agentic RAG Chatbot.

All inter-module communication uses these strongly-typed schemas to ensure
data integrity, serialization safety, and self-documenting interfaces.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# RAG Pipeline Models
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A single text chunk with provenance metadata."""

    text: str
    source: str
    chunk_id: str
    page_number: int = 0
    score: float = 0.0  # Reranker score
    bm25_score: float = 0.0
    semantic_score: float = 0.0


class Citation(BaseModel):
    """An inline citation extracted from the LLM answer."""

    source: str
    chunk_id: str
    snippet: str
    page_number: int = 0


class RAGResponse(BaseModel):
    """Structured response from the RAG answering pipeline."""

    answer: str
    citations: list[Citation]
    retrieved_chunks: int = 0


# ---------------------------------------------------------------------------
# Memory Models
# ---------------------------------------------------------------------------

class MemoryDecision(BaseModel):
    """LLM-driven decision on whether to persist a new memory fact."""

    should_write: bool
    target: Literal["user", "company", "none"]
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Weather Tool Models
# ---------------------------------------------------------------------------

class WeatherDaySummary(BaseModel):
    """Aggregated weather statistics for a single day."""

    date: str
    avg_temp: float
    max_temp: float
    min_temp: float
    total_precipitation: float
    avg_windspeed: float
    is_anomaly: bool = False


class WeatherAnalysis(BaseModel):
    """Complete weather analysis result for a location and time range."""

    location: str
    country: str = ""
    timezone: str = ""
    period_days: int
    daily_summary: list[WeatherDaySummary]
    overall: dict
    explanation: str


# ---------------------------------------------------------------------------
# Sanity-Check / Evaluation Models
# ---------------------------------------------------------------------------

class SanityOutput(BaseModel):
    """Schema for the artefact produced by ``make sanity``."""

    implemented_features: list[str]
    qa: list[dict]
    demo: dict
