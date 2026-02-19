"""Pydantic v2 data models for the Agentic RAG Chatbot."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    source: str
    chunk_id: str
    page_number: int = 0
    score: float = 0.0
    bm25_score: float = 0.0
    semantic_score: float = 0.0


class Citation(BaseModel):
    source: str
    chunk_id: str
    snippet: str
    page_number: int = 0


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: int = 0


class MemoryDecision(BaseModel):
    should_write: bool
    target: Literal["user", "company", "none"]
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class WeatherDaySummary(BaseModel):
    date: str
    avg_temp: float
    max_temp: float
    min_temp: float
    total_precipitation: float
    avg_windspeed: float
    is_anomaly: bool = False


class WeatherAnalysis(BaseModel):
    location: str
    country: str = ""
    timezone: str = ""
    period_days: int
    daily_summary: list[WeatherDaySummary]
    overall: dict
    explanation: str


class SanityOutput(BaseModel):
    implemented_features: list[str]
    qa: list[dict]
    demo: dict
