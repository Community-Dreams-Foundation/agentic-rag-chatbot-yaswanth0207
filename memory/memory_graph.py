"""Agentic memory decision flow using LangGraph.

Three-node state machine: analyze → deduplicate → write.
The LLM decides whether a conversation turn contains a fact worth
persisting, deduplicates against existing memory, and appends if confident.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from loguru import logger

from models.schemas import MemoryDecision

OLLAMA_MODEL = "llama3.2"
CONFIDENCE_THRESHOLD = 0.7
USER_MEMORY_PATH = Path("USER_MEMORY.md")
COMPANY_MEMORY_PATH = Path("COMPANY_MEMORY.md")
SIMILARITY_THRESHOLD = 0.6

ANALYZE_PROMPT = """\
You extract facts from conversations to remember for later.

User said: "{user_message}"
Bot replied: "{assistant_response}"

If the user shared a personal fact (job title, preference, habit, schedule, name, location), set target="user".
If the user shared a company/team fact (tools, processes, migrations, team info), set target="company".
If the user just asked a question or said something generic, set target="none".

Reply with ONLY valid JSON, nothing else:
{{"should_write": true, "target": "user", "summary": "one line fact", "confidence": 0.9}}

Examples of what TO save:
- "I'm a data engineer at Google" → {{"should_write": true, "target": "user", "summary": "User is a data engineer at Google", "confidence": 0.95}}
- "I prefer weekly reports on Mondays" → {{"should_write": true, "target": "user", "summary": "User prefers weekly reports on Mondays", "confidence": 0.9}}
- "I run a batch job every Monday morning" → {{"should_write": true, "target": "user", "summary": "User runs a batch job every Monday morning", "confidence": 0.9}}
- "Our team uses Airflow for ETL" → {{"should_write": true, "target": "company", "summary": "Team uses Airflow for ETL pipelines", "confidence": 0.9}}
- "We migrated from AWS to GCP last quarter" → {{"should_write": true, "target": "company", "summary": "Company migrated from AWS to GCP last quarter", "confidence": 0.9}}
- "I'm a Project Finance Analyst" → {{"should_write": true, "target": "user", "summary": "User is a Project Finance Analyst", "confidence": 0.95}}

Examples of what NOT to save:
- "What is NovaTech?" → {{"should_write": false, "target": "none", "summary": "", "confidence": 0.0}}
- "thanks" → {{"should_write": false, "target": "none", "summary": "", "confidence": 0.0}}
- "summarize the document" → {{"should_write": false, "target": "none", "summary": "", "confidence": 0.0}}

Now extract from the conversation above. Reply with ONLY the JSON:"""


class MemoryGraphState(TypedDict):
    user_message: str
    assistant_response: str
    decision: MemoryDecision | None
    written: bool
    error: str | None


class MemoryGraph:
    """LangGraph-based agentic memory decision flow."""

    def __init__(self) -> None:
        logger.info("Building MemoryGraph …")
        self._llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
        self._graph = self._build_graph()
        logger.info("MemoryGraph compiled")

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(MemoryGraphState)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("deduplicate", self._deduplicate_node)
        graph.add_node("write", self._write_node)
        graph.set_entry_point("analyze")
        graph.add_conditional_edges(
            "analyze", self._should_deduplicate,
            {"deduplicate": "deduplicate", "end": END},
        )
        graph.add_edge("deduplicate", "write")
        graph.add_edge("write", END)
        return graph.compile()

    def _analyze_node(self, state: MemoryGraphState) -> dict:
        try:
            prompt = ChatPromptTemplate.from_template(ANALYZE_PROMPT)
            result = (prompt | self._llm).invoke({
                "user_message": state["user_message"],
                "assistant_response": state["assistant_response"],
            })
            decision = self._parse_decision(result.content or "")  # type: ignore[union-attr]
            logger.info("Memory decision: {}", decision)
            return {"decision": decision}
        except Exception as exc:
            logger.exception("Analyze node failed")
            return {
                "decision": MemoryDecision(should_write=False, target="none", summary="", confidence=0.0),
                "error": str(exc),
            }

    @staticmethod
    def _parse_decision(raw: str) -> MemoryDecision:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return MemoryDecision(
                    should_write=bool(data.get("should_write", False)),
                    target=data.get("target", "none"),
                    summary=data.get("summary", ""),
                    confidence=float(data.get("confidence", 0.0)),
                )
            except (json.JSONDecodeError, ValueError):
                pass
        return MemoryDecision(should_write=False, target="none", summary="", confidence=0.0)

    def _deduplicate_node(self, state: MemoryGraphState) -> dict:
        """Check whether the fact already exists via keyword overlap."""
        decision: MemoryDecision = state["decision"]  # type: ignore[assignment]
        try:
            mem_path = self._target_path(decision.target)
            existing = mem_path.read_text(encoding="utf-8") if mem_path.exists() else ""
            existing_facts = [
                line.strip().lstrip("- ").lower()
                for line in existing.splitlines()
                if line.strip().startswith("- ")
            ]
            if not existing_facts:
                return {"decision": decision}

            new_words = set(decision.summary.lower().split())
            for fact in existing_facts:
                fact_words = set(fact.split())
                if not new_words or not fact_words:
                    continue
                overlap = len(new_words & fact_words) / max(len(new_words), len(fact_words))
                if overlap >= SIMILARITY_THRESHOLD:
                    logger.info("Duplicate detected (overlap={:.0%}): '{}' ~ '{}'",
                                overlap, decision.summary, fact)
                    decision.should_write = False
                    return {"decision": decision}
            return {"decision": decision}
        except Exception as exc:
            logger.exception("Dedup node failed")
            return {"decision": decision, "error": str(exc)}

    def _write_node(self, state: MemoryGraphState) -> dict:
        decision: MemoryDecision = state["decision"]  # type: ignore[assignment]
        if not decision.should_write or decision.confidence < CONFIDENCE_THRESHOLD:
            logger.info("Write skipped (should_write={}, conf={:.2f})", decision.should_write, decision.confidence)
            return {"written": False}
        try:
            mem_path = self._target_path(decision.target)
            with open(mem_path, "a", encoding="utf-8") as f:
                f.write(f"- {decision.summary}\n")
            logger.info("Wrote to {}: {}", mem_path.name, decision.summary)
            return {"written": True}
        except Exception as exc:
            logger.exception("Write node failed")
            return {"written": False, "error": str(exc)}

    @staticmethod
    def _should_deduplicate(state: MemoryGraphState) -> str:
        decision = state.get("decision")
        if decision and decision.should_write:  # type: ignore[union-attr]
            return "deduplicate"
        return "end"

    @staticmethod
    def _target_path(target: str) -> Path:
        return COMPANY_MEMORY_PATH if target == "company" else USER_MEMORY_PATH

    def run(self, user_message: str, assistant_response: str) -> tuple[bool, str]:
        """Execute the full memory graph and return (written, target)."""
        initial_state: MemoryGraphState = {
            "user_message": user_message, "assistant_response": assistant_response,
            "decision": None, "written": False, "error": None,
        }
        try:
            result = self._graph.invoke(initial_state)
            written = bool(result.get("written", False))
            target = result.get("decision").target if result.get("decision") else "none"  # type: ignore[union-attr]
            return written, target
        except Exception:
            logger.exception("MemoryGraph execution failed")
            return False, "none"
