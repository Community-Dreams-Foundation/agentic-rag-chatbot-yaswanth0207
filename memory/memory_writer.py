"""Backward-compatible wrapper around MemoryGraph.

Provides a simple ``process()`` API that hides the LangGraph internals
from callers that only need a boolean result.
"""

from __future__ import annotations

from loguru import logger

from memory.memory_graph import MemoryGraph


class MemoryWriter:
    """Convenience façade for the agentic memory decision flow."""

    def __init__(self) -> None:
        logger.info("Initialising MemoryWriter …")
        self._graph = MemoryGraph()
        logger.info("MemoryWriter ready")

    def process(self, user_message: str, assistant_response: str) -> tuple[bool, str]:
        """Evaluate and optionally persist a memory fact.

        Returns:
            Tuple of (was_written: bool, target: str).
        """
        written, target = self._graph.run(user_message, assistant_response)
        logger.info("MemoryWriter result: written={}, target={}", written, target)
        return written, target
