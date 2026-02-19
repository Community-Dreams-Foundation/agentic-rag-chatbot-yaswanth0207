"""Query rewriting: turn the current message (and conversation) into a standalone search query."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from loguru import logger

OLLAMA_MODEL = "llama3.2"

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search-query rewriter. Given a user's latest message and optional prior conversation, output a single standalone search query that would find relevant document chunks.

Rules:
- Output ONLY the search query: one short sentence, no quotes, no explanation.
- If the message is already standalone (e.g. "What are the key assumptions?"), return it unchanged or slightly cleaned.
- If the message is a follow-up (e.g. "What about the limitations?", "Who led it?"), turn it into a full question using the conversation context.
- Keep the query concise and focused on document retrieval (names, concepts, facts)."""),
    ("human", """Prior conversation:
{conversation_context}

Current user message: {query}

Standalone search query:"""),
])


class QueryRewriter:
    """Rewrites the user's message into a standalone search query for retrieval."""

    def __init__(self) -> None:
        logger.info("Initialising QueryRewriter (model={})", OLLAMA_MODEL)
        self._llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
        self._chain = REWRITE_PROMPT | self._llm
        logger.info("QueryRewriter ready")

    def rewrite(self, query: str, conversation_context: str = "") -> str:
        """Return a standalone search query. On failure, returns the original query."""
        if not query or not query.strip():
            return query
        try:
            ctx = conversation_context.strip() or "(No prior messages)"
            response = self._chain.invoke({
                "query": query.strip(),
                "conversation_context": ctx[:1500],
            })
            text = (response.content if hasattr(response, "content") else str(response)).strip()
            if not text:
                return query.strip()
            text = text.split("\n")[0].strip()
            if len(text) > 500:
                text = text[:500]
            logger.info("Query rewrite: '{}' -> '{}'", query[:50], text[:50])
            return text
        except Exception:
            logger.warning("Query rewrite failed, using original")
            return query.strip()
