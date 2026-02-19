#!/usr/bin/env python3
"""
Evaluation harness: run EVAL_QUESTIONS (or a subset), get answers + citations,
and assert "has citations" and optionally "expected source in citations".

Usage:
  python scripts/run_eval_harness.py                    # Section A only (RAG + citations)
  python scripts/run_eval_harness.py --section A,B       # Sections A and B
  python scripts/run_eval_harness.py --questions 1,2,4  # Specific question numbers
  python scripts/run_eval_harness.py --expected-source sample.txt   # Assert source in citations
  python scripts/run_eval_harness.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
EVAL_QUESTIONS_PATH = ROOT / "EVAL_QUESTIONS.md"
NO_INFO_PHRASE = "I don't have enough information in the uploaded documents to answer this question."


def _parse_eval_questions(path: Path) -> list[dict]:
    """Parse EVAL_QUESTIONS.md into list of {num, question, section_id, expect_citations, expect_refusal}."""
    text = path.read_text(encoding="utf-8")
    entries: list[dict] = []
    # Section header: ## Section X — Title (find last section before each question)
    section_re = re.compile(r"^##\s*Section\s+([A-Z])\s*—", re.MULTILINE)
    # Numbered question: 1. **"question text"**
    question_re = re.compile(r'^(\d+)\.\s*\*\*"([^"]+)"\*\*', re.MULTILINE)
    expect_re = re.compile(r"\*Expect:\s*([^*]+)\*", re.IGNORECASE)

    section_positions = [(m.start(), m.group(1)) for m in section_re.finditer(text)]

    for m in question_re.finditer(text):
        num = int(m.group(1))
        question = m.group(2).strip()
        start = m.end()
        section_id = "A"
        for pos, sid in reversed(section_positions):
            if pos < m.start():
                section_id = sid
                break
        snippet = text[start : start + 400]
        expect_m = expect_re.search(snippet)
        expect_text = (expect_m.group(1) or "").lower()

        expect_citations = ("citation" in expect_text or "citations" in expect_text) and "no fake" not in expect_text
        expect_refusal = (
            "refusal" in expect_text
            or "can't find" in expect_text
            or "cannot find" in expect_text
            or "not in uploaded" in expect_text
            or "don't have enough information" in expect_text
        )
        if section_id == "A":
            expect_citations = True
            expect_refusal = False
        elif section_id == "B":
            expect_refusal = True
            expect_citations = False

        entries.append({
            "num": num,
            "question": question,
            "section_id": section_id,
            "expect_citations": expect_citations,
            "expect_refusal": expect_refusal,
        })
    return entries


def run_harness(
    subset_sections: list[str] | None = None,
    subset_questions: list[int] | None = None,
    expected_sources: list[str] | None = None,
    verbose: bool = False,
) -> tuple[int, int]:
    """Run RAG pipeline for selected questions; assert citations. Returns (passed, failed)."""
    sys.path.insert(0, str(ROOT))

    from rag.answerer import RAGAnswerer
    from rag.ingestor import DocumentIngestor
    from rag.reranker import FlashRankReranker
    from rag.retriever import HybridRetriever

    entries = _parse_eval_questions(EVAL_QUESTIONS_PATH)
    if subset_questions is not None:
        qset = set(subset_questions)
        entries = [e for e in entries if e["num"] in qset]
    elif subset_sections:
        entries = [e for e in entries if e["section_id"] in subset_sections]
    else:
        entries = [e for e in entries if e["section_id"] == "A"]

    if not entries:
        print("No questions selected.")
        return 0, 0

    ingestor = DocumentIngestor()
    retriever = HybridRetriever()
    reranker = FlashRankReranker()
    answerer = RAGAnswerer()

    passed = 0
    failed = 0
    for e in entries:
        num, question = e["num"], e["question"]
        expect_citations = e["expect_citations"]
        expect_refusal = e["expect_refusal"]

        try:
            chunks = retriever.hybrid_search(question, top_k=10)
            reranked = reranker.rerank(question, chunks, top_k=5)
            response = answerer.answer(question, reranked)
        except Exception as ex:
            print(f"  Q{num}: ERROR — {ex}")
            failed += 1
            continue

        citations = response.citations or []
        answer_lower = (response.answer or "").lower()
        has_no_info = NO_INFO_PHRASE.lower() in answer_lower or "don't have enough information" in answer_lower

        ok = True
        msg_parts = []

        if expect_refusal:
            if citations and not has_no_info:
                ok = False
                msg_parts.append("expected refusal / no citations")
            elif not has_no_info and not citations:
                refusal_phrases = ("can't", "cannot", "not in", "don't have", "enough information", "find this")
                if any(p in answer_lower for p in refusal_phrases):
                    ok = True
        else:
            if expect_citations and len(citations) == 0:
                ok = False
                msg_parts.append("expected at least one citation")
            if expected_sources and citations:
                cited_sources = [c.source for c in citations]
                if not any(any(exp in s for s in cited_sources) for exp in expected_sources):
                    ok = False
                    msg_parts.append(f"expected one of {expected_sources} in citations (got {cited_sources})")

        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
        line = f"  Q{num}: {status} — {question[:50]}…"
        if msg_parts:
            line += " [" + "; ".join(msg_parts) + "]"
        print(line)
        if verbose:
            print(f"      answer: {(response.answer or '')[:120]}…")
            print(f"      citations: {[(c.source, c.chunk_id) for c in citations]}")

    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval harness: EVAL_QUESTIONS + assert citations")
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        help="Comma-separated section IDs (e.g. A,B). Default: A only.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated question numbers (e.g. 1,2,4). Overrides --section.",
    )
    parser.add_argument(
        "--expected-source",
        type=str,
        default=None,
        help="Assert this source (substring) appears in citations (e.g. sample.txt).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print answer and citation details")
    args = parser.parse_args()

    subset_sections = [s.strip() for s in args.section.split(",")] if args.section else None
    subset_questions = [int(q.strip()) for q in args.questions.split(",")] if args.questions else None
    expected_sources = [args.expected_source] if args.expected_source else None

    print("Eval harness (EVAL_QUESTIONS + citations)")
    if subset_questions:
        print(f"  Questions: {subset_questions}")
    elif subset_sections:
        print(f"  Sections: {subset_sections}")
    else:
        print("  Sections: A (RAG + citations)")
    if expected_sources:
        print(f"  Expected source in citations: {expected_sources}")
    print()

    passed, failed = run_harness(
        subset_sections=subset_sections,
        subset_questions=subset_questions,
        expected_sources=expected_sources,
        verbose=args.verbose,
    )
    total = passed + failed
    print()
    print(f"Result: {passed}/{total} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
