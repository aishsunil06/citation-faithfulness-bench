"""Answer engine interface.

An engine takes a question and returns prose plus the sources it cited. Every
engine is a *configuration* under test, not just a vendor: "BM25 with 90-word
chunks feeding gpt-4o-mini" is one engine, and the same model with 300-word
chunks is a different one. That is how the benchmark attributes faithfulness
to chunking rather than to the model alone.
"""

from __future__ import annotations

import time
from typing import Protocol

from ..claims import decompose_answer
from ..schema import AnswerRecord, Citation, Question


class AnswerEngine(Protocol):
    name: str

    def answer(self, question: Question) -> AnswerRecord: ...


def build_record(
    question: Question,
    engine_name: str,
    answer_text: str,
    citations: list[Citation],
    latency_s: float | None = None,
    error: str | None = None,
) -> AnswerRecord:
    """Assemble an AnswerRecord and decompose its prose into claims."""
    record = AnswerRecord(
        question_id=question.id,
        engine=engine_name,
        answer_text=answer_text,
        citations=citations,
        latency_s=latency_s,
        error=error,
    )
    record.claims = decompose_answer(question.id, answer_text, citations)
    return record


class _Timer:
    """Context manager yielding elapsed wall-clock seconds."""

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start

    elapsed: float = 0.0


def timer() -> _Timer:
    return _Timer()
