"""Scripted engine for tests and for demonstrating the pipeline.

Answers are supplied verbatim, so a test can construct an engine that cites
correctly, one that cites a plausible-but-wrong source, and one that omits
citations entirely, then assert the scorer tells them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import Citation, Question
from .base import build_record


@dataclass
class MockEngine:
    """Returns pre-written (answer_text, citations) keyed by question id."""

    script: dict[str, tuple[str, list[tuple[str, str]]]] = field(default_factory=dict)
    default_answer: str = "No relevant source found."
    name: str = "mock"

    def answer(self, question: Question):
        answer_text, raw_citations = self.script.get(
            question.id, (self.default_answer, [])
        )
        citations = [
            Citation(url=url, snippet=snippet) for url, snippet in raw_citations
        ]
        return build_record(
            question=question,
            engine_name=self.name,
            answer_text=answer_text,
            citations=citations,
            latency_s=0.0,
        )
