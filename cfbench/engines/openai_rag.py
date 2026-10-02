"""Generative RAG engine: BM25 retrieval, then an LLM that must cite.

The prompt asks for inline [n] markers because that is what `claims.py` parses,
and because forcing per-sentence attribution is what makes the failure
measurable at all. An engine that cites only at the end of a paragraph cannot
be scored per claim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..retrieval import BM25Index
from ..schema import Citation, Question
from .base import build_record, timer

SYSTEM_PROMPT = """\
Answer the question using only the numbered sources provided.

Rules:
- Put an inline citation marker like [1] at the end of every sentence, naming
  the source that supports that specific sentence.
- A sentence may cite more than one source: [1][3].
- Copy numbers, dates, and names exactly as the sources give them.
- If the sources do not answer the question, say so in one sentence and cite
  nothing. Do not fall back on your own knowledge.
- Three sentences at most.
"""


@dataclass
class OpenAIRAGEngine:
    index: BM25Index
    model: str = "gpt-4o-mini"
    top_k: int = 3
    temperature: float = 0.0
    max_snippet_chars: int = 2000
    name: str = ""
    _label: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            # Chunk geometry is part of the engine identity: comparing engines
            # is meaningless if the name hides the retrieval configuration.
            sample = self.index.chunks[0].text.split() if self.index.chunks else []
            width = len(sample) if sample else 0
            self.name = f"rag-{self.model}-k{self.top_k}-c{width}"

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "OpenAIRAGEngine needs the openai package: pip install openai"
            ) from exc
        if not self.available():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use ExtractiveEngine for offline runs."
            )
        return OpenAI()

    def answer(self, question: Question):
        with timer() as t:
            hits = self.index.search(question.text, top_k=self.top_k)
            citations = [
                Citation(url=c.url, snippet=c.text, title=c.title)
                for c, _ in hits
            ]

            if not citations:
                return build_record(
                    question=question,
                    engine_name=self.name,
                    answer_text="No relevant source found.",
                    citations=[],
                    latency_s=t.elapsed,
                )

            sources = "\n\n".join(
                f"[{i}] {c.title or c.url}\n{c.snippet[:self.max_snippet_chars]}"
                for i, c in enumerate(citations, 1)
            )
            try:
                response = self._client().chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"QUESTION: {question.text}\n\nSOURCES:\n{sources}",
                        },
                    ],
                )
                answer_text = (response.choices[0].message.content or "").strip()
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                answer_text, error = "", f"{type(exc).__name__}: {exc}"

        return build_record(
            question=question,
            engine_name=self.name,
            answer_text=answer_text,
            citations=citations,
            latency_s=t.elapsed,
            error=error,
        )
