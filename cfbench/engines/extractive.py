"""Retrieval-only engine: answers by quoting what it retrieved.

This engine cannot hallucinate, because every sentence it emits is copied from
the chunk it cites. Its faithfulness score is therefore close to the ceiling
BM25 retrieval allows, which makes it the reference line for every generative
engine: an LLM engine scoring below it is losing faithfulness in generation,
not in retrieval.

Runs fully offline with no API key, so the pipeline is end-to-end testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..claims import split_sentences
from ..retrieval import BM25Index, tokenize
from ..schema import Citation, Question
from .base import build_record, timer


@dataclass
class ExtractiveEngine:
    index: BM25Index
    top_k: int = 3
    name: str = "extractive-bm25"

    def answer(self, question: Question):
        with timer() as t:
            hits = self.index.search(question.text, top_k=self.top_k)

            citations: list[Citation] = []
            parts: list[str] = []
            q_terms = set(tokenize(question.text))

            for position, (chunk, _score) in enumerate(hits, start=1):
                citations.append(
                    Citation(url=chunk.url, snippet=chunk.text, title=chunk.title)
                )
                sentence = self._best_sentence(chunk.text, q_terms)
                if sentence:
                    parts.append(f"{sentence.rstrip('.')} [{position}].")

            answer_text = " ".join(parts) if parts else "No relevant source found."

        return build_record(
            question=question,
            engine_name=self.name,
            answer_text=answer_text,
            citations=citations,
            latency_s=t.elapsed,
        )

    @staticmethod
    def _best_sentence(chunk_text: str, q_terms: set[str]) -> str:
        """Pick the chunk sentence sharing the most content words with the query."""
        sentences = split_sentences(chunk_text)
        if not sentences:
            return chunk_text.strip()
        best, best_score = "", -1.0
        for sentence in sentences:
            tokens = set(tokenize(sentence))
            if not tokens:
                continue
            score = len(tokens & q_terms)
            if score > best_score:
                best, best_score = sentence, score
        return best or sentences[0]
