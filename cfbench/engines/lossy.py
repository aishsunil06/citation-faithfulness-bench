"""An engine that paraphrases lossily, producing genuinely borderline claims.

Why this exists: `ExtractiveEngine` quotes its source verbatim, so every claim
it makes is trivially supported. A human labelling its output presses the same
key every time, the label set has no variance, and Cohen's kappa collapses to
zero no matter how good the judge is. Calibration needs disagreement to
measure, and verbatim quoting cannot produce any.

This engine rewrites the quoted sentence in ways that are *arguably* still
supported. It drops a temporal qualifier, or rounds a figure, or welds two
retrieved sentences into one claim citing only the first. A careful human will
not agree with themselves across all of these, which is the point: these are
the cases where "does the source support this?" is a real question.

Crucially this emits **no oracle labels**. Fault injection (see faulty.py)
produces unambiguous corruption with a known correct verdict. This produces
ambiguity on purpose, and only a human can resolve it.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ..claims import split_sentences
from ..retrieval import BM25Index, tokenize
from ..schema import Citation, Question
from .base import build_record, timer

# Trailing clauses that scope a claim in time or place. Removing one makes the
# claim broader than its source warrants, which is the commonest real-world
# "partial" case: true as stated, but stated too generally.
_QUALIFIER = re.compile(
    r",?\s+(?:"
    r"as of (?:the )?[^,.]*|"
    r"during (?:the )?[^,.]*|"
    r"in fiscal \d{4}|"
    r"in \d{4}|"
    r"at the close of [^,.]*|"
    r"for the [^,.]*quarter[^,.]*"
    r")(?=[.,]|$)",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"\b(\d{2,}(?:\.\d+)?)\b")


def _drop_qualifier(sentence: str, rng) -> tuple[str, str] | None:
    match = _QUALIFIER.search(sentence)
    if not match:
        return None
    out = (sentence[:match.start()] + sentence[match.end():]).strip()
    out = re.sub(r"\s{2,}", " ", out).rstrip(",")
    if not out.endswith("."):
        out += "."
    if len(tokenize(out)) < 3:
        return None
    return out, f"dropped scope qualifier {match.group().strip()!r}"


def _looks_like_year(raw: str, value: float) -> bool:
    """Years must never be rounded.

    "March 2025" becoming "March roughly 2000" is nonsense, not a borderline
    case: a human marks it unsupported instantly and the claim teaches nothing
    about where judge and human genuinely diverge.
    """
    return len(raw) == 4 and value == int(value) and 1800 <= value <= 2200


def _round_number(sentence: str, rng) -> tuple[str, str] | None:
    candidates = [
        m for m in _NUMBER.finditer(sentence)
        if not _looks_like_year(m.group(1), float(m.group(1)))
    ]
    if not candidates:
        return None
    target = rng.choice(candidates)
    raw = target.group(1)
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 10:
        return None

    # Two significant figures, so 412 -> 410 rather than 400. The goal is a
    # figure a reader might accept as "roughly right", not an obvious miss.
    digits = len(str(int(value)))
    magnitude = 10 ** max(digits - 2, 0)
    rounded = int(round(value / magnitude) * magnitude)
    if rounded == int(value):
        return None

    out = (
        sentence[:target.start(1)]
        + f"roughly {rounded}"
        + sentence[target.end(1):]
    )
    return out, f"rounded {raw} to roughly {rounded}"


# Words that only ever start a sentence, so lowercasing them mid-sentence is
# safe. Anything else is assumed to be a proper noun and left capitalised:
# "and helion Grid sells" is a tell that the text was machine-mangled, which
# would let a labeller spot transformed claims instead of judging them.
_SAFE_TO_LOWER = frozenset(
    "the it this these those its their a an he she they there".split()
)


def _weld(sentences: list[str], rng) -> tuple[str, str] | None:
    """Join two sentences into one claim; only the first will be cited."""
    if len(sentences) < 2:
        return None
    first, second = sentences[0], sentences[1]

    head = second.split(" ", 1)[0].strip(",.;:")
    if head.lower() in _SAFE_TO_LOWER:
        second = second[0].lower() + second[1:]

    out = f"{first.rstrip('.')}, and {second}"
    return out, "welded two sentences into one claim citing only the first"


_TRANSFORMS = (
    ("drop-qualifier", _drop_qualifier),
    ("round-number", _round_number),
)


@dataclass
class LossyEngine:
    """Retrieval plus lossy paraphrase. No LLM, no API key, no oracle labels."""

    index: BM25Index
    top_k: int = 3
    seed: int = 0
    weld_rate: float = 0.34
    name: str = "lossy-bm25"

    def answer(self, question: Question):
        rng = random.Random(f"{self.seed}:{question.id}")
        with timer() as t:
            hits = self.index.search(question.text, top_k=self.top_k)
            citations: list[Citation] = []
            parts: list[str] = []
            q_terms = set(tokenize(question.text))

            for position, (chunk, _score) in enumerate(hits, start=1):
                citations.append(
                    Citation(url=chunk.url, snippet=chunk.text, title=chunk.title)
                )
                ranked = self._rank_sentences(chunk.text, q_terms)
                if not ranked:
                    continue

                sentence = ranked[0]
                if rng.random() < self.weld_rate:
                    welded = _weld(ranked, rng)
                    if welded:
                        sentence = welded[0]
                else:
                    rng.shuffle(order := list(_TRANSFORMS))
                    for _tname, fn in order:
                        result = fn(sentence, rng)
                        if result:
                            sentence = result[0]
                            break

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
    def _rank_sentences(chunk_text: str, q_terms: set[str]) -> list[str]:
        sentences = split_sentences(chunk_text) or [chunk_text.strip()]
        scored = sorted(
            sentences,
            key=lambda s: -len(set(tokenize(s)) & q_terms),
        )
        return [s for s in scored if s.strip()]
