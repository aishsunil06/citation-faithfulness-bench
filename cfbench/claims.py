"""Decompose an engine's answer into atomic claims with their citations.

Answer engines emit prose with inline markers, e.g.

    "Revenue grew 12% in 2024 [1]. The company also entered Brazil [2][3]."

Scoring needs one row per assertion, so this module splits the prose into
sentences and attaches whichever citations that sentence pointed at.

Sentence splitting is deliberately conservative: the failure that matters is
splitting "3.5 million" or "U.S. Treasury" into two claims, which would create
a fake unsupported claim out of nothing and corrupt the score. Over-merging two
real sentences is the safer error, so ambiguous cases stay merged.
"""

from __future__ import annotations

import re

from .schema import Citation, Claim

# Inline citation markers: [1], [12], [1, 2], [1][2]
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Abbreviations that end in a period but do not end a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "al", "inc", "ltd", "co", "corp", "dept", "est", "fig", "no", "vol", "approx",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m",
}

_SENT_END = re.compile(r"([.!?])(\s+)")


def _looks_like_abbrev(chunk: str) -> bool:
    tail = chunk.rstrip()
    if not tail.endswith("."):
        return False
    last = tail[:-1].split()[-1].lower() if tail[:-1].split() else ""
    last = last.strip("([\"'")
    if last in _ABBREV:
        return True
    # Single initial, e.g. "J." in "J. Smith"
    if len(last) == 1 and last.isalpha():
        return True
    # Dotted acronym, e.g. "U.S." -> "u.s"
    if re.fullmatch(r"(?:[a-z]\.)+[a-z]", last):
        return True
    return False


def _ends_mid_number(text: str, end_idx: int) -> bool:
    """True if the period at end_idx-1 sits between two digits (a decimal)."""
    if end_idx - 2 < 0 or end_idx >= len(text):
        return False
    return text[end_idx - 2].isdigit() and text[end_idx].isdigit()


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, guarding decimals and abbreviations."""
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    start = 0
    for match in _SENT_END.finditer(text):
        end = match.end(1)  # index just past the punctuation
        if _ends_mid_number(text, end):
            continue
        candidate = text[start:end]
        if _looks_like_abbrev(candidate):
            continue
        # Require the next non-space char to look like a sentence start.
        nxt = text[match.end():match.end() + 1]
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"'([“‘"):
            continue
        out.append(candidate.strip())
        start = match.end()

    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _marker_numbers(sentence: str) -> list[int]:
    nums: list[int] = []
    for match in _MARKER.finditer(sentence):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                nums.append(int(part))
    # Preserve order, drop duplicates
    seen: set[int] = set()
    return [n for n in nums if not (n in seen or seen.add(n))]


def strip_markers(sentence: str) -> str:
    """Remove inline citation markers and tidy the leftover spacing."""
    cleaned = _MARKER.sub("", sentence)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def decompose_answer(
    question_id: str,
    answer_text: str,
    citations: list[Citation],
) -> list[Claim]:
    """Turn an answer into one Claim per sentence.

    Markers are 1-indexed against `citations` order, matching how answer
    engines render them. Out-of-range markers are dropped rather than raising,
    because a hallucinated citation index is itself a real engine failure and
    should surface as an uncited claim, not crash the run.
    """
    claims: list[Claim] = []
    for sentence in split_sentences(answer_text):
        numbers = _marker_numbers(sentence)
        text = strip_markers(sentence)
        if not text:
            continue
        citation_ids = [
            citations[n - 1].id
            for n in numbers
            if 1 <= n <= len(citations)
        ]
        claims.append(
            Claim(question_id=question_id, text=text, citation_ids=citation_ids)
        )
    return claims
