"""Turn operator-captured web pages into a corpus the benchmark can trust.

The benchmark measures whether an engine's citations are supported by the text
it cited. That measurement is only as honest as the corpus underneath it, and
raw captured pages break it in two specific ways:

1. Navigation chrome and footer apparatus. A chunk of menu links or a "See
   also" list is high-entropy junk that BM25 will happily return as the best
   match for some query. The engine then cites it, the judge finds no support,
   and the engine is penalised for the corpus's defect rather than its own.
2. Near-duplicate pages. Related articles restate each other. If two documents
   carry the same sentence, an engine can cite either one and look correct
   against whichever it picked, which inflates retrieval precision for free.

So cleaning and deduping happen once, at ingestion, and the cleaned corpus is
committed as data. Capture itself stays outside the library: the operator
pastes pages into a raw JSONL file, which keeps the core dependency-free and
offline, and keeps a published run reproducible after the live web moves.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .retrieval import Document, tokenize

# Inline citation and editorial markers. Captured prose is littered with these;
# left in place they leak into judged snippets and into claim text, where a
# stray "[3]" is noise in every lexical comparison downstream.
_MARKERS = re.compile(r"\[\s*(?:\d+|citation needed|clarification needed|sic)\s*\]", re.I)

# Lines whose *prefix* matches one of these are dropped whole. Prefix rather
# than substring matching is deliberate: "Contents" as a heading is chrome,
# but "The contents of the reservoir fell" is prose worth keeping, and a
# substring rule would delete the sentence.
_BOILERPLATE = (
    "jump to content",
    "main menu",
    "main page",
    "navigation",
    "see also",
    "references",
    "external links",
    "further reading",
    "retrieved from",
    "categories:",
    "category:",
    "edit",
    "contents",
    "current events",
    "random article",
    "from wikipedia",
    "this article",
)

# A captured page's chrome is overwhelmingly short lines: menu entries, section
# headings, single-word links. 25 characters sits above essentially all of them
# and below any real sentence; the shortest prose sentence worth keeping in a
# factual article ("Construction began in 1931.") clears it. The cost of the
# floor is the occasional terse caption, which is an acceptable loss because a
# caption is rarely the only place a fact appears.
_MIN_LINE_CHARS = 25

# A document under 60 words cannot fill even the narrowest chunk width the
# benchmark sweeps (90 words), so it would only ever contribute a stub chunk.
# Rejecting it here is better than letting it distort the chunk-width
# comparison, which is one of the benchmark's actual measurements.
_MIN_DOC_WORDS = 60


def clean_text(raw: str) -> str:
    """Strip boilerplate and reference markers from a captured page.

    Returns prose as a single whitespace-normalised line. Newlines are not
    preserved: captured text wraps at whatever width the operator's terminal
    used, so the line breaks inside a paragraph carry no information, and
    removing them keeps a judged snippet readable.
    """
    without_markers = _MARKERS.sub("", raw)

    kept: list[str] = []
    for line in without_markers.splitlines():
        line = line.strip()
        if len(line) < _MIN_LINE_CHARS:
            continue
        lowered = line.lower()
        if any(lowered.startswith(prefix) for prefix in _BOILERPLATE):
            continue
        kept.append(line)

    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def to_document(url: str, title: str, raw: str) -> Document:
    """Clean a captured page into a Document, or reject it as too thin.

    Raises ValueError rather than returning None so a bad capture is loud at
    ingest time; `ingest_rows` is the layer that chooses to tolerate it.
    """
    text = clean_text(raw)
    n_words = len(text.split())
    if n_words < _MIN_DOC_WORDS:
        raise ValueError(
            f"{url}: only {n_words} words after cleaning, need {_MIN_DOC_WORDS}"
        )
    return Document(url=url, text=text, title=title)


def _token_set(doc: Document) -> frozenset[str]:
    # retrieval.tokenize is reused rather than reimplemented so that two
    # documents judged distinct here are also distinct to the index that will
    # retrieve them. Stopwords are dropped, which is what makes the similarity
    # reflect shared content rather than shared English.
    return frozenset(tokenize(doc.text))


def dedupe(
    docs: Sequence[Document], threshold: float = 0.8
) -> tuple[list[Document], list[str]]:
    """Drop documents that duplicate an earlier one.

    Returns the kept documents in input order and the urls dropped.

    Similarity is Jaccard over token sets, which ignores word order and
    length. That is the right blunt instrument here: the failure being guarded
    against is two pages covering the same facts, not two pages sharing a
    phrase. 0.8 is deliberately high -- two articles in the same domain share a
    lot of vocabulary, and dropping a genuinely distinct document costs the
    question set a source, which is worse than keeping a partial overlap.

    Each candidate is compared only against documents already *kept*, not
    against dropped ones, so a chain of partial overlaps cannot cascade into
    deleting a page that no surviving document actually duplicates.
    """
    kept: list[Document] = []
    kept_tokens: list[frozenset[str]] = []
    dropped: list[str] = []

    for doc in docs:
        tokens = _token_set(doc)
        duplicate = False
        for other in kept_tokens:
            union = tokens | other
            if not union:
                # Both sides tokenise to nothing. Scoring that as identical
                # would delete unrelated pages on no evidence, so treat it as
                # dissimilar and let the word floor catch genuine emptiness.
                continue
            if len(tokens & other) / len(union) >= threshold:
                duplicate = True
                break
        if duplicate:
            dropped.append(doc.url)
        else:
            kept.append(doc)
            kept_tokens.append(tokens)

    return kept, dropped


def ingest_rows(rows: Iterable[dict]) -> tuple[list[Document], list[str]]:
    """Clean, validate and dedupe `{url, title, raw}` rows into a corpus.

    Returns the kept documents and the urls dropped for any reason -- too thin
    after cleaning, or duplicating a kept document. The two rejection reasons
    are merged into one list because the caller's question is "which captures
    do I need to replace", and both answers are the same.
    """
    cleaned: list[Document] = []
    dropped: list[str] = []

    for row in rows:
        url = row["url"]
        try:
            cleaned.append(to_document(url, row.get("title", ""), row.get("raw", "")))
        except ValueError:
            dropped.append(url)

    kept, duplicates = dedupe(cleaned)
    return kept, dropped + duplicates
