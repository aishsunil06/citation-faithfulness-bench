"""Dependency-free BM25 retrieval over a local document corpus.

The benchmark needs its own retrieval so that "which chunking strategy hurts
faithfulness" is a question it can answer, rather than a property hidden inside
a third-party engine. BM25 is implemented here rather than pulled in as a
dependency so the whole pipeline runs offline with nothing installed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")

# Dropped before scoring: they carry no discriminative signal and inflate
# overlap between unrelated texts.
_STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have he her his how i in is
it its of on or that the their there they this to was were what when where
which who will with would you your
""".split())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


@dataclass
class Document:
    """A source document, pre-chunk."""

    url: str
    text: str
    title: str = ""


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    url: str
    text: str
    title: str = ""
    doc_index: int = 0
    chunk_index: int = 0


def chunk_document(
    doc: Document,
    doc_index: int,
    words_per_chunk: int = 90,
    overlap_words: int = 20,
) -> list[Chunk]:
    """Split a document into overlapping word windows.

    `words_per_chunk` and `overlap_words` are the knobs the benchmark sweeps:
    chunks too small truncate the evidence a claim needs, chunks too large
    dilute retrieval and let an engine cite a passage that only topically
    resembles its claim.
    """
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive")
    if overlap_words < 0 or overlap_words >= words_per_chunk:
        raise ValueError("overlap_words must be in [0, words_per_chunk)")

    words = doc.text.split()
    if not words:
        return []

    stride = words_per_chunk - overlap_words
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(words), stride)):
        window = words[start:start + words_per_chunk]
        if not window:
            break
        chunks.append(
            Chunk(
                url=doc.url,
                text=" ".join(window),
                title=doc.title,
                doc_index=doc_index,
                chunk_index=i,
            )
        )
        if start + words_per_chunk >= len(words):
            break
    return chunks


@dataclass
class BM25Index:
    """Okapi BM25 over a fixed chunk set."""

    chunks: list[Chunk]
    k1: float = 1.5
    b: float = 0.75
    _tokens: list[list[str]] = field(default_factory=list, repr=False)
    _tf: list[Counter] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avg_len: float = 0.0

    def __post_init__(self) -> None:
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self._tf = [Counter(toks) for toks in self._tokens]
        self._df = Counter()
        for toks in self._tokens:
            for term in set(toks):
                self._df[term] += 1
        lengths = [len(t) for t in self._tokens]
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        # BM25+ style flooring keeps very common terms from going negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            tf = self._tf[i]
            length = len(self._tokens[i]) or 1
            total = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1)
                )
                total += self._idf(term) * (freq * (self.k1 + 1)) / denom
            if total > 0:
                scored.append((chunk, total))

        scored.sort(key=lambda pair: (-pair[1], pair[0].doc_index, pair[0].chunk_index))
        return scored[:top_k]


def build_index(
    docs: Sequence[Document],
    words_per_chunk: int = 90,
    overlap_words: int = 20,
    **bm25_kwargs,
) -> BM25Index:
    chunks: list[Chunk] = []
    for i, doc in enumerate(docs):
        chunks.extend(chunk_document(doc, i, words_per_chunk, overlap_words))
    return BM25Index(chunks=chunks, **bm25_kwargs)


def load_corpus(rows: Iterable[dict]) -> list[Document]:
    return [
        Document(url=r["url"], text=r["text"], title=r.get("title", ""))
        for r in rows
    ]
