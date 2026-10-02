"""Core data model for the citation-faithfulness benchmark.

The unit of measurement is a (claim, cited-source) pair, not a whole answer.
An answer engine that says three things and cites correctly for two of them
scores 2/3, which is the resolution needed to compare engines meaningfully.

Everything is plain dataclasses serialised to JSONL so the dataset stays
diffable in git and readable without the library.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator


class FailureMode(str, Enum):
    """Why a question is hard, and therefore where citations tend to break.

    The question set is stratified over these so a headline faithfulness score
    can be decomposed instead of just being an average over whatever was easy
    to collect.
    """

    SIMPLE = "simple"            # single fact, single source; the control group
    MULTI_HOP = "multi_hop"      # answer requires joining two+ sources
    NUMERIC = "numeric"          # a number/statistic that must match the source
    NEGATION = "negation"        # correct answer asserts something is NOT the case
    RECENCY = "recency"          # answer depends on which snapshot of the world
    CONTESTED = "contested"      # sources legitimately disagree


class Verdict(str, Enum):
    """Does the cited source actually support the claim?"""

    SUPPORTED = "supported"          # source states the claim
    PARTIAL = "partial"              # source is related but weaker, narrower, or off in detail
    UNSUPPORTED = "unsupported"      # source simply does not support the claim
    CONTRADICTED = "contradicted"    # source states the opposite
    UNCITED = "uncited"              # claim carried no citation at all

    @property
    def is_credit(self) -> bool:
        """Whether this verdict earns credit under the strict scoring rule."""
        return self is Verdict.SUPPORTED

    @property
    def is_failure(self) -> bool:
        """Whether the citation is outright wrong, as opposed to merely weak.

        CONTRADICTED is kept separate from UNSUPPORTED because they are
        different product failures. An irrelevant source is careless
        attribution; a source that states the opposite means the engine read
        the evidence and asserted against it, which is both more damaging and
        more diagnostic.
        """
        return self in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass(frozen=True)
class Question:
    text: str
    failure_modes: tuple[FailureMode, ...]
    domain: str = "general"
    id: str = field(default_factory=lambda: _new_id("q"))
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"question {self.id} has empty text")
        if not self.failure_modes:
            raise ValueError(f"question {self.id} must declare >=1 failure mode")


@dataclass
class Citation:
    """A source the engine offered as evidence.

    `snippet` is the text actually shown/retrieved. Judging happens against the
    snippet, not the live URL, so a run stays reproducible after the web moves.
    """

    url: str
    snippet: str
    title: str = ""
    id: str = field(default_factory=lambda: _new_id("c"))


@dataclass
class Claim:
    """One atomic assertion lifted out of an answer, with the sources it cited."""

    question_id: str
    text: str
    citation_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _new_id("cl"))

    @property
    def is_uncited(self) -> bool:
        return not self.citation_ids


@dataclass
class AnswerRecord:
    """One engine's response to one question, decomposed into claims."""

    question_id: str
    engine: str
    answer_text: str
    claims: list[Claim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    latency_s: float | None = None
    error: str | None = None

    def citation_by_id(self, cid: str) -> Citation | None:
        return next((c for c in self.citations if c.id == cid), None)

    def snippets_for(self, claim: Claim) -> list[str]:
        out = []
        for cid in claim.citation_ids:
            cit = self.citation_by_id(cid)
            if cit is not None:
                out.append(cit.snippet)
        return out


@dataclass
class Label:
    """A verdict on one claim, from a human or a judge.

    `labeler` is free text so human and judge labels live in the same store and
    can be compared directly: "human:aishwarya", "judge:lexical",
    "judge:gpt-4o-mini".
    """

    claim_id: str
    verdict: Verdict
    labeler: str
    rationale: str = ""

    @property
    def is_human(self) -> bool:
        return self.labeler.startswith("human")


# --------------------------------------------------------------------------
# Stable, content-addressed identifiers
# --------------------------------------------------------------------------
#
# Human labels are this project's only irreplaceable asset, and they are stored
# by claim id. Random per-run ids meant that re-running an engine invalidated
# every label collected against it: on the first real labelling session, 17
# labels were orphaned by a single re-run.
#
# Ids are therefore derived from content. The same engine, over the same
# question, producing the same sentence with the same sources, yields the same
# id on every run, so labels accumulate instead of evaporating. A changed claim
# gets a new id on purpose, because the old label no longer describes it.


def content_citation_id(url: str, snippet: str) -> str:
    digest = hashlib.sha1(f"{url}|{snippet.strip()}".encode("utf-8")).hexdigest()
    return f"c_{digest[:12]}"


def content_claim_id(engine: str, question_id: str, index: int, text: str) -> str:
    payload = f"{engine}|{question_id}|{index}|{text.strip()}"
    return f"cl_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def stabilize_ids(record: "AnswerRecord") -> "AnswerRecord":
    """Rewrite a record's citation and claim ids to content-addressed ones.

    Must be called *after* any mutation of claim text, citations, or the engine
    name, since all three feed the hash. Fault injectors therefore call it at
    the end of injection rather than relying on build_record.
    """
    remap: dict[str, str] = {}
    for citation in record.citations:
        new_id = content_citation_id(citation.url, citation.snippet)
        remap[citation.id] = new_id
        citation.id = new_id

    for index, claim in enumerate(record.claims):
        claim.citation_ids = [remap.get(cid, cid) for cid in claim.citation_ids]
        claim.id = content_claim_id(
            record.engine, record.question_id, index, claim.text
        )
    return record


# --------------------------------------------------------------------------
# JSONL persistence
# --------------------------------------------------------------------------

_ENUMS: dict[str, type[Enum]] = {"verdict": Verdict}


def _encode(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = row if isinstance(row, dict) else asdict(row)
            fh.write(json.dumps(payload, default=_encode, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc


def question_from_dict(d: dict) -> Question:
    return Question(
        id=d.get("id") or _new_id("q"),
        text=d["text"],
        failure_modes=tuple(FailureMode(m) for m in d["failure_modes"]),
        domain=d.get("domain", "general"),
        notes=d.get("notes", ""),
    )


def label_from_dict(d: dict) -> Label:
    return Label(
        claim_id=d["claim_id"],
        verdict=Verdict(d["verdict"]),
        labeler=d["labeler"],
        rationale=d.get("rationale", ""),
    )


def answer_from_dict(d: dict) -> AnswerRecord:
    return AnswerRecord(
        question_id=d["question_id"],
        engine=d["engine"],
        answer_text=d["answer_text"],
        claims=[
            Claim(
                id=c["id"],
                question_id=c["question_id"],
                text=c["text"],
                citation_ids=list(c.get("citation_ids", [])),
            )
            for c in d.get("claims", [])
        ],
        citations=[
            Citation(
                id=c["id"],
                url=c["url"],
                snippet=c["snippet"],
                title=c.get("title", ""),
            )
            for c in d.get("citations", [])
        ],
        latency_s=d.get("latency_s"),
        error=d.get("error"),
    )
