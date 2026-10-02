"""Fault-injection wrappers: engines whose citation errors are known by design.

The problem this solves: validating a judge normally requires human labels,
which are slow and expensive. But if a *known* corruption is applied to an
otherwise-faithful answer, the correct verdict for the corrupted claim is known
without any human reading it.

That turns judge validation into a measurable quantity on day one: what
fraction of injected faults does each judge catch? It does not replace human
labelling, because fault injection only produces unambiguous cases and real
citation failures are often borderline. It does establish a floor: a judge that
misses synthetic numeric corruption will certainly miss subtler real ones.

Each wrapper emits `Label`s under a `oracle:<fault>` labeler alongside the
corrupted record, so the existing agreement machinery compares judge against
oracle with no special-casing.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..schema import AnswerRecord, Citation, Label, Question, Verdict, stabilize_ids

_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?)\b")


@dataclass
class InjectionResult:
    record: AnswerRecord
    oracle: list[Label] = field(default_factory=list)


class _Base:
    """Shared plumbing for fault injectors."""

    fault: str = "none"

    def __init__(self, engine, rate: float = 1.0, seed: int = 0):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self.engine = engine
        self.rate = rate
        self._rng = random.Random(seed)
        self.name = f"{engine.name}+{self.fault}"

    def answer(self, question: Question) -> AnswerRecord:
        return self.inject(question).record

    def inject(self, question: Question) -> InjectionResult:
        record = self.engine.answer(question)
        record.engine = self.name

        # Corrupt first, collecting verdicts against the claim *objects*. Ids
        # are deliberately not read yet: corruption changes claim text and the
        # engine name, both of which feed the content-addressed id, so any id
        # captured here would be stale by the time the record is written.
        pending: list[tuple[object, Verdict, str]] = []
        for claim in list(record.claims):
            if self._rng.random() > self.rate:
                continue
            outcome = self._corrupt(record, claim)
            if outcome is not None:
                verdict, why, target = outcome
                pending.append((target, verdict, why))

        stabilize_ids(record)

        oracle = [
            Label(
                claim_id=target.id,
                verdict=verdict,
                labeler=f"oracle:{self.fault}",
                rationale=why,
            )
            for target, verdict, why in pending
        ]
        return InjectionResult(record=record, oracle=oracle)

    def _corrupt(self, record: AnswerRecord, claim):
        """Corrupt a claim.

        Returns `(verdict, rationale, target_claim)` or None. The target is
        returned explicitly because an injector may label a claim it appended
        rather than the one it was handed.
        """
        raise NotImplementedError


class NumericDrift(_Base):
    """Perturb a number in the claim so it no longer matches its source.

    The commonest real citation failure: the sentence is about the right thing
    and points at the right document, but the figure is wrong. Correct verdict
    is CONTRADICTED, not UNSUPPORTED: the source addresses this exact fact and
    states a different number, which is a stronger failure than an irrelevant
    citation.
    """

    fault = "numeric-drift"

    def _corrupt(self, record: AnswerRecord, claim):
        matches = list(_NUMBER.finditer(claim.text))
        if not matches:
            return None
        target = self._rng.choice(matches)
        original = target.group(1)

        # Shift by a visible but plausible amount so the corruption is not
        # detectable from implausibility alone.
        try:
            value = float(original)
        except ValueError:
            return None
        shifted = value * self._rng.choice([1.35, 1.6, 0.6, 0.45])
        replacement = str(int(shifted)) if value == int(value) else f"{shifted:.1f}"

        claim.text = (
            claim.text[:target.start(1)] + replacement + claim.text[target.end(1):]
        )
        record.answer_text = record.answer_text.replace(original, replacement, 1)
        return (
            Verdict.CONTRADICTED,
            f"injected numeric drift: {original} -> {replacement}",
            claim,
        )


class WrongSource(_Base):
    """Repoint the claim at a different source that it does not come from.

    Simulates an engine that retrieved adequately but attributed carelessly.
    Correct verdict is UNSUPPORTED: the claim may well be true, but the source
    now attached to it does not support it.
    """

    fault = "wrong-source"

    def _corrupt(self, record: AnswerRecord, claim):
        if len(record.citations) < 2 or not claim.citation_ids:
            return None
        others = [c for c in record.citations if c.id not in claim.citation_ids]
        if not others:
            return None
        claim.citation_ids = [self._rng.choice(others).id]
        return (
            Verdict.UNSUPPORTED, "injected wrong-source attribution", claim,
        )


class DroppedCitation(_Base):
    """Strip the citation entirely, leaving a bare assertion.

    Correct verdict is UNCITED, which the benchmark keeps distinct from
    UNSUPPORTED: asserting something with no source is a different product
    failure from asserting it with the wrong source.
    """

    fault = "dropped-citation"

    def _corrupt(self, record: AnswerRecord, claim):
        if not claim.citation_ids:
            return None
        claim.citation_ids = []
        return (Verdict.UNCITED, "injected dropped citation", claim)


class UnsupportedPadding(_Base):
    """Append a confident sentence no source supports, citing source 1 anyway.

    Simulates the padding failure where an engine adds a generic closing claim
    to sound complete. Correct verdict is UNSUPPORTED.
    """

    fault = "padded-claim"
    PADDING = (
        "This trend is expected to continue across the remainder of the decade"
    )

    def __init__(self, engine, rate: float = 1.0, seed: int = 0):
        super().__init__(engine, rate=rate, seed=seed)
        self._done: set[str] = set()

    def _corrupt(self, record: AnswerRecord, claim):
        # One padded claim per answer, appended rather than modifying a real one.
        if record.question_id in self._done or not record.citations:
            return None
        self._done.add(record.question_id)

        from ..schema import Claim

        padded = Claim(
            question_id=record.question_id,
            text=self.PADDING + ".",
            citation_ids=[record.citations[0].id],
        )
        record.claims.append(padded)
        record.answer_text = f"{record.answer_text} {self.PADDING} [1]."
        return (
            Verdict.UNSUPPORTED, "injected unsupported padding claim", padded,
        )


FAULTS: dict[str, type[_Base]] = {
    NumericDrift.fault: NumericDrift,
    WrongSource.fault: WrongSource,
    DroppedCitation.fault: DroppedCitation,
    UnsupportedPadding.fault: UnsupportedPadding,
}


def wrap(engine, fault: str, rate: float = 1.0, seed: int = 0) -> _Base:
    if fault not in FAULTS:
        raise ValueError(
            f"unknown fault {fault!r}; options: {sorted(FAULTS)}"
        )
    return FAULTS[fault](engine, rate=rate, seed=seed)
