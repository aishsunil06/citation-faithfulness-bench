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


def _looks_like_year(raw: str, value: float) -> bool:
    """Four-digit values in a calendar range are years, not quantities."""
    return len(raw) == 4 and value == int(value) and 1800 <= value <= 2200


# Cue words that make a following number a BOUND rather than a point value.
# Shifting a bounded figure does not produce a contradiction, it produces a
# claim that is logically weaker or stronger than the source:
#
#   source: "will reach 3.1 billion by 2030"
#   claim:  "will reach 3.1 billion by 3248"
#
# The later deadline is *entailed* by the earlier one, so the source supports
# the claim. Calling that CONTRADICTED makes the oracle wrong, and an oracle
# that is wrong measures nothing: a judge "detecting" it is only agreeing with
# a mistake. Fault injection is only valid where the correct verdict is
# indisputable, so these contexts are skipped and left to human labelling.
_DIRECTIONAL = (
    "by", "before", "after", "since", "until", "till", "within", "from",
    "over", "under", "above", "below", "beyond", "exceeds", "exceeding",
    "least", "most", "minimum", "maximum", "up", "more", "less",
    "fewer", "greater", "nearly", "almost", "approximately", "around",
    "roughly", "about", "upto",
    # The word immediately before the figure in "up to 240", "more than 240",
    # "fewer than 240". Matching only the head word would miss all of these.
    "to", "than",
)


def _is_directional(text: str, start: int) -> bool:
    """True if the number at `start` is preceded by a bound-setting cue."""
    prefix = text[max(0, start - 24):start].lower()
    words = re.findall(r"[a-z]+", prefix)
    return bool(words) and words[-1] in _DIRECTIONAL


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

    Only magnitudes are touched. Years and bounded quantities are skipped,
    because shifting them yields a verdict that is arguable rather than
    certain, and an arguable oracle defeats the purpose.
    """

    fault = "numeric-drift"

    def _corrupt(self, record: AnswerRecord, claim):
        # Only magnitudes are corrupted, never years or bounds.
        #
        # A shifted magnitude is an indisputable contradiction: the source says
        # the quantity is 412 and the claim says 474 about the same thing.
        #
        # A shifted year is not. "Capacity was 240 as of 2026" cited to a 2025
        # source is UNSUPPORTED, because the source is silent on 2026 rather
        # than denying it. A shifted bound is weaker still: "by 3248" is
        # entailed by "by 2030". Fault injection is only worth anything while
        # the oracle verdict is beyond argument, so both are left alone and
        # resolved by human labelling instead.
        candidates = [
            m for m in _NUMBER.finditer(claim.text)
            if not _is_directional(claim.text, m.start(1))
            and not _looks_like_year(m.group(1), float(m.group(1)))
        ]
        if not candidates:
            return None
        target = self._rng.choice(candidates)
        original = target.group(1)

        try:
            value = float(original)
        except ValueError:
            return None

        # Corruption must be WRONG but PLAUSIBLE. An earlier version scaled
        # every figure multiplicatively, which turned the year 2030 into 3248.
        # That is detectable from implausibility alone, so it measured nothing
        # about a judge's ability to check a source and inflated detection
        # rates toward 100%. Years therefore shift by a year or two, and other
        # figures by a modest proportion.
        shifted = value * self._rng.choice([1.08, 1.15, 0.88, 0.82])
        if value == int(value) and abs(shifted - value) >= 1:
            replacement = str(int(round(shifted)))
        else:
            replacement = f"{shifted:.1f}"
        if replacement == original:
            return None

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
