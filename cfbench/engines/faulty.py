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

from ..claims import split_sentences
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


# --------------------------------------------------------------------------
# Shared text inspection used by the entity and polarity injectors
# --------------------------------------------------------------------------
#
# Both of those injectors claim CONTRADICTED, which is only true when the
# cited source makes the *competing* statement. The helpers below exist to
# establish that, and every one of them is built to return a negative answer
# when it cannot be sure: a declined injection costs a sample, a wrong oracle
# label costs the whole result.

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "its", "their",
    "was", "were", "are", "has", "have", "had", "been", "but", "which",
    "also", "per", "than", "into", "over", "under", "about", "said", "says",
    "during", "between", "after", "before", "they", "them", "there", "then",
    "such", "some", "any", "all", "one", "both", "each", "other", "while",
}

# A named entity is taken to be two or more consecutive capitalised words.
# Single capitalised words are deliberately out of scope: "Revenue rose" and
# "Northwind rose" are indistinguishable at sentence start, and swapping a
# common noun that merely happens to start a sentence produces gibberish whose
# correct verdict nobody could defend.
_ENTITY = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\b")

# Words that are capitalised because a sentence started, not because they name
# anything. Stripped from the front of a candidate span.
_NOT_A_NAME = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "their",
    "his", "her", "our", "in", "on", "at", "as", "by", "for", "from", "and",
    "but", "however", "although", "if", "when", "while", "after", "before",
    "both", "each", "every", "no", "not", "during", "under", "over", "with",
    "without", "since", "per", "according", "many", "most", "several", "some",
    "there", "they", "we", "he", "she", "revenue", "capacity",
}


def _content_words(text: str) -> set[str]:
    """Lower-cased words that carry the sentence's subject matter."""
    return {
        w for w in re.findall(r"[a-z]+", text.lower())
        if len(w) > 2 and w not in _STOP
    }


def _entities(text: str) -> list[str]:
    """Multi-word capitalised spans, in order of appearance, deduplicated."""
    out: list[str] = []
    for match in _ENTITY.finditer(text):
        words = match.group(0).split()
        if words and words[0].lower() in _NOT_A_NAME:
            words = words[1:]
        if len(words) < 2:
            continue
        span = " ".join(words)
        if span not in out:
            out.append(span)
    return out


def _all_numbers(text: str) -> list[str]:
    return [m.group(1) for m in _NUMBER.finditer(text)]


def _magnitudes(text: str) -> list[str]:
    """Numbers that are point values: not years, not bounds.

    The same exclusions `NumericDrift` relies on. A bound or a year cannot
    anchor a contradiction, because a differing bound is an entailment and a
    differing year is a silence.
    """
    return [
        m.group(1) for m in _NUMBER.finditer(text)
        if not _is_directional(text, m.start(1))
        and not _looks_like_year(m.group(1), float(m.group(1)))
    ]


def _overlaps(predicate: set[str], text: str) -> bool:
    """Whether `text` restates most of `predicate`, i.e. the same fact."""
    if len(predicate) < 2:
        return False
    hit = predicate & _content_words(text)
    return len(hit) >= 2 and len(hit) / len(predicate) >= 0.6


def _competing_sentence(
    snippet: str, claim_text: str, entity_a: str, entity_b: str
) -> str | None:
    """Find the sentence proving the source disagrees about `entity_b`.

    Returns a sentence from `snippet` that makes the claim's own statement
    about `entity_b` but with a *different* figure, or None.

    Both halves matter. The source must state the claim's figure about
    `entity_a`, so the uncorrupted claim is genuinely supported and the swap is
    the only fault present. And it must state a different figure about
    `entity_b`, so the corrupted claim is denied by the source rather than
    merely absent from it. Without the second half the honest verdict is
    UNSUPPORTED, since "Helion Grid earned 412 million" is perfectly
    compatible with a source that only discusses Northwind.
    """
    figures = _magnitudes(claim_text)
    if not figures:
        return None

    predicate = (
        _content_words(claim_text)
        - _content_words(entity_a)
        - _content_words(entity_b)
    )
    if len(predicate) < 2:
        return None

    sentences = split_sentences(snippet)

    supports_a = any(
        entity_a in s and any(f in _all_numbers(s) for f in figures)
        for s in sentences
    )
    if not supports_a:
        return None

    for sentence in sentences:
        if entity_b not in sentence:
            continue
        numbers = _all_numbers(sentence)
        # The source agrees with the figure for B: the swap would produce a
        # SUPPORTED claim, the opposite of the intended fault.
        if any(f in numbers for f in figures):
            continue
        rivals = [n for n in _magnitudes(sentence) if n not in figures]
        if not rivals:
            continue
        if not _overlaps(predicate, sentence):
            continue
        return sentence
    return None


def _rewrite(
    record: AnswerRecord,
    claim,
    new_text: str,
    old_token: str = "",
    new_token: str = "",
) -> None:
    """Apply corrupted text to the claim and keep `answer_text` in step.

    The claim's text is marker-stripped, so its body is still a substring of
    the answer prose and can be swapped wholesale. The token fallback covers
    claims whose markers sit mid-sentence.
    """
    core = claim.text.strip().rstrip(".").strip()
    replacement = new_text.strip().rstrip(".").strip()
    if core and core in record.answer_text:
        record.answer_text = record.answer_text.replace(core, replacement, 1)
    elif old_token and record.answer_text.count(old_token) == 1:
        # Only when the token is unambiguous: a blind single replacement could
        # otherwise rewrite a different sentence of the same answer.
        record.answer_text = record.answer_text.replace(old_token, new_token, 1)
    claim.text = new_text


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


class EntitySwap(_Base):
    """Re-attribute the claim's fact to a different named entity.

    The real failure this mimics: an engine reads two companies out of the
    same retrieved passage and attaches the first one's figure to the second.

    Correct verdict is CONTRADICTED, but only under a condition the injector
    enforces before it fires. The candidate entity is drawn from a *non-cited*
    citation on the same record, which is what makes the fault realistic, and
    it is then required to appear in the cited snippet with a *different*
    figure for the same fact. Both halves are necessary:

      source: "Northwind earned 412 million. Helion Grid earned 98 million."
      claim:  "Helion Grid earned 412 million."          -> CONTRADICTED

      source: "Northwind earned 412 million."
      claim:  "Helion Grid earned 412 million."           -> UNSUPPORTED

    The second case is the trap. Nothing in that source denies Helion Grid
    also earned 412 million; it simply does not discuss the company. Labelling
    it CONTRADICTED would hand the benchmark exactly the kind of indefensible
    oracle that the numeric and bounded-quantity guards were written to stop,
    so the injector declines instead. On a corpus where no cited snippet
    contradicts an available alternative, this injector produces nothing, and
    that is the intended behaviour rather than a bug to be loosened.
    """

    fault = "entity-swap"

    def _corrupt(self, record: AnswerRecord, claim):
        if not claim.citation_ids:
            return None
        cited = [c for c in record.citations if c.id in claim.citation_ids]
        others = [c for c in record.citations if c.id not in claim.citation_ids]
        if not cited or not others:
            return None

        alternatives: list[str] = []
        for citation in others:
            for name in _entities(citation.snippet):
                if name not in alternatives:
                    alternatives.append(name)
        if not alternatives:
            return None

        options: list[tuple[str, str]] = []
        for original in _entities(claim.text):
            # Two mentions of the same name would leave the claim half-swapped
            # and incoherent, which has no defensible verdict at all.
            if claim.text.count(original) != 1:
                continue
            for other in alternatives:
                if other == original or other in claim.text:
                    continue
                for citation in cited:
                    if original not in citation.snippet:
                        continue
                    if _competing_sentence(
                        citation.snippet, claim.text, original, other
                    ) is not None:
                        options.append((original, other))
                        break
        if not options:
            return None

        original, other = self._rng.choice(options)
        _rewrite(
            record,
            claim,
            claim.text.replace(original, other, 1),
            old_token=original,
            new_token=other,
        )
        return (
            Verdict.CONTRADICTED,
            f"injected entity swap: {original} -> {other}; the cited source "
            f"states this fact of {original} and a different figure for {other}",
            claim,
        )


# Negation markers that can be toggled one-for-one.
_NEGATION = re.compile(r"\b(no|not|never|without)\b", re.IGNORECASE)

# Hedges. A hedged sentence has no clean opposite: the negation of "may not
# have occurred" is not "occurred", so flipping it yields a claim whose verdict
# is a matter of opinion.
_MODALS = (
    "may", "might", "could", "appears", "appear", "suggests", "suggest",
    "seems", "seem", "possibly", "perhaps", "likely", "unlikely",
    "reportedly", "allegedly", "apparently", "presumably",
)

# Contexts where the marker is not a plain negation:
#   - "no longer operational"   deleting the marker yields word salad
#   - "no more than 240"        the marker sets a BOUND, and a flipped bound
#                               is an entailment rather than a contradiction,
#                               the exact error that made 3.1 billion "by
#                               3248" look like a fault
#   - "not only late"           a correlative, not a denial
#   - "No. 3"                   an abbreviation for a number
#   - "denied that no ..."      a second negative the marker interacts with,
#                               so flipping one of them does not reliably
#                               reverse what the sentence asserts
_NEGATION_IDIOM = re.compile(
    r"\bno(?:t)?\s+(?:longer|more|less|fewer|later|earlier|greater|sooner"
    r"|only|just|doubt|matter|least|other|until|before)\b"
    r"|\bnone\b|\bwhether\b|\bno\.\s*\d"
    r"|\b(?:deny|denies|denied|refuse[ds]?|reject(?:s|ed)?|fail(?:s|ed)?"
    r"|lack(?:s|ed|ing)?|absence|neither|nor|unable|unless|except"
    r"|hardly|scarcely|barely)\b",
    re.IGNORECASE,
)


def _hedged(text: str) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{word}\b", low) for word in _MODALS)


class PolarityFlip(_Base):
    """Invert a single unhedged negation, reversing what the claim asserts.

    "The audit found no incidents" becomes "The audit found incidents", which
    the source denies outright: CONTRADICTED.

    The injector only fires on a claim the source genuinely negates. If the
    cited snippet never states the negative, the *original* claim was already
    unsupported and the flipped one is equally unsupported, so there is no
    contradiction to label. Hedges, bound-setting idioms, and claims carrying
    more than one marker are all declined, because inverting them produces an
    arguable verdict rather than a certain one.
    """

    fault = "polarity-flip"

    def _corrupt(self, record: AnswerRecord, claim):
        text = claim.text
        if not claim.citation_ids:
            return None
        if _hedged(text) or _NEGATION_IDIOM.search(text):
            return None

        markers = list(_NEGATION.finditer(text))
        if len(markers) != 1:
            # Zero markers: nothing to flip. Two or more: flipping one leaves
            # a half-negated sentence whose meaning, and therefore verdict,
            # is not determinate.
            return None
        marker = markers[0]
        word = marker.group(1).lower()

        snippets = [
            c.snippet for c in record.citations if c.id in claim.citation_ids
        ]
        if not any(self._source_negates(s, text, word) for s in snippets):
            return None

        flipped = self._flip(text, marker)
        if flipped is None:
            return None

        _rewrite(
            record,
            claim,
            flipped,
            old_token=f"{marker.group(1)} ",
            new_token="with " if word == "without" else "",
        )
        return (
            Verdict.CONTRADICTED,
            f"injected polarity flip: negation {marker.group(1)!r} inverted; "
            "the cited source states the negative",
            claim,
        )

    @staticmethod
    def _source_negates(snippet: str, claim_text: str, word: str) -> bool:
        """Whether the source itself negates this claim's fact.

        Scoped to a single sentence of the snippet, not the snippet as a
        whole. The sentence carrying the negation is what has to be plain:
        a hedge four sentences away says nothing about this fact, while a
        snippet-wide hedge test silently declines every long chunk.
        """
        predicate = _content_words(claim_text)
        for sentence in split_sentences(snippet):
            if not re.search(rf"\b{word}\b", sentence, re.IGNORECASE):
                continue
            if _hedged(sentence) or _NEGATION_IDIOM.search(sentence):
                continue
            if _overlaps(predicate, sentence):
                return True
        return False

    @staticmethod
    def _flip(text: str, marker: re.Match) -> str | None:
        word = marker.group(1)
        if word.lower() == "without":
            swap = "With" if word[0].isupper() else "with"
            out = text[:marker.start(1)] + swap + text[marker.end(1):]
        else:
            out = text[:marker.start(1)] + text[marker.end(1):]

        out = re.sub(r"\s{2,}", " ", out).strip()
        out = re.sub(r"\s+([.,;:!?])", r"\1", out)
        if out[:1].islower():
            out = out[0].upper() + out[1:]
        if len(out.split()) < 3 or out == text:
            return None
        return out


FAULTS: dict[str, type[_Base]] = {
    NumericDrift.fault: NumericDrift,
    WrongSource.fault: WrongSource,
    DroppedCitation.fault: DroppedCitation,
    UnsupportedPadding.fault: UnsupportedPadding,
    EntitySwap.fault: EntitySwap,
    PolarityFlip.fault: PolarityFlip,
}


def wrap(engine, fault: str, rate: float = 1.0, seed: int = 0) -> _Base:
    if fault not in FAULTS:
        raise ValueError(
            f"unknown fault {fault!r}; options: {sorted(FAULTS)}"
        )
    return FAULTS[fault](engine, rate=rate, seed=seed)
