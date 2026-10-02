"""Answer-completeness scoring: the second measurement axis.

WHY THIS AXIS EXISTS
--------------------
Faithfulness is measured per (claim, cited-source) pair, so it only ever asks
"is what the engine said supported by what it cited?". That question is blind
to an engine that cites every sentence immaculately and answers the wrong
question. A one-sentence answer that addresses half of a two-part question,
with a perfect citation on that sentence, scores 1.0 faithfulness. Without a
coverage axis the leaderboard would reward terse, well-sourced evasion -- and
the cheapest way to raise a faithfulness score is to say less.

Each question therefore declares `aspects`: the things a complete answer must
mention. Completeness is the fraction of those aspects present in the answer.

WHAT THIS GETS WRONG
--------------------
Coverage here is *keyword presence*, which is a crude proxy for semantic
coverage, and the README must say so. Known error modes, all of them real:

* False negatives (the dominant failure). A correct paraphrase that avoids the
  keyword scores as a miss. Aspect "Helion Grid" against "the grid operator
  named in the filing" is a miss; aspect "240" against "two hundred and forty"
  is a miss; synonyms, abbreviations ("MWh" vs "megawatt-hours") and
  inflections ("acquired" vs "acquisition") all miss. Completeness is therefore
  a *lower bound* on real coverage, and should be read as a comparison between
  engines over identical aspects, not as an absolute percentage.
* False positives. Presence is not assertion. An answer saying "the filing does
  not state the 240 MWh figure", or one that merely restates the question, or
  one that mentions the aspect inside an unrelated sentence, is credited. Short
  textual aspects also match inside longer words, because textual matching is
  substring-based: an aspect "art" is covered by "cart".
* Numeric aspects are handled more strictly precisely because the substring
  rule is badly wrong for them -- "240" must not be satisfied by "1240" -- so
  they match only as whole tokens.

A semantic judge would fix the paraphrase problem, but it would also put a
model in the loop of the one metric that is currently deterministic and
auditable. The tradeoff taken here is a cheap, reproducible, deliberately
conservative measure whose biases are documented, rather than a sharper one
whose biases are not.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import AnswerRecord, Question

# Anything that is not a letter or a digit becomes a space. This is blunter
# than `retrieval.tokenize` on purpose: here we want punctuation and
# hyphenation differences ("megawatt-hours" / "megawatt hours") to stop
# mattering, and we must *not* drop stopwords, because an aspect can
# legitimately be a phrase like "no change in the rate".
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, replace runs of non-alphanumerics with one space, trim."""
    return _NON_WORD.sub(" ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []


def _is_numeric(tokens: Sequence[str]) -> bool:
    """Whether an aspect is purely numeric once normalised.

    "240", "3.5" (-> "3 5") and "1,200" (-> "1 200") all qualify; "240 MWh"
    does not, and falls through to substring matching.
    """
    return bool(tokens) and all(t.isdigit() for t in tokens)


def _contains_token_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Whether `needle` appears in `haystack` as a contiguous token run.

    Whole-token comparison is what makes a numeric aspect safe: "240" is not
    satisfied by "1240", "240th" or "2400", all of which a substring test would
    accept, turning a numeric-accuracy failure into a pass.
    """
    span = len(needle)
    if not span or span > len(haystack):
        return False
    return any(
        list(haystack[i:i + span]) == list(needle)
        for i in range(len(haystack) - span + 1)
    )


def is_covered(aspect: str, answer_text: str) -> bool:
    """Whether one aspect is present in an answer under the matching rules."""
    aspect_tokens = _tokens(aspect)
    if not aspect_tokens:
        # An empty aspect string is a dataset bug, not a free point. Counting it
        # as uncovered surfaces the bug in `most_missed` instead of silently
        # inflating every engine's score.
        return False

    if _is_numeric(aspect_tokens):
        return _contains_token_run(_tokens(answer_text), aspect_tokens)

    normalized_answer = normalize(answer_text)
    if not normalized_answer:
        return False
    return " ".join(aspect_tokens) in normalized_answer


@dataclass(frozen=True)
class CompletenessScore:
    """Aspect coverage for one (question, answer) pair."""

    n_aspects: int
    n_covered: int
    covered: tuple[str, ...] = ()
    missed: tuple[str, ...] = ()

    @property
    def fraction(self) -> float:
        """Covered share of declared aspects; 0.0 when nothing was declared.

        A question with no aspects scores 0.0 rather than 1.0 or NaN. 1.0 would
        hand free credit to every engine for an unauthored question, and the
        zero makes an unpopulated question set look obviously wrong instead of
        quietly perfect.
        """
        if not self.n_aspects:
            return 0.0
        return self.n_covered / self.n_aspects

    def as_dict(self) -> dict:
        return {
            "n_aspects": self.n_aspects,
            "n_covered": self.n_covered,
            "fraction": round(self.fraction, 4),
            "covered": list(self.covered),
            "missed": list(self.missed),
        }


def completeness(question: Question, record: AnswerRecord) -> CompletenessScore:
    """Score one answer's coverage of its question's declared aspects.

    Scored against `record.answer_text` rather than the extracted claims: a
    claim splitter can drop a fragment, and completeness should measure what
    the engine actually said, not what the pipeline managed to parse.
    """
    covered: list[str] = []
    missed: list[str] = []
    for aspect in question.aspects:
        (covered if is_covered(aspect, record.answer_text) else missed).append(aspect)

    return CompletenessScore(
        n_aspects=len(question.aspects),
        n_covered=len(covered),
        covered=tuple(covered),
        missed=tuple(missed),
    )


def aggregate(scores: Iterable[CompletenessScore]) -> float:
    """Mean coverage fraction, macro-averaged over questions.

    Macro rather than micro (total covered / total aspects) so a question with
    eight aspects does not outweigh four one-aspect questions; each question is
    one observation of "did the engine answer what was asked". Returns 0.0 on
    empty input rather than raising, so a report section can render before any
    answers exist.
    """
    fractions = [s.fraction for s in scores]
    if not fractions:
        return 0.0
    return sum(fractions) / len(fractions)


def most_missed(
    pairs: Iterable[tuple[Question, AnswerRecord]],
    limit: int = 10,
) -> list[tuple[str, int]]:
    """The aspects engines omit most often, as (aspect, miss count) pairs.

    This is the diagnostic half of the axis: an aggregate of 0.6 says engines
    are incomplete, while a ranked miss list says *what* they skip -- and, just
    as usefully, flags aspects phrased so narrowly that no answer ever matches
    them, which is a dataset bug rather than an engine failure.

    Ties break alphabetically so the output is stable across runs.
    """
    counts: Counter[str] = Counter()
    for question, record in pairs:
        counts.update(completeness(question, record).missed)

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:limit]
