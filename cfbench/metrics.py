"""Scoring and judge-validation metrics.

Two distinct jobs live here and should not be confused:

1. **Faithfulness scoring** - how well did an *engine* do, given labels.
2. **Judge validation** - how well did the *automated judge* reproduce the
   human labels. This is the part most eval projects skip. An automated judge
   whose agreement with humans is unmeasured is not a measurement instrument,
   it is a vibe. Every number this benchmark reports from the judge is paired
   with the judge's agreement on the human-labelled subset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .schema import FailureMode, Label, Question, Verdict

# Partial credit weight used by the "lenient" score. Strict scoring ignores it.
PARTIAL_WEIGHT = 0.5


@dataclass
class FaithfulnessScore:
    n_claims: int
    counts: dict[Verdict, int]

    @property
    def strict(self) -> float:
        """Fraction of claims whose citation fully supports them."""
        if not self.n_claims:
            return 0.0
        return self.counts.get(Verdict.SUPPORTED, 0) / self.n_claims

    @property
    def lenient(self) -> float:
        """Strict, but partial support earns half credit."""
        if not self.n_claims:
            return 0.0
        full = self.counts.get(Verdict.SUPPORTED, 0)
        part = self.counts.get(Verdict.PARTIAL, 0)
        return (full + PARTIAL_WEIGHT * part) / self.n_claims

    @property
    def uncited_rate(self) -> float:
        """Fraction of claims the engine asserted with no citation at all."""
        if not self.n_claims:
            return 0.0
        return self.counts.get(Verdict.UNCITED, 0) / self.n_claims

    def as_dict(self) -> dict:
        return {
            "n_claims": self.n_claims,
            "strict": round(self.strict, 4),
            "lenient": round(self.lenient, 4),
            "uncited_rate": round(self.uncited_rate, 4),
            "counts": {v.value: self.counts.get(v, 0) for v in Verdict},
        }


def score(labels: Iterable[Label]) -> FaithfulnessScore:
    counts: Counter[Verdict] = Counter(lb.verdict for lb in labels)
    return FaithfulnessScore(n_claims=sum(counts.values()), counts=dict(counts))


def score_by_failure_mode(
    labels: Iterable[Label],
    claim_to_question: Mapping[str, str],
    questions: Sequence[Question],
) -> dict[FailureMode, FaithfulnessScore]:
    """Decompose a headline score across failure modes.

    A question with two failure modes contributes its claims to both buckets,
    so bucket sizes intentionally sum to more than the claim count.
    """
    by_qid = {q.id: q for q in questions}
    buckets: dict[FailureMode, list[Label]] = defaultdict(list)

    for lb in labels:
        qid = claim_to_question.get(lb.claim_id)
        question = by_qid.get(qid) if qid else None
        if question is None:
            continue
        for mode in question.failure_modes:
            buckets[mode].append(lb)

    return {mode: score(group) for mode, group in buckets.items()}


# --------------------------------------------------------------------------
# Judge validation
# --------------------------------------------------------------------------


@dataclass
class Agreement:
    """How closely a judge reproduced human labels on the overlapping claims."""

    n_compared: int
    n_agree: int
    confusion: dict[tuple[Verdict, Verdict], int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.n_agree / self.n_compared if self.n_compared else 0.0

    @property
    def cohens_kappa(self) -> float:
        """Chance-corrected agreement.

        Reported alongside raw accuracy because these verdicts are heavily
        skewed toward SUPPORTED: a judge that blindly answers SUPPORTED can
        post high accuracy and a kappa near zero. Convention: <0.2 poor,
        0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 near-perfect.
        """
        n = self.n_compared
        if not n:
            return 0.0

        human_marg: Counter[Verdict] = Counter()
        judge_marg: Counter[Verdict] = Counter()
        for (human_v, judge_v), count in self.confusion.items():
            human_marg[human_v] += count
            judge_marg[judge_v] += count

        p_observed = self.n_agree / n
        p_expected = sum(
            (human_marg[v] / n) * (judge_marg[v] / n) for v in Verdict
        )
        if p_expected >= 1.0:
            return 1.0
        return (p_observed - p_expected) / (1.0 - p_expected)

    def per_verdict(self) -> dict[str, dict[str, float]]:
        """Precision/recall per verdict, treating the human label as truth."""
        out: dict[str, dict[str, float]] = {}
        for v in Verdict:
            tp = self.confusion.get((v, v), 0)
            judge_total = sum(
                c for (h, j), c in self.confusion.items() if j is v
            )
            human_total = sum(
                c for (h, j), c in self.confusion.items() if h is v
            )
            out[v.value] = {
                "precision": round(tp / judge_total, 4) if judge_total else 0.0,
                "recall": round(tp / human_total, 4) if human_total else 0.0,
                "human_n": human_total,
                "judge_n": judge_total,
            }
        return out

    def as_dict(self) -> dict:
        return {
            "n_compared": self.n_compared,
            "accuracy": round(self.accuracy, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "per_verdict": self.per_verdict(),
            "confusion": {
                f"{h.value}->{j.value}": c for (h, j), c in sorted(
                    self.confusion.items(), key=lambda kv: (kv[0][0].value, kv[0][1].value)
                )
            },
        }


def agreement(human: Iterable[Label], judge: Iterable[Label]) -> Agreement:
    """Compare a judge against humans on claims both labelled.

    Claims labelled by only one side are ignored rather than counted as
    disagreement, so partial human labelling does not depress the judge's
    apparent quality.
    """
    human_by_claim = {lb.claim_id: lb.verdict for lb in human}
    judge_by_claim = {lb.claim_id: lb.verdict for lb in judge}

    shared = sorted(set(human_by_claim) & set(judge_by_claim))
    confusion: Counter[tuple[Verdict, Verdict]] = Counter()
    n_agree = 0
    for claim_id in shared:
        hv, jv = human_by_claim[claim_id], judge_by_claim[claim_id]
        confusion[(hv, jv)] += 1
        if hv is jv:
            n_agree += 1

    return Agreement(n_compared=len(shared), n_agree=n_agree, confusion=dict(confusion))


def split_labels(labels: Iterable[Label]) -> tuple[list[Label], dict[str, list[Label]]]:
    """Partition a label store into human labels and judge labels by labeler."""
    humans: list[Label] = []
    judges: dict[str, list[Label]] = defaultdict(list)
    for lb in labels:
        if lb.is_human:
            humans.append(lb)
        else:
            judges[lb.labeler].append(lb)
    return humans, dict(judges)
