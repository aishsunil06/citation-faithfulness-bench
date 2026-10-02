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


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def bootstrap_ci(
    labels: Sequence[Label],
    statistic: str = "strict",
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a faithfulness statistic.

    Reported on every score because this benchmark runs on tens to low
    hundreds of claims, where a 6-point gap between two engines is routinely
    noise. An interval is the difference between "engine A is better" and
    "engine A scored higher this run".

    Resampling is over claims, which slightly understates uncertainty: claims
    from the same question are correlated, so a question-level cluster
    bootstrap would be wider. Noted rather than hidden.
    """
    import random

    pool = list(labels)
    if not pool:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(pool)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [pool[rng.randrange(n)] for _ in range(n)]
        stats.append(getattr(score(sample), statistic))

    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = stats[int(alpha * n_boot)]
    hi = stats[min(int((1.0 - alpha) * n_boot), n_boot - 1)]
    return (lo, hi)


@dataclass
class DetectionRate:
    """How often a judge returned the oracle's verdict on injected faults."""

    fault: str
    n: int
    n_detected: int
    confusions: dict[Verdict, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.n_detected / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        return {
            "fault": self.fault,
            "n": self.n,
            "detected": self.n_detected,
            "rate": round(self.rate, 4),
            "judge_said": {v.value: c for v, c in sorted(
                self.confusions.items(), key=lambda kv: kv[0].value
            )},
        }


def detection_rates(
    oracle: Iterable[Label],
    judge: Iterable[Label],
) -> dict[str, DetectionRate]:
    """Per-fault detection rate for one judge.

    Oracle labelers are named `oracle:<fault>`, so the fault type is recovered
    from the labeler string and each injection family is scored separately. A
    judge can be excellent at numeric drift and blind to wrong-source
    attribution, and an aggregate number would hide that.
    """
    judge_by_claim = {lb.claim_id: lb.verdict for lb in judge}

    grouped: dict[str, list[Label]] = defaultdict(list)
    for lb in oracle:
        fault = lb.labeler.split(":", 1)[1] if ":" in lb.labeler else lb.labeler
        grouped[fault].append(lb)

    out: dict[str, DetectionRate] = {}
    for fault, truth in grouped.items():
        n = 0
        hit = 0
        confusions: Counter[Verdict] = Counter()
        for lb in truth:
            got = judge_by_claim.get(lb.claim_id)
            if got is None:
                continue
            n += 1
            confusions[got] += 1
            if got is lb.verdict:
                hit += 1
        out[fault] = DetectionRate(
            fault=fault, n=n, n_detected=hit, confusions=dict(confusions)
        )
    return out


def labels_needed(
    target_half_width: float = 0.05,
    p_estimate: float = 0.5,
    confidence_z: float = 1.96,
) -> int:
    """How many labels to reach a target confidence-interval half-width.

    Answers the practical question "how many pairs do I have to label?" using
    the normal approximation to a proportion. `p_estimate=0.5` is the
    worst case and therefore the safe default.
    """
    if not 0 < target_half_width < 1:
        raise ValueError("target_half_width must be in (0, 1)")
    n = (confidence_z ** 2) * p_estimate * (1 - p_estimate) / (target_half_width ** 2)
    return int(n) + 1
