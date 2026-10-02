from cfbench.metrics import agreement, score, score_by_failure_mode
from cfbench.schema import FailureMode, Label, Question, Verdict


def lab(claim_id, verdict, labeler="human:test"):
    return Label(claim_id=claim_id, verdict=verdict, labeler=labeler)


def test_strict_counts_only_supported():
    labels = [
        lab("a", Verdict.SUPPORTED),
        lab("b", Verdict.PARTIAL),
        lab("c", Verdict.UNSUPPORTED),
        lab("d", Verdict.UNCITED),
    ]
    sc = score(labels)

    assert sc.n_claims == 4
    assert sc.strict == 0.25
    assert sc.lenient == 0.375        # 1 full + 0.5 partial, over 4
    assert sc.uncited_rate == 0.25


def test_empty_label_set_scores_zero_not_nan():
    sc = score([])
    assert sc.strict == 0.0 and sc.lenient == 0.0 and sc.uncited_rate == 0.0


def test_perfect_agreement_gives_kappa_one():
    human = [lab("a", Verdict.SUPPORTED), lab("b", Verdict.UNSUPPORTED)]
    judge = [
        lab("a", Verdict.SUPPORTED, "judge:x"),
        lab("b", Verdict.UNSUPPORTED, "judge:x"),
    ]
    agr = agreement(human, judge)

    assert agr.n_compared == 2
    assert agr.accuracy == 1.0
    assert agr.cohens_kappa == 1.0


def test_constant_judge_has_high_accuracy_but_no_kappa():
    # The exact pathology kappa exists to expose: a judge that always says
    # SUPPORTED looks good on accuracy alone.
    human = [lab(str(i), Verdict.SUPPORTED) for i in range(9)]
    human.append(lab("9", Verdict.UNSUPPORTED))
    judge = [lab(str(i), Verdict.SUPPORTED, "judge:lazy") for i in range(10)]

    agr = agreement(human, judge)

    assert agr.accuracy == 0.9
    assert agr.cohens_kappa == 0.0


def test_claims_labelled_by_only_one_side_are_ignored():
    human = [lab("a", Verdict.SUPPORTED), lab("b", Verdict.PARTIAL)]
    judge = [lab("a", Verdict.SUPPORTED, "judge:x")]

    agr = agreement(human, judge)

    assert agr.n_compared == 1
    assert agr.accuracy == 1.0


def test_per_verdict_precision_and_recall():
    human = [lab("a", Verdict.SUPPORTED), lab("b", Verdict.UNSUPPORTED)]
    judge = [
        lab("a", Verdict.SUPPORTED, "judge:x"),
        lab("b", Verdict.SUPPORTED, "judge:x"),
    ]
    stats = agreement(human, judge).per_verdict()

    assert stats["supported"]["recall"] == 1.0
    assert stats["supported"]["precision"] == 0.5
    assert stats["unsupported"]["recall"] == 0.0


def test_failure_mode_buckets_overlap_for_multi_tagged_questions():
    q = Question(
        id="q1",
        text="?",
        failure_modes=(FailureMode.NUMERIC, FailureMode.RECENCY),
    )
    labels = [lab("c1", Verdict.SUPPORTED)]

    buckets = score_by_failure_mode(labels, {"c1": "q1"}, [q])

    assert buckets[FailureMode.NUMERIC].strict == 1.0
    assert buckets[FailureMode.RECENCY].strict == 1.0


def test_labels_for_unknown_questions_are_skipped():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,))
    labels = [lab("orphan", Verdict.SUPPORTED)]

    assert score_by_failure_mode(labels, {}, [q]) == {}
