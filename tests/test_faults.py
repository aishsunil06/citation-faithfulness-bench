"""Fault injection and uncertainty metrics."""

import pytest

from cfbench.engines import MockEngine
from cfbench.engines.faulty import (
    DroppedCitation,
    NumericDrift,
    UnsupportedPadding,
    WrongSource,
    wrap,
)
from cfbench.judge import LexicalJudge
from cfbench.metrics import bootstrap_ci, detection_rates, labels_needed, score
from cfbench.schema import FailureMode, Label, Question, Verdict

Q = Question(id="q1", text="Revenue?", failure_modes=(FailureMode.NUMERIC,))

SNIPPET = "Northwind reported revenue of 412 million dollars in fiscal 2024."
OTHER = "Helion Grid operates battery storage in Mesa County."


def engine(answer="Northwind reported revenue of 412 million dollars [1].",
           citations=((("u1"), SNIPPET),)):
    return MockEngine(script={"q1": (answer, [tuple(c) for c in citations])})


def test_numeric_drift_changes_the_number_and_flags_unsupported():
    result = NumericDrift(engine(), seed=1).inject(Q)

    assert "412" not in result.record.claims[0].text
    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.UNSUPPORTED
    assert "412" in result.oracle[0].rationale


def test_numeric_drift_skips_claims_without_numbers():
    result = NumericDrift(
        engine(answer="Revenue rose sharply [1].")
    ).inject(Q)
    assert result.oracle == []


def test_wrong_source_repoints_the_citation():
    result = WrongSource(
        engine(citations=(("u1", SNIPPET), ("u2", OTHER)))
    ).inject(Q)

    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.UNSUPPORTED
    claim = result.record.claims[0]
    cited = result.record.citation_by_id(claim.citation_ids[0])
    assert cited.snippet == OTHER


def test_wrong_source_needs_an_alternative_to_point_at():
    # Only one citation available, so there is nothing to mis-attribute to.
    result = WrongSource(engine()).inject(Q)
    assert result.oracle == []


def test_dropped_citation_yields_uncited_not_unsupported():
    result = DroppedCitation(engine()).inject(Q)

    assert result.record.claims[0].is_uncited
    assert result.oracle[0].verdict is Verdict.UNCITED


def test_padding_appends_exactly_one_claim_per_answer():
    injector = UnsupportedPadding(engine())
    before = len(injector.engine.answer(Q).claims)
    result = injector.inject(Q)

    assert len(result.record.claims) == before + 1
    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.UNSUPPORTED


def test_rate_zero_injects_nothing():
    result = NumericDrift(engine(), rate=0.0, seed=3).inject(Q)
    assert result.oracle == []


def test_rate_must_be_a_probability():
    with pytest.raises(ValueError):
        NumericDrift(engine(), rate=1.5)


def test_injection_is_deterministic_for_a_given_seed():
    a = NumericDrift(engine(), seed=7).inject(Q).record.claims[0].text
    b = NumericDrift(engine(), seed=7).inject(Q).record.claims[0].text
    assert a == b


def test_wrap_rejects_unknown_faults():
    with pytest.raises(ValueError):
        wrap(engine(), "not-a-fault")


def test_engine_name_records_the_fault():
    assert wrap(engine(), "numeric-drift").name.endswith("+numeric-drift")


def test_lexical_judge_catches_injected_numeric_drift():
    # The end-to-end claim the benchmark rests on: a known corruption is
    # detected without any human reading it.
    result = NumericDrift(engine(), seed=2).inject(Q)
    judge = LexicalJudge()
    judged = [
        judge.judge(c, result.record.snippets_for(c))
        for c in result.record.claims
    ]

    rates = detection_rates(result.oracle, judged)
    assert rates["numeric-drift"].rate == 1.0


def test_detection_rates_group_by_fault_family():
    oracle = [
        Label(claim_id="a", verdict=Verdict.UNSUPPORTED, labeler="oracle:numeric-drift"),
        Label(claim_id="b", verdict=Verdict.UNCITED, labeler="oracle:dropped-citation"),
    ]
    judged = [
        Label(claim_id="a", verdict=Verdict.UNSUPPORTED, labeler="judge:x"),
        Label(claim_id="b", verdict=Verdict.SUPPORTED, labeler="judge:x"),
    ]
    rates = detection_rates(oracle, judged)

    assert rates["numeric-drift"].rate == 1.0
    assert rates["dropped-citation"].rate == 0.0


def test_detection_ignores_oracle_claims_the_judge_never_saw():
    oracle = [
        Label(claim_id="a", verdict=Verdict.UNSUPPORTED, labeler="oracle:f"),
        Label(claim_id="ghost", verdict=Verdict.UNSUPPORTED, labeler="oracle:f"),
    ]
    judged = [Label(claim_id="a", verdict=Verdict.UNSUPPORTED, labeler="judge:x")]

    assert detection_rates(oracle, judged)["f"].n == 1


def test_bootstrap_interval_brackets_the_point_estimate():
    labels = [
        Label(claim_id=str(i), verdict=Verdict.SUPPORTED, labeler="judge:x")
        for i in range(30)
    ] + [
        Label(claim_id=f"u{i}", verdict=Verdict.UNSUPPORTED, labeler="judge:x")
        for i in range(10)
    ]
    point = score(labels).strict
    lo, hi = bootstrap_ci(labels, "strict", n_boot=500)

    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_on_unanimous_labels_has_zero_width():
    labels = [
        Label(claim_id=str(i), verdict=Verdict.SUPPORTED, labeler="judge:x")
        for i in range(20)
    ]
    assert bootstrap_ci(labels, "strict", n_boot=200) == (1.0, 1.0)


def test_bootstrap_on_empty_input_is_degenerate_not_an_error():
    assert bootstrap_ci([], "strict") == (0.0, 0.0)


def test_smaller_intervals_require_more_labels():
    assert labels_needed(0.10) < labels_needed(0.05) < labels_needed(0.03)
    assert labels_needed(0.05) == 385      # textbook value for p=0.5, z=1.96


def test_labels_needed_rejects_impossible_targets():
    with pytest.raises(ValueError):
        labels_needed(0.0)
