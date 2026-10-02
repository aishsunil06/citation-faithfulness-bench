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


# --------------------------------------------------------------------------
# Label prioritisation
# --------------------------------------------------------------------------


def _two_mode_setup():
    from cfbench.engines.faulty import DroppedCitation

    q_num = Question(id="q1", text="Revenue?", failure_modes=(FailureMode.NUMERIC,))
    q_neg = Question(
        id="q2", text="Any incidents?", failure_modes=(FailureMode.NEGATION,)
    )
    eng = MockEngine(
        script={
            "q1": ("Revenue was 412 million dollars [1].", [("u1", SNIPPET)]),
            "q2": ("No incidents were found [1].", [("u2", "No incidents found.")]),
        }
    )
    clean = [eng.answer(q_num), eng.answer(q_neg)]

    faulty = DroppedCitation(eng, seed=0)
    inj = [faulty.inject(q_num), faulty.inject(q_neg)]
    oracle = [lb for r in inj for lb in r.oracle]
    return [q_num, q_neg], clean + [r.record for r in inj], oracle


def test_oracle_fraction_zero_drops_known_verdict_claims():
    from cfbench.labeling import pending_claims, prioritize

    questions, answers, oracle = _two_mode_setup()

    raw = pending_claims(answers, oracle, "human:a")
    pri = prioritize(questions, answers, oracle, "human:a", oracle_fraction=0.0)

    assert len(pri) < len(raw)
    oracle_ids = {lb.claim_id for lb in oracle}
    assert not any(claim.id in oracle_ids for _rec, claim in pri)


def test_oracle_fraction_one_keeps_everything():
    from cfbench.labeling import pending_claims, prioritize

    questions, answers, oracle = _two_mode_setup()
    pri = prioritize(questions, answers, oracle, "human:a", oracle_fraction=1.0)
    assert len(pri) == len(pending_claims(answers, oracle, "human:a"))


def test_default_queue_keeps_some_known_verdict_claims():
    """Regression: excluding all oracle claims destroyed label variance.

    With a verbatim-quoting engine, oracle claims are the only ones that are
    not trivially supported. Dropping them made every human label 'supported',
    which drove Cohen's kappa to 0.000 at 82% raw accuracy. The default must
    retain some so the label set can disagree with itself.
    """
    from cfbench.labeling import prioritize

    questions, answers, oracle = _two_mode_setup()
    oracle_ids = {lb.claim_id for lb in oracle}

    pri = prioritize(questions, answers, oracle, "human:a")

    assert any(claim.id in oracle_ids for _rec, claim in pri)


def test_oracle_fraction_must_be_a_probability():
    from cfbench.labeling import prioritize

    questions, answers, oracle = _two_mode_setup()
    with pytest.raises(ValueError):
        prioritize(questions, answers, oracle, "human:a", oracle_fraction=2.0)


def test_prioritize_alternates_between_failure_modes():
    # A 2-label budget must not spend both on the same mode.
    from cfbench.labeling import prioritize

    questions, answers, oracle = _two_mode_setup()
    pri = prioritize(questions, answers, oracle, "human:a")
    q_by_id = {q.id: q for q in questions}
    first_two = [
        q_by_id[rec.question_id].failure_modes[0] for rec, _c in pri[:2]
    ]
    assert len(set(first_two)) == 2


def test_ambiguity_peaks_near_a_judge_decision_boundary():
    from cfbench.labeling import claim_ambiguity
    from cfbench.schema import Claim

    def mk(text):
        return Claim(question_id="q1", text=text, citation_ids=["c1"])

    verbatim = mk("Northwind reported revenue of 412 million dollars")
    unrelated = mk("Completely different subject matter entirely here")

    clear = claim_ambiguity(verbatim, [SNIPPET])
    borderline = claim_ambiguity(
        mk("Northwind revenue dollars unrelated padding words here now"), [SNIPPET]
    )

    assert claim_ambiguity(unrelated, [SNIPPET]) < borderline
    assert clear < borderline


def test_ambiguity_is_zero_for_uncited_claims():
    from cfbench.labeling import claim_ambiguity
    from cfbench.schema import Claim

    bare = Claim(question_id="q1", text="Something asserted.", citation_ids=[])
    assert claim_ambiguity(bare, []) == 0.0


def test_label_session_respects_prioritised_order(tmp_path):
    from cfbench.labeling import label_session, load_labels, prioritize

    questions, answers, oracle = _two_mode_setup()
    path = tmp_path / "labels.jsonl"
    from cfbench.schema import write_jsonl
    write_jsonl(path, oracle)

    expected_first = prioritize(questions, answers, oracle, "human:a")[0][1].id

    replies = iter(["s", ""])
    label_session(
        questions, answers, path, "human:a", limit=1,
        input_fn=lambda _: next(replies), print_fn=lambda *a, **k: None,
    )

    human = [lb for lb in load_labels(path) if lb.is_human]
    assert len(human) == 1
    assert human[0].claim_id == expected_first


# --------------------------------------------------------------------------
# Lossy engine: producing genuinely borderline claims
# --------------------------------------------------------------------------


def _lossy_engine(top_k=2):
    from cfbench.engines.lossy import LossyEngine
    from cfbench.retrieval import build_index, load_corpus
    from cfbench.schema import read_jsonl

    corpus = load_corpus(read_jsonl("data/corpus.jsonl"))
    return LossyEngine(index=build_index(corpus, 90, 20), top_k=top_k)


def _verbatim_share(engine, questions):
    verbatim = total = 0
    for q in questions:
        rec = engine.answer(q)
        for c in rec.claims:
            snips = rec.snippets_for(c)
            if not snips:
                continue
            total += 1
            if any(c.text.rstrip(".").strip() in s for s in snips):
                verbatim += 1
    return verbatim / total if total else 0.0


def test_lossy_engine_is_substantially_less_verbatim_than_extractive():
    # The whole point: the extractive engine quotes verbatim, so a human
    # labelling it always presses the same key and kappa cannot be computed.
    from cfbench.engines import ExtractiveEngine
    from cfbench.retrieval import build_index, load_corpus
    from cfbench.schema import question_from_dict, read_jsonl

    questions = [question_from_dict(r) for r in read_jsonl("data/questions.jsonl")]
    corpus = load_corpus(read_jsonl("data/corpus.jsonl"))
    index = build_index(corpus, 90, 20)

    extractive = _verbatim_share(ExtractiveEngine(index=index, top_k=2), questions)
    lossy = _verbatim_share(_lossy_engine(), questions)

    assert extractive == 1.0
    assert lossy < 0.75


def test_lossy_engine_produces_a_mixed_verdict_distribution():
    from cfbench.judge import LexicalJudge
    from cfbench.schema import question_from_dict, read_jsonl

    questions = [question_from_dict(r) for r in read_jsonl("data/questions.jsonl")]
    engine, judge = _lossy_engine(), LexicalJudge()

    verdicts = set()
    for q in questions:
        rec = engine.answer(q)
        for c in rec.claims:
            verdicts.add(judge.judge(c, rec.snippets_for(c)).verdict)

    assert len(verdicts) >= 2, "no variance means kappa cannot be computed"


def test_lossy_engine_is_deterministic_per_question():
    from cfbench.schema import FailureMode, Question

    q = Question(id="q1", text="Revenue?", failure_modes=(FailureMode.NUMERIC,))
    a = _lossy_engine().answer(q).answer_text
    b = _lossy_engine().answer(q).answer_text
    assert a == b


def test_years_are_never_rounded():
    from cfbench.engines.lossy import _looks_like_year, _round_number
    import random

    assert _looks_like_year("2025", 2025.0)
    assert not _looks_like_year("412", 412.0)

    # "March 2025" must survive untouched; rounding it to 2000 is nonsense,
    # not a borderline case.
    assert _round_number("Signed in March 2025.", random.Random(0)) is None


def test_rounding_uses_two_significant_figures():
    import random

    from cfbench.engines.lossy import _round_number

    out, _why = _round_number("Revenue was 412 million.", random.Random(0))
    assert "roughly 410" in out


def test_welding_preserves_proper_nouns():
    import random

    from cfbench.engines.lossy import _weld

    out, _why = _weld(
        ["The site has capacity.", "Helion Grid sells capacity."], random.Random(0)
    )
    assert "Helion Grid" in out and "helion" not in out


def test_welding_lowercases_safe_sentence_starters():
    import random

    from cfbench.engines.lossy import _weld

    out, _why = _weld(
        ["The site has capacity.", "The company was founded in 2019."],
        random.Random(0),
    )
    assert ", and the company" in out


def test_dropping_a_qualifier_removes_the_scope_clause():
    import random

    from cfbench.engines.lossy import _drop_qualifier

    result = _drop_qualifier(
        "Revenue was 412 million dollars in fiscal 2024.", random.Random(0)
    )
    assert result is not None
    out, why = result
    assert "fiscal 2024" not in out
    assert "412 million dollars" in out


# --------------------------------------------------------------------------
# Label durability across re-runs
# --------------------------------------------------------------------------


def test_claim_ids_are_stable_across_identical_runs():
    """Regression: random ids orphaned 17 real human labels on one re-run.

    Human labels are stored by claim id and are the only irreplaceable asset
    here, so re-running the same engine over the same data must reproduce the
    same ids.
    """
    questions, answers_a, _ = _two_mode_setup()
    _questions, answers_b, _ = _two_mode_setup()

    ids_a = [c.id for r in answers_a for c in r.claims]
    ids_b = [c.id for r in answers_b for c in r.claims]

    assert ids_a == ids_b
    assert all(i.startswith("cl_") for i in ids_a)


def test_lossy_engine_claim_ids_are_stable():
    from cfbench.schema import FailureMode, Question

    q = Question(id="q1", text="Revenue?", failure_modes=(FailureMode.NUMERIC,))
    a = [c.id for c in _lossy_engine().answer(q).claims]
    b = [c.id for c in _lossy_engine().answer(q).claims]
    assert a == b and a


def test_changing_a_claim_changes_its_id():
    # A reworded claim must NOT inherit the old label: the label described
    # different text.
    from cfbench.schema import content_claim_id

    base = content_claim_id("eng", "q1", 0, "Revenue was 412 million.")
    drifted = content_claim_id("eng", "q1", 0, "Revenue was 900 million.")
    assert base != drifted


def test_fault_injected_claims_do_not_collide_with_clean_ones():
    # Both are built from the same base engine, so the engine name must be
    # part of the id or the clean and corrupted claims would share labels.
    from cfbench.engines.faulty import NumericDrift

    clean = engine().answer(Q)
    injected = NumericDrift(engine(), seed=1).inject(Q).record

    assert {c.id for c in clean.claims}.isdisjoint({c.id for c in injected.claims})


def test_oracle_labels_point_at_post_corruption_ids():
    from cfbench.engines.faulty import NumericDrift

    result = NumericDrift(engine(), seed=1).inject(Q)
    live = {c.id for c in result.record.claims}

    assert result.oracle
    assert all(lb.claim_id in live for lb in result.oracle), "orphaned oracle label"


def test_citation_ids_are_content_addressed_and_resolve():
    result = NumericDrift(
        engine(citations=(("u1", SNIPPET), ("u2", OTHER))), seed=0
    ).inject(Q)
    rec = result.record

    assert all(c.id.startswith("c_") for c in rec.citations)
    for claim in rec.claims:
        for cid in claim.citation_ids:
            assert rec.citation_by_id(cid) is not None
