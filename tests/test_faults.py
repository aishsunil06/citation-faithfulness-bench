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


def test_numeric_drift_changes_the_number_and_flags_contradicted():
    # The source still states the original figure, so it actively disagrees
    # with the corrupted claim rather than merely failing to support it.
    result = NumericDrift(engine(), seed=1).inject(Q)

    assert "412" not in result.record.claims[0].text
    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.CONTRADICTED
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


# --------------------------------------------------------------------------
# Directional quantities must not be corrupted
# --------------------------------------------------------------------------


def test_bounded_quantities_are_never_drifted():
    """Regression: shifting a bound produces entailment, not contradiction.

    "reach 3.1 billion by 2030" entails "reach 3.1 billion by 3248", because a
    later deadline is a weaker claim the source already supports. Labelling
    that CONTRADICTED makes the oracle wrong, and a judge agreeing with a wrong
    oracle measures nothing.
    """
    from cfbench.engines.faulty import _is_directional

    text = "The market will reach 3.1 billion dollars by 2030."
    idx = text.index("2030")
    assert _is_directional(text, idx)

    plain = "Revenue was 412 million dollars."
    assert not _is_directional(plain, plain.index("412"))


def test_drift_leaves_the_bound_alone_but_may_hit_the_magnitude():
    # In "reach 3.1 billion by 2030" only the deadline is a bound. Changing
    # 3.1 to 3.4 genuinely contradicts the source, so that remains fair game;
    # changing 2030 would not, so the year must survive untouched.
    result = NumericDrift(
        engine(answer="The market will reach 3.1 billion dollars by 2030 [1].",
               citations=(("u1", "The market will reach 3.1 billion by 2030."),)),
        seed=0,
    ).inject(Q)

    assert "2030" in result.record.claims[0].text, "corrupted a bounded quantity"


def test_drift_skips_a_claim_whose_only_figure_is_a_bound():
    result = NumericDrift(
        engine(answer="The milestone will be met by 2030 [1].",
               citations=(("u1", "The milestone will be met by 2030."),)),
        seed=0,
    ).inject(Q)
    assert result.oracle == [], "corrupted a bounded quantity"


def test_drift_still_fires_on_point_values():
    result = NumericDrift(engine(), seed=0).inject(Q)
    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.CONTRADICTED


@pytest.mark.parametrize("cue", ["at least", "up to", "more than", "under", "nearly"])
def test_common_bound_cues_are_all_recognised(cue):
    from cfbench.engines.faulty import _is_directional

    text = f"The site holds {cue} 240 megawatt-hours."
    assert _is_directional(text, text.index("240"))


def test_years_are_never_drifted_even_outside_a_bound():
    # "as of the 2025 reporting year" has no bound cue, but shifting it to 2026
    # makes the claim UNSUPPORTED (source silent on 2026), not CONTRADICTED.
    # An oracle that cannot be defended is worse than no oracle.
    result = NumericDrift(
        engine(answer="Capacity was 240 units as of the 2025 reporting year [1].",
               citations=(("u1", "Capacity was 240 units as of the 2025 year."),)),
        seed=0,
    ).inject(Q)

    assert "2025" in result.record.claims[0].text
    assert result.oracle, "should still have corrupted the magnitude"
    assert "240" not in result.record.claims[0].text


# --------------------------------------------------------------------------
# EntitySwap: substituting a named entity the source contradicts
# --------------------------------------------------------------------------
#
# The oracle bar for this injector is CONTRADICTED, which is only defensible
# when the cited source itself makes a *competing* statement about the
# substituted entity. Swapping in an entity the source never discusses yields
# a claim the source is merely silent about, which is UNSUPPORTED, so the
# injector must decline instead of guessing.


A_AND_B = (
    "Northwind Energy reported revenue of 412 million dollars in fiscal 2024. "
    "Helion Grid reported revenue of 98 million dollars in fiscal 2024."
)
A_ONLY = "Northwind Energy reported revenue of 412 million dollars in fiscal 2024."
B_ELSEWHERE = "Helion Grid operates battery storage in Mesa County."

SWAP_ANSWER = "Northwind Energy reported revenue of 412 million dollars [1]."


def swap_engine(answer=SWAP_ANSWER, cited=A_AND_B, other=B_ELSEWHERE):
    cits = [("u1", cited)]
    if other is not None:
        cits.append(("u2", other))
    return MockEngine(script={"q1": (answer, cits)})


def test_entity_swap_replaces_a_named_entity_and_flags_contradicted():
    from cfbench.engines.faulty import EntitySwap

    result = EntitySwap(swap_engine(), seed=1).inject(Q)
    text = result.record.claims[0].text

    assert "Northwind Energy" not in text
    assert "Helion Grid" in text
    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.CONTRADICTED
    assert "Northwind Energy" in result.oracle[0].rationale


def test_entity_swap_rewrites_the_answer_text_too():
    from cfbench.engines.faulty import EntitySwap

    result = EntitySwap(swap_engine(), seed=1).inject(Q)
    assert "Helion Grid" in result.record.answer_text
    assert "Northwind Energy" not in result.record.answer_text


def test_entity_swap_oracle_points_at_a_live_claim_id():
    from cfbench.engines.faulty import EntitySwap

    result = EntitySwap(swap_engine(), seed=1).inject(Q)
    live = {c.id for c in result.record.claims}
    assert result.oracle and all(lb.claim_id in live for lb in result.oracle)


def test_entity_swap_declines_when_no_alternative_entity_is_available():
    # A single citation means there is no non-cited source to draw from.
    from cfbench.engines.faulty import EntitySwap

    assert EntitySwap(swap_engine(cited=A_AND_B, other=None)).inject(Q).oracle == []


def test_entity_swap_declines_when_the_cited_source_is_silent_about_the_swap():
    """The guard that keeps the oracle honest.

    "Helion Grid reported revenue of 412 million" cited to a source that only
    discusses Northwind is UNSUPPORTED, not CONTRADICTED: both companies could
    perfectly well have earned 412 million. Declining is the only defensible
    move.
    """
    from cfbench.engines.faulty import EntitySwap

    result = EntitySwap(swap_engine(cited=A_ONLY), seed=1).inject(Q)
    assert result.oracle == []
    assert "Northwind Energy" in result.record.claims[0].text


def test_entity_swap_declines_when_the_source_states_the_same_figure():
    # If the source says Helion Grid also reported 412 million, the swapped
    # claim is SUPPORTED, which is the opposite of the intended fault.
    from cfbench.engines.faulty import EntitySwap

    cited = (
        "Northwind Energy reported revenue of 412 million dollars in fiscal 2024. "
        "Helion Grid reported revenue of 412 million dollars in fiscal 2023."
    )
    assert EntitySwap(swap_engine(cited=cited), seed=1).inject(Q).oracle == []


def test_entity_swap_declines_a_claim_with_no_multi_word_entity():
    from cfbench.engines.faulty import EntitySwap

    result = EntitySwap(
        swap_engine(answer="Revenue reached 412 million dollars [1].")
    ).inject(Q)
    assert result.oracle == []


def test_entity_swap_declines_a_claim_whose_only_figure_is_a_bound():
    # "more than 412" is a bound: a competing figure does not contradict it
    # cleanly, so the number cannot anchor the comparison.
    from cfbench.engines.faulty import EntitySwap

    answer = "Northwind Energy reported more than 412 million dollars [1]."
    cited = (
        "Northwind Energy reported more than 412 million dollars in fiscal 2024. "
        "Helion Grid reported revenue of 98 million dollars in fiscal 2024."
    )
    assert EntitySwap(swap_engine(answer=answer, cited=cited)).inject(Q).oracle == []


def test_entity_swap_declines_when_the_competing_figure_is_only_a_year():
    # A sentence about Helion Grid containing nothing but a year states no
    # competing quantity, so there is no contradiction to rely on.
    from cfbench.engines.faulty import EntitySwap

    cited = (
        "Northwind Energy reported revenue of 412 million dollars in fiscal 2024. "
        "Helion Grid reported revenue in fiscal 2019."
    )
    assert EntitySwap(swap_engine(cited=cited), seed=1).inject(Q).oracle == []


def test_entity_swap_declines_when_the_predicate_does_not_match():
    # The source says something about Helion Grid, but not about revenue, so
    # it is not a competing statement about the claim's fact.
    from cfbench.engines.faulty import EntitySwap

    cited = (
        "Northwind Energy reported revenue of 412 million dollars in fiscal 2024. "
        "Helion Grid installed 98 battery racks at its depot."
    )
    assert EntitySwap(swap_engine(cited=cited), seed=1).inject(Q).oracle == []


def test_entity_swap_declines_when_the_entity_already_appears_in_the_claim():
    from cfbench.engines.faulty import EntitySwap

    answer = (
        "Northwind Energy outbid Helion Grid for revenue of 412 million dollars [1]."
    )
    assert EntitySwap(swap_engine(answer=answer), seed=1).inject(Q).oracle == []


def test_entity_swap_declines_an_uncited_claim():
    from cfbench.engines.faulty import EntitySwap

    answer = "Northwind Energy reported revenue of 412 million dollars."
    assert EntitySwap(swap_engine(answer=answer), seed=1).inject(Q).oracle == []


def test_entity_swap_is_deterministic_for_a_given_seed():
    from cfbench.engines.faulty import EntitySwap

    a = EntitySwap(swap_engine(), seed=5).inject(Q).record.claims[0].text
    b = EntitySwap(swap_engine(), seed=5).inject(Q).record.claims[0].text
    assert a == b


# --------------------------------------------------------------------------
# PolarityFlip: inverting a single unhedged negation
# --------------------------------------------------------------------------


NEG_SNIPPET = (
    "The audit found no incidents at the Mesa site during the review period."
)
NEG_ANSWER = "The audit found no incidents at the Mesa site [1]."


def neg_engine(answer=NEG_ANSWER, snippet=NEG_SNIPPET):
    return MockEngine(script={"q1": (answer, [("u1", snippet)])})


def test_polarity_flip_inverts_a_negation_and_flags_contradicted():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(neg_engine(), seed=0).inject(Q)
    text = result.record.claims[0].text

    assert "no incidents" not in text
    assert "found incidents" in text
    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.CONTRADICTED


def test_polarity_flip_rewrites_the_answer_text_too():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(neg_engine(), seed=0).inject(Q)
    assert "no incidents" not in result.record.answer_text


def test_polarity_flip_handles_a_leading_negation_and_recapitalises():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(
            answer="No incidents were recorded at the Mesa site [1].",
            snippet="No incidents were recorded at the Mesa site last year.",
        ),
        seed=0,
    ).inject(Q)

    assert result.oracle[0].verdict is Verdict.CONTRADICTED
    assert result.record.claims[0].text.startswith("Incidents were recorded")


def test_polarity_flip_turns_without_into_with():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(
            answer="The plant operated without a permit in 2024 [1].",
            snippet="The plant operated without a permit in 2024, regulators said.",
        ),
        seed=0,
    ).inject(Q)

    assert "with a permit" in result.record.claims[0].text
    assert result.oracle[0].verdict is Verdict.CONTRADICTED


def test_polarity_flip_removes_never():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(
            answer="The company never disclosed the agreement [1].",
            snippet="The company never disclosed the agreement to shareholders.",
        ),
        seed=0,
    ).inject(Q)

    assert result.record.claims[0].text == "The company disclosed the agreement."
    assert result.oracle[0].verdict is Verdict.CONTRADICTED


def test_polarity_flip_declines_a_claim_with_no_polarity_marker():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(
            answer="The audit found three incidents at the Mesa site [1].",
            snippet="The audit found three incidents at the Mesa site.",
        )
    ).inject(Q)
    assert result.oracle == []


def test_polarity_flip_declines_a_hedged_claim():
    # "may not have" is modal: inverting it does not produce a clean
    # contradiction, so the injector must decline rather than guess.
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(
            answer="The audit may not have found incidents at the Mesa site [1].",
            snippet="The audit may not have found incidents at the Mesa site.",
        )
    ).inject(Q)
    assert result.oracle == []
    assert "not" in result.record.claims[0].text


def test_polarity_flip_declines_two_negation_markers():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(
            answer="The audit found no incidents and no violations at Mesa [1].",
            snippet="The audit found no incidents and no violations at Mesa.",
        )
    ).inject(Q)
    assert result.oracle == []


def test_polarity_flip_declines_when_the_source_never_states_the_negative():
    """The guard that keeps the oracle honest.

    If the cited source says nothing about incidents, the original claim was
    already unsupported, and the flipped claim is equally unsupported rather
    than contradicted.
    """
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(snippet="The Mesa site has a rated capacity of 240 megawatt-hours.")
    ).inject(Q)
    assert result.oracle == []


def test_polarity_flip_declines_when_the_source_negates_a_different_fact():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(snippet="No permits were issued for the Clearwater refinery.")
    ).inject(Q)
    assert result.oracle == []


def test_polarity_flip_declines_when_the_source_itself_hedges():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(
            snippet=(
                "The audit found no incidents at the Mesa site, which may "
                "suggest reporting gaps."
            )
        )
    ).inject(Q)
    assert result.oracle == []


@pytest.mark.parametrize(
    "claim_text",
    [
        "The site is no longer operational",
        "The site holds no more than 240 megawatt-hours",
        "The report is not only incomplete but late",
        "The filing was no later than the deadline",
    ],
)
def test_polarity_flip_declines_idioms_and_bounds(claim_text):
    # Deleting the marker from these either produces nonsense ("is longer
    # operational") or silently alters a bound, neither of which is a clean
    # contradiction.
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(answer=f"{claim_text} [1].", snippet=f"{claim_text}.")
    ).inject(Q)
    assert result.oracle == []


def test_polarity_flip_declines_an_uncited_claim():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(answer="The audit found no incidents at the Mesa site.")
    ).inject(Q)
    assert result.oracle == []


def test_polarity_flip_oracle_points_at_a_live_claim_id():
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(neg_engine(), seed=0).inject(Q)
    live = {c.id for c in result.record.claims}
    assert result.oracle and all(lb.claim_id in live for lb in result.oracle)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fault", ["entity-swap", "polarity-flip"])
def test_new_faults_are_registered_and_resolved_by_wrap(fault):
    from cfbench.engines.faulty import FAULTS

    assert fault in FAULTS
    assert wrap(engine(), fault).name.endswith(f"+{fault}")


def test_rate_zero_injects_nothing_for_the_new_faults():
    from cfbench.engines.faulty import EntitySwap, PolarityFlip

    assert EntitySwap(swap_engine(), rate=0.0, seed=3).inject(Q).oracle == []
    assert PolarityFlip(neg_engine(), rate=0.0, seed=3).inject(Q).oracle == []


@pytest.mark.parametrize(
    "claim_text",
    [
        "The agency denied that no incidents occurred at Mesa",
        "The operator failed to report no incidents at Mesa",
        "Neither operator reported no incidents at Mesa",
    ],
)
def test_polarity_flip_declines_a_second_negative_in_the_claim(claim_text):
    # Two interacting negatives: flipping one does not reliably reverse what
    # the sentence asserts, so the verdict would be arguable.
    from cfbench.engines.faulty import PolarityFlip

    result = PolarityFlip(
        neg_engine(answer=f"{claim_text} [1].", snippet=f"{claim_text}.")
    ).inject(Q)
    assert result.oracle == []


def test_polarity_flip_reads_the_negating_sentence_not_the_whole_chunk():
    # A hedge elsewhere in a long retrieved chunk says nothing about this
    # fact, so it must not veto an otherwise plain negation.
    from cfbench.engines.faulty import PolarityFlip

    snippet = (
        "The audit found no incidents at the Mesa site. "
        "Analysts may revisit the county forecast later this year."
    )
    result = PolarityFlip(neg_engine(snippet=snippet), seed=0).inject(Q)
    assert result.oracle and result.oracle[0].verdict is Verdict.CONTRADICTED


def test_polarity_flip_declines_when_only_a_hedged_sentence_negates():
    from cfbench.engines.faulty import PolarityFlip

    snippet = (
        "The Mesa site was reviewed in detail. "
        "The audit may have found no incidents at the Mesa site."
    )
    assert PolarityFlip(neg_engine(snippet=snippet), seed=0).inject(Q).oracle == []


# --------------------------------------------------------------------------
# NumericDrift: the oracle requires a supported base claim
# --------------------------------------------------------------------------
#
# Found on the first real-corpus run. The injector had assumed the claim it
# was handed was already faithful, which the extractive engine guaranteed by
# quoting verbatim but the lossy engine does not.


def test_drift_declines_when_the_snippet_never_stated_the_figure():
    # The uncorrupted claim was already UNSUPPORTED, so the corrupted one is
    # unsupported too, not contradicted. Nothing to contradict.
    result = NumericDrift(
        engine(answer="Fab 8 will produce 420 million units [1].",
               citations=(("u1", "An unrelated passage about wafer capacity."),)),
        seed=0,
    ).inject(Q)
    assert result.oracle == []


def test_drift_fires_when_the_snippet_does_state_the_figure():
    result = NumericDrift(
        engine(answer="The site produced 420 million units [1].",
               citations=(("u1", "The site produced 420 million units last year."),)),
        seed=0,
    ).inject(Q)
    assert len(result.oracle) == 1
    assert result.oracle[0].verdict is Verdict.CONTRADICTED


def test_drift_declines_on_a_number_that_is_part_of_a_name():
    # "Fab 8" -> "Fab 7" changes which facility the claim is about, which is a
    # different kind of error with an arguable verdict.
    from cfbench.engines.faulty import _is_identifier

    text = "Fab 8 produced 60000 wafers."
    assert _is_identifier(text, text.index("8"))
    assert not _is_identifier(text, text.index("60000"))


def test_identifier_guard_covers_vehicle_and_model_names():
    from cfbench.engines.faulty import _is_identifier

    for text, token in [
        ("Falcon 9 launched the payload.", "9"),
        ("The Boeing 787 entered service.", "787"),
        ("Ariane 6 replaced it.", "6"),
        ("Voyager 1 left the heliosphere.", "1"),
    ]:
        assert _is_identifier(text, text.index(token)), text


def test_drift_declines_on_a_suffixed_bound():
    # "EUR 10+ billion" -> "12+ billion" is a strictly stronger claim the
    # source does not support: unsupported, not contradicted.
    from cfbench.engines.faulty import _has_bound_suffix

    text = "committed to a 10+ billion factory"
    assert _has_bound_suffix(text, text.index("10") + 2)
    plain = "committed 412 million dollars"
    assert not _has_bound_suffix(plain, plain.index("412") + 3)


def test_states_number_requires_a_whole_token():
    from cfbench.engines.faulty import _states_number

    assert _states_number("produced 7 nm parts", "7")
    assert not _states_number("produced 17 nm parts", "7")
    assert not _states_number("in the year 2007", "7")
    assert _states_number("revenue of 3.5 billion", "3.5")


# --------------------------------------------------------------------------
# Comma-grouped numerals
# --------------------------------------------------------------------------
#
# Found on the real corpus: "3,000" tokenised as "3" and "000", and corrupting
# the "000" produced "3,0.0" -- a formatting mangle, not a competing value, so
# the oracle verdict became arguable. Real hits were "69,980,000 passengers"
# and "3,000 MWh".


def test_comma_grouped_numerals_tokenise_as_one_number():
    from cfbench.engines.faulty import _NUMBER

    text = "handled 69,980,000 passengers and 3,000 MWh and 412 units"
    assert [m.group(1) for m in _NUMBER.finditer(text)] == [
        "69,980,000", "3,000", "412",
    ]


def test_parse_number_handles_separators():
    from cfbench.engines.faulty import _parse_number

    assert _parse_number("69,980,000") == 69980000.0
    assert _parse_number("3.5") == 3.5
    assert _parse_number("not a number") is None


def test_corrupted_figures_keep_the_original_formatting():
    # A corruption visible from formatting alone tests nothing, which is the
    # same mistake as turning 2030 into 3248.
    from cfbench.engines.faulty import _format_like

    assert _format_like(75000000, "69,980,000") == "75,000,000"
    assert _format_like(450, "412") == "450"
    assert _format_like(3.4, "3.1") == "3.4"


def test_a_comma_grouped_figure_drifts_without_mangling():
    result = NumericDrift(
        engine(answer="The airport handled 69,980,000 passengers [1].",
               citations=(("u1", "The airport handled 69,980,000 passengers."),)),
        seed=0,
    ).inject(Q)

    assert len(result.oracle) == 1
    text = result.record.claims[0].text
    assert "69,980,000" not in text
    # No mangled hybrid like "69,980,0.0"
    import re
    assert not re.search(r"\d,\d*\.\d", text), text


def test_a_comma_grouped_numeral_is_not_mistaken_for_a_year():
    from cfbench.engines.faulty import _looks_like_year

    assert _looks_like_year("2030", 2030.0)
    assert not _looks_like_year("1,800", 1800.0)


# --------------------------------------------------------------------------
# LossyEngine must paraphrase, not garble
# --------------------------------------------------------------------------
#
# Graders flagged "7,roughly 180 locomotives" and "The Boeing roughly 790" as
# unjudgeable. A borderline-case generator that emits garbled strings produces
# rows nobody can grade, which is worse than producing none.


def test_lossy_rounding_keeps_comma_groups_intact():
    import random
    from cfbench.engines.lossy import _round_number

    result = _round_number("a roster of 7,175 locomotives", random.Random(0))
    assert result is not None
    out, _why = result
    import re
    assert not re.search(r"\d,roughly", out), out
    assert "roughly 7,200" in out or "roughly 7,100" in out, out


def test_lossy_rounding_skips_model_numbers():
    import random
    from cfbench.engines.lossy import _round_number

    # "The Boeing 787" must not become "The Boeing roughly 790".
    assert _round_number("The Boeing 787 entered service", random.Random(0)) is None


def test_lossy_rounding_still_fires_on_plain_magnitudes():
    import random
    from cfbench.engines.lossy import _round_number

    result = _round_number("revenue of 412 million dollars", random.Random(0))
    assert result is not None
    assert "roughly 410" in result[0]
