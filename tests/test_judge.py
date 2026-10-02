from cfbench.judge import LexicalJudge, LLMJudge, get_judge
from cfbench.schema import Claim, Verdict


def claim(text, cited=True):
    return Claim(question_id="q1", text=text, citation_ids=["c1"] if cited else [])


def test_uncited_claim_is_uncited_not_unsupported():
    # These must stay distinct: "cited the wrong thing" and "cited nothing"
    # are different engine failures.
    label = LexicalJudge().judge(claim("Anything.", cited=False), [])
    assert label.verdict is Verdict.UNCITED


def test_high_overlap_is_supported():
    label = LexicalJudge().judge(
        claim("Northwind revenue was 412 million dollars"),
        ["Northwind Logistics reported revenue of 412 million dollars in 2024."],
    )
    assert label.verdict is Verdict.SUPPORTED


def test_swapped_number_on_the_same_fact_is_contradicted():
    # Every word matches except the figure. The source addresses this exact
    # point and states something else, which is contradiction rather than an
    # irrelevant citation.
    label = LexicalJudge().judge(
        claim("Northwind reported revenue of 500 million dollars"),
        ["Northwind reported revenue of 412 million dollars."],
    )
    assert label.verdict is Verdict.CONTRADICTED
    assert "500" in label.rationale


def test_number_absent_from_an_unrelated_source_is_unsupported():
    # Low overlap means the source is not about this fact at all, so there is
    # nothing to contradict.
    label = LexicalJudge().judge(
        claim("Northwind reported revenue of 500 million dollars"),
        ["Quarterly aviation charter bookings rose across regional hubs."],
    )
    assert label.verdict is Verdict.UNSUPPORTED


def test_contradiction_needs_a_number_in_the_source():
    # A source with no figures at all cannot be said to state a different one.
    label = LexicalJudge().judge(
        claim("Northwind reported revenue of 500 million dollars"),
        ["Northwind reported revenue growth driven by cold-chain contracts."],
    )
    assert label.verdict is Verdict.UNSUPPORTED


def test_number_formats_normalise():
    label = LexicalJudge().judge(
        claim("Revenue grew 12% year over year"),
        ["Revenue grew 12 percent year over year."],
    )
    assert label.verdict is Verdict.SUPPORTED


def test_negation_mismatch_downgrades_to_partial():
    label = LexicalJudge().judge(
        claim("The review found thermal runaway incidents at the site"),
        ["The review found no thermal runaway incidents at the site."],
    )
    assert label.verdict is Verdict.PARTIAL
    assert "negation" in label.rationale


def test_unrelated_snippet_is_unsupported():
    label = LexicalJudge().judge(
        claim("Helion Grid operates battery storage in Mesa County"),
        ["Quarterly aviation charter bookings rose across three regional hubs."],
    )
    assert label.verdict is Verdict.UNSUPPORTED


def test_judge_picks_best_of_several_snippets():
    label = LexicalJudge().judge(
        claim("The Mesa site has a capacity of 240 megawatt-hours"),
        [
            "Unrelated text about charter aircraft fleets.",
            "The Mesa site has a rated capacity of 240 megawatt-hours.",
        ],
    )
    assert label.verdict is Verdict.SUPPORTED


def test_malformed_llm_response_is_recorded_not_dropped():
    # Keeps the sample size honest: a broken judge call becomes a visible row,
    # not a silently missing one.
    label = LLMJudge(model="m")._parse(claim("x"), "not json at all")
    assert label.verdict is Verdict.PARTIAL
    assert "unparseable" in label.rationale


def test_llm_response_parses_verdict_and_rationale():
    raw = '{"verdict": "unsupported", "rationale": "date mismatch"}'
    label = LLMJudge(model="m")._parse(claim("x"), raw)
    assert label.verdict is Verdict.UNSUPPORTED
    assert label.rationale == "date mismatch"


def test_get_judge_resolves_names():
    assert isinstance(get_judge("lexical"), LexicalJudge)
    assert get_judge("llm:gpt-4o").model == "gpt-4o"
    try:
        get_judge("nope")
    except ValueError as exc:
        assert "unknown judge" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_contradicted_counts_as_a_failure_but_not_as_credit():
    assert Verdict.CONTRADICTED.is_failure
    assert not Verdict.CONTRADICTED.is_credit
    assert Verdict.UNSUPPORTED.is_failure
    assert not Verdict.PARTIAL.is_failure       # weak, but not outright wrong
    assert Verdict.SUPPORTED.is_credit


def test_llm_judge_parses_the_contradicted_verdict():
    raw = '{"verdict": "contradicted", "rationale": "source says 412 not 500"}'
    label = LLMJudge(model="m")._parse(claim("x"), raw)
    assert label.verdict is Verdict.CONTRADICTED


def test_llm_prompt_documents_every_verdict_it_may_return():
    from cfbench.judge import JUDGE_SYSTEM_PROMPT

    for v in (Verdict.SUPPORTED, Verdict.PARTIAL, Verdict.UNSUPPORTED,
              Verdict.CONTRADICTED):
        assert f'"{v.value}"' in JUDGE_SYSTEM_PROMPT
