from cfbench.claims import decompose_answer, split_sentences, strip_markers
from cfbench.schema import Citation


def test_splits_plain_sentences():
    assert split_sentences("One thing. Two things. Three.") == [
        "One thing.", "Two things.", "Three.",
    ]


def test_does_not_split_decimals():
    # The failure that matters: splitting here would invent a second claim.
    out = split_sentences("Revenue was 3.5 million dollars. Margin held.")
    assert out == ["Revenue was 3.5 million dollars.", "Margin held."]


def test_does_not_split_abbreviations():
    assert split_sentences("The U.S. market grew. Then it fell.") == [
        "The U.S. market grew.", "Then it fell.",
    ]
    assert len(split_sentences("Dr. Smith signed it.")) == 1


def test_strip_markers_tidies_spacing():
    assert strip_markers("Revenue grew [1] .") == "Revenue grew."
    assert strip_markers("A [1][2] and B [3].") == "A and B."


def test_decompose_attaches_citations_by_index():
    cits = [
        Citation(url="u1", snippet="s1"),
        Citation(url="u2", snippet="s2"),
    ]
    claims = decompose_answer("q1", "First [1]. Second [2].", cits)

    assert [c.text for c in claims] == ["First.", "Second."]
    assert claims[0].citation_ids == [cits[0].id]
    assert claims[1].citation_ids == [cits[1].id]


def test_decompose_handles_multi_and_comma_markers():
    cits = [Citation(url=f"u{i}", snippet=f"s{i}") for i in range(3)]
    claims = decompose_answer("q1", "Both [1][3]. Comma form [1, 2].", cits)

    assert claims[0].citation_ids == [cits[0].id, cits[2].id]
    assert claims[1].citation_ids == [cits[0].id, cits[1].id]


def test_out_of_range_marker_becomes_uncited_not_crash():
    # A hallucinated citation index is a real engine failure; it must surface
    # as an uncited claim rather than raising.
    cits = [Citation(url="u1", snippet="s1")]
    claims = decompose_answer("q1", "Claims too much [7].", cits)

    assert len(claims) == 1
    assert claims[0].is_uncited


def test_uncited_sentence_is_flagged():
    claims = decompose_answer("q1", "No citation here.", [])
    assert claims[0].is_uncited


def test_empty_answer_yields_no_claims():
    assert decompose_answer("q1", "   ", []) == []
