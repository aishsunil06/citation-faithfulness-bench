"""End-to-end: retrieval -> engine -> claims -> judge -> score -> report.

Runs entirely offline, so CI needs no API key.
"""

from cfbench.engines import ExtractiveEngine, MockEngine
from cfbench.judge import LexicalJudge
from cfbench.labeling import label_session, pending_claims
from cfbench.metrics import agreement, score
from cfbench.report import build_report, disagreements
from cfbench.claims import split_sentences
from cfbench.retrieval import Document, build_index, chunk_document
from cfbench.schema import (
    FailureMode,
    Label,
    Question,
    Verdict,
    answer_from_dict,
    read_jsonl,
    write_jsonl,
)

DOCS = [
    Document(
        url="https://example.test/northwind",
        title="Northwind FY2024",
        text=(
            "Northwind Logistics reported revenue of 412 million dollars in "
            "fiscal 2024. The company operated 47 distribution centres."
        ),
    ),
    Document(
        url="https://example.test/helion",
        title="Helion Grid",
        text=(
            "Helion Grid operates grid-scale battery storage. The Mesa site "
            "has a rated capacity of 240 megawatt-hours."
        ),
    ),
]

Q_REVENUE = Question(
    id="q1",
    text="What was Northwind Logistics' revenue in fiscal 2024?",
    failure_modes=(FailureMode.NUMERIC,),
)
Q_CAPACITY = Question(
    id="q2",
    text="What is the capacity of Helion Grid's Mesa site?",
    failure_modes=(FailureMode.SIMPLE,),
)


def test_chunking_respects_overlap_and_width():
    doc = Document(url="u", text=" ".join(str(i) for i in range(100)))
    chunks = chunk_document(doc, 0, words_per_chunk=40, overlap_words=10)

    assert all(len(c.text.split()) <= 40 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[30:40] == second[0:10]      # the overlap is real


def test_bm25_retrieves_the_right_document():
    index = build_index(DOCS, words_per_chunk=50, overlap_words=10)
    hits = index.search("Mesa site capacity megawatt-hours", top_k=1)

    assert hits and "Mesa" in hits[0][0].text


def test_bm25_empty_query_returns_nothing():
    index = build_index(DOCS)
    assert index.search("the and of", top_k=3) == []


def test_extractive_engine_cites_what_it_quotes():
    index = build_index(DOCS, words_per_chunk=50, overlap_words=10)
    record = ExtractiveEngine(index=index, top_k=2).answer(Q_REVENUE)

    assert record.claims, "engine produced no claims"
    assert all(not c.is_uncited for c in record.claims)
    # Every claim resolves to a citation that exists on the record.
    for claim in record.claims:
        assert record.snippets_for(claim)


def test_extractive_engine_scores_well_under_the_lexical_judge():
    # The extractive engine copies from its source, so it should mostly pass.
    index = build_index(DOCS, words_per_chunk=50, overlap_words=10)
    engine = ExtractiveEngine(index=index, top_k=1)
    judge = LexicalJudge()

    labels = []
    for q in (Q_REVENUE, Q_CAPACITY):
        record = engine.answer(q)
        for claim in record.claims:
            labels.append(judge.judge(claim, record.snippets_for(claim)))

    assert score(labels).strict >= 0.5


def test_scorer_separates_a_good_engine_from_a_fabricating_one():
    good = MockEngine(
        name="good",
        script={
            "q1": (
                "Northwind Logistics reported revenue of 412 million dollars [1].",
                [("u", "Northwind Logistics reported revenue of 412 million dollars.")],
            )
        },
    )
    liar = MockEngine(
        name="liar",
        script={
            "q1": (
                "Northwind Logistics reported revenue of 900 million dollars [1].",
                [("u", "Northwind Logistics reported revenue of 412 million dollars.")],
            )
        },
    )
    judge = LexicalJudge()

    def run(engine):
        rec = engine.answer(Q_REVENUE)
        return score([judge.judge(c, rec.snippets_for(c)) for c in rec.claims])

    assert run(good).strict == 1.0
    assert run(liar).strict == 0.0


def test_uncited_engine_is_penalised_as_uncited():
    silent = MockEngine(
        name="silent", script={"q1": ("Revenue was 412 million dollars.", [])}
    )
    rec = silent.answer(Q_REVENUE)
    judge = LexicalJudge()
    sc = score([judge.judge(c, rec.snippets_for(c)) for c in rec.claims])

    assert sc.uncited_rate == 1.0
    assert sc.strict == 0.0


def test_report_renders_and_warns_without_human_labels():
    index = build_index(DOCS, words_per_chunk=50, overlap_words=10)
    engine = ExtractiveEngine(index=index, top_k=1)
    records = [engine.answer(Q_REVENUE), engine.answer(Q_CAPACITY)]
    judge = LexicalJudge()
    labels = [
        judge.judge(c, r.snippets_for(c)) for r in records for c in r.claims
    ]

    text = build_report([Q_REVENUE, Q_CAPACITY], records, labels)

    assert "Faithfulness leaderboard" in text
    assert "No human labels yet" in text


def test_report_includes_calibration_once_humans_label():
    index = build_index(DOCS, words_per_chunk=50, overlap_words=10)
    engine = ExtractiveEngine(index=index, top_k=1)
    records = [engine.answer(Q_REVENUE)]
    judge = LexicalJudge()
    judge_labels = [
        judge.judge(c, records[0].snippets_for(c)) for c in records[0].claims
    ]
    human_labels = [
        Label(claim_id=c.id, verdict=Verdict.SUPPORTED, labeler="human:t")
        for c in records[0].claims
    ]

    text = build_report([Q_REVENUE], records, judge_labels + human_labels)

    assert "cohen's kappa" in text
    assert "No human labels yet" not in text


def test_disagreements_surface_only_mismatches():
    records = [
        MockEngine(script={"q1": ("A claim here [1].", [("u", "Unrelated text.")])})
        .answer(Q_REVENUE)
    ]
    claim_id = records[0].claims[0].id
    labels = [
        Label(claim_id=claim_id, verdict=Verdict.SUPPORTED, labeler="human:t"),
        Label(claim_id=claim_id, verdict=Verdict.UNSUPPORTED, labeler="judge:lexical"),
    ]

    rows = disagreements(labels, records, "judge:lexical")

    assert len(rows) == 1
    assert rows[0]["human"] == "supported"
    assert rows[0]["judge"] == "unsupported"


def test_answer_records_survive_a_jsonl_round_trip(tmp_path):
    index = build_index(DOCS, words_per_chunk=50, overlap_words=10)
    record = ExtractiveEngine(index=index, top_k=2).answer(Q_REVENUE)
    path = tmp_path / "answers.jsonl"

    write_jsonl(path, [record])
    restored = [answer_from_dict(r) for r in read_jsonl(path)]

    assert len(restored) == 1
    assert restored[0].answer_text == record.answer_text
    assert [c.id for c in restored[0].claims] == [c.id for c in record.claims]
    assert restored[0].snippets_for(restored[0].claims[0])


def test_labelling_is_resumable(tmp_path):
    index = build_index(DOCS, words_per_chunk=50, overlap_words=10)
    record = ExtractiveEngine(index=index, top_k=2).answer(Q_REVENUE)
    path = tmp_path / "labels.jsonl"

    # Label one claim, then quit.
    replies = iter(["s", "looks right", "q"])
    n = label_session(
        [Q_REVENUE], [record], path, "human:t",
        input_fn=lambda _: next(replies), print_fn=lambda *a, **k: None,
    )
    assert n == 1

    from cfbench.labeling import load_labels
    remaining = pending_claims([record], load_labels(path), "human:t")
    assert len(remaining) == len(record.claims) - 1


def test_labelling_rejects_unknown_keys_then_accepts(tmp_path):
    record = MockEngine(script={"q1": ("One claim [1].", [("u", "One claim.")])}).answer(
        Q_REVENUE
    )
    replies = iter(["x", "s", ""])
    n = label_session(
        [Q_REVENUE], [record], tmp_path / "l.jsonl", "human:t",
        input_fn=lambda _: next(replies), print_fn=lambda *a, **k: None,
    )
    assert n == 1


# --------------------------------------------------------------------------
# Sentence-aware chunking
# --------------------------------------------------------------------------
#
# Found by graders working the real corpus: chunks cut at an exact word count
# produced claims like "with Intel's at," which assert nothing, and chunks that
# began mid-sentence dropped the subject clause so a snippet could fail to name
# the entity its own sentence was about. Either way the row becomes
# unjudgeable, which is worse than having no row.


SENTENCES = (
    "The first plant opened in 1978 with a rated output of 290 megawatts. "
    "A second site followed in 1991 and reached 110 megawatts. "
    "Neither facility recorded a safety incident during the review period. "
    "The regulator confirmed both remained in compliance throughout. "
    "A third expansion was proposed but never entered construction. "
)


def _chunks(text, width=30, overlap=8):
    return chunk_document(Document(url="u", text=text), 0, width, overlap)


def test_chunks_end_on_sentence_boundaries():
    for chunk in _chunks(SENTENCES * 3):
        assert chunk.text.rstrip().endswith("."), chunk.text


def test_chunks_start_on_sentence_boundaries():
    for chunk in _chunks(SENTENCES * 3):
        first = chunk.text.lstrip()[0]
        assert first.isupper() or first.isdigit(), chunk.text[:60]


def test_no_chunk_ends_in_a_dangling_fragment():
    # The concrete failure: a trailing clause with no predicate.
    for chunk in _chunks(SENTENCES * 3):
        assert not chunk.text.rstrip().endswith(","), chunk.text


def test_the_word_budget_stays_a_ceiling():
    # Widths must remain comparable across a sweep, since the sweep is the
    # headline comparison. A sentence that would overflow starts the next
    # chunk rather than overshooting this one.
    for width in (20, 30, 60):
        for chunk in _chunks(SENTENCES * 3, width=width, overlap=width // 4):
            assert len(chunk.text.split()) <= width, (width, chunk.text)


def test_overlap_is_never_silently_zero_for_long_sentences():
    # The bug this pins: a purely budget-driven carry-back produced NO overlap
    # whenever sentences were longer than overlap_words, which is the common
    # case for prose.
    chunks = _chunks(SENTENCES * 3, width=30, overlap=2)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        assert set(split_sentences(a.text)) & set(split_sentences(b.text))


def test_overlap_carries_whole_sentences_between_chunks():
    chunks = _chunks(SENTENCES * 3)
    assert len(chunks) >= 2
    # Some sentence of chunk N reappears in chunk N+1.
    for a, b in zip(chunks, chunks[1:]):
        shared = set(split_sentences(a.text)) & set(split_sentences(b.text))
        assert shared, (a.text[-60:], b.text[:60])


def test_text_without_sentence_boundaries_falls_back_to_word_windows():
    # A table or list has nothing to pack on; it must still chunk.
    doc = Document(url="u", text=" ".join(str(i) for i in range(100)))
    chunks = chunk_document(doc, 0, words_per_chunk=40, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 40 for c in chunks)


def test_a_single_sentence_longer_than_the_budget_still_chunks():
    long_sentence = "word " * 200 + "end."
    chunks = chunk_document(
        Document(url="u", text=long_sentence), 0, words_per_chunk=40, overlap_words=10
    )
    assert len(chunks) > 1


def test_engine_claims_are_whole_sentences_after_chunking(tmp_path):
    # End to end: the engine's claims should no longer be fragments.
    index = build_index(
        [Document(url="u", title="T", text=SENTENCES * 4)],
        words_per_chunk=40,
        overlap_words=10,
    )
    record = ExtractiveEngine(index=index, top_k=2).answer(
        Question(id="q1", text="What output did the first plant have?",
                 failure_modes=(FailureMode.NUMERIC,))
    )
    assert record.claims
    for claim in record.claims:
        assert not claim.text.rstrip(".").endswith(","), claim.text
