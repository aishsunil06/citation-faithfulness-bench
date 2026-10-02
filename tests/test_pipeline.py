"""End-to-end: retrieval -> engine -> claims -> judge -> score -> report.

Runs entirely offline, so CI needs no API key.
"""

from cfbench.engines import ExtractiveEngine, MockEngine
from cfbench.judge import LexicalJudge
from cfbench.labeling import label_session, pending_claims
from cfbench.metrics import agreement, score
from cfbench.report import build_report, disagreements
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
