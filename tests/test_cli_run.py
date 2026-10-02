"""The `run` subcommand end to end, offline.

Regression cover for a counting bug that made the console output lie: the
per-variant "faults injected" figure was summed over the accumulated oracle
list rather than the current variant, so a two-chunk-width sweep reported more
injections than there were claims. The numbers in the JSONL were right, which
is what makes this kind of bug survive -- only the operator-facing count was
wrong, and it was wrong in the direction that looks like success.
"""

from collections import Counter

from cfbench.cli import main
from cfbench.labeling import load_labels
from cfbench.schema import answer_from_dict, read_jsonl, write_jsonl

CORPUS = [
    {
        "url": "https://example.test/a",
        "title": "Alpha",
        "text": (
            "The Alpha plant reported output of 412 gigawatt hours in the "
            "reporting year. It employs 180 staff across two shifts and has "
            "operated without interruption since commissioning. The site did "
            "not record any safety incidents during the period. "
        ) * 4,
    },
    {
        "url": "https://example.test/b",
        "title": "Beta",
        "text": (
            "The Beta facility reported output of 118 gigawatt hours over the "
            "same period. It employs 240 staff and runs a single shift. "
            "Maintenance was deferred twice during the year. "
        ) * 4,
    },
]

QUESTIONS = [
    {
        "id": "q1",
        "text": "What output did the Alpha plant report?",
        "failure_modes": ["numeric", "simple"],
        "aspects": ["412"],
    },
    {
        "id": "q2",
        "text": "How many staff does the Beta facility employ?",
        "failure_modes": ["numeric"],
        "aspects": ["240"],
    },
]


def _fixture(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    questions = tmp_path / "questions.jsonl"
    write_jsonl(corpus, CORPUS)
    write_jsonl(questions, QUESTIONS)
    return corpus, questions


def test_run_writes_answers_for_every_question(tmp_path):
    corpus, questions = _fixture(tmp_path)
    answers = tmp_path / "answers.jsonl"

    code = main([
        "run", "--engine", "extractive",
        "--corpus", str(corpus), "--questions", str(questions),
        "--out", str(answers), "--labels", str(tmp_path / "labels.jsonl"),
        "--chunk-words", "60",
    ])

    assert code == 0
    records = [answer_from_dict(r) for r in read_jsonl(answers)]
    assert {r.question_id for r in records} == {"q1", "q2"}
    assert all(r.claims for r in records)


def test_a_chunk_sweep_names_each_width_as_a_separate_engine(tmp_path):
    # Chunk geometry is part of engine identity; collapsing widths into one
    # name would make the sweep unanalysable.
    corpus, questions = _fixture(tmp_path)
    answers = tmp_path / "answers.jsonl"

    main([
        "run", "--engine", "extractive",
        "--corpus", str(corpus), "--questions", str(questions),
        "--out", str(answers), "--labels", str(tmp_path / "labels.jsonl"),
        "--chunk-words", "60", "--chunk-words", "120",
    ])

    engines = {r.engine for r in (answer_from_dict(x) for x in read_jsonl(answers))}
    assert engines == {"extractive-k3-c60", "extractive-k3-c120"}


def test_oracle_label_count_matches_injections_across_a_sweep(tmp_path):
    """The regression: counts must be per variant, not cumulative.

    Two chunk widths and one injector must produce oracle labels only for the
    two faulty variants, and never more labels than there are claims in them.
    """
    corpus, questions = _fixture(tmp_path)
    answers = tmp_path / "answers.jsonl"
    labels = tmp_path / "labels.jsonl"

    main([
        "run", "--engine", "extractive",
        "--corpus", str(corpus), "--questions", str(questions),
        "--out", str(answers), "--labels", str(labels),
        "--chunk-words", "60", "--chunk-words", "120",
        "--fault", "numeric-drift",
    ])

    records = [answer_from_dict(r) for r in read_jsonl(answers)]
    claims_by_engine = Counter(
        {r.engine: len(r.claims) for r in records}
    )
    for record in records:
        claims_by_engine[record.engine] = sum(
            len(x.claims) for x in records if x.engine == record.engine
        )

    oracle = [lb for lb in load_labels(labels) if lb.labeler.startswith("oracle:")]
    per_engine_oracle: Counter[str] = Counter()
    claim_to_engine = {c.id: r.engine for r in records for c in r.claims}
    for lb in oracle:
        per_engine_oracle[claim_to_engine[lb.claim_id]] += 1

    # Only the faulty variants carry oracle labels.
    assert all("+numeric-drift" in e for e in per_engine_oracle)
    # And never more than that variant actually produced.
    for engine, count in per_engine_oracle.items():
        assert count <= claims_by_engine[engine], (
            f"{engine}: {count} oracle labels over {claims_by_engine[engine]} claims"
        )


def test_append_preserves_earlier_records(tmp_path):
    corpus, questions = _fixture(tmp_path)
    answers = tmp_path / "answers.jsonl"
    labels = tmp_path / "labels.jsonl"
    common = [
        "run", "--engine", "extractive",
        "--corpus", str(corpus), "--questions", str(questions),
        "--out", str(answers), "--labels", str(labels),
    ]

    main(common + ["--chunk-words", "60"])
    first = len(list(read_jsonl(answers)))
    main(common + ["--chunk-words", "120", "--append"])
    second = len(list(read_jsonl(answers)))

    assert second == first * 2


def test_run_without_append_replaces_earlier_records(tmp_path):
    corpus, questions = _fixture(tmp_path)
    answers = tmp_path / "answers.jsonl"
    labels = tmp_path / "labels.jsonl"
    common = [
        "run", "--engine", "extractive",
        "--corpus", str(corpus), "--questions", str(questions),
        "--out", str(answers), "--labels", str(labels),
    ]

    main(common + ["--chunk-words", "60"])
    first = len(list(read_jsonl(answers)))
    main(common + ["--chunk-words", "120"])

    assert len(list(read_jsonl(answers))) == first
