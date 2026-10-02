"""Dataset validation, especially answerability.

An unanswerable question is the most damaging dataset bug this project can
have: no engine can cite a source that does not exist, so every engine looks
unfaithful and the blame lands on the engines rather than the data. These
tests pin the check that catches it.
"""

from cfbench.cli import answerability
from cfbench.retrieval import Document, build_index
from cfbench.schema import FailureMode, Question


def q(text, qid="q1", modes=(FailureMode.SIMPLE,)):
    return Question(id=qid, text=text, failure_modes=modes)


def test_answerability_flags_a_question_the_corpus_cannot_answer():
    index = build_index(
        [Document(url="u", text="Battery storage capacity in Mesa County. " * 10)]
    )
    assert answerability([q("What is the capital of Peru?")], index) == ["q1"]


def test_answerability_passes_a_question_the_corpus_covers():
    index = build_index([
        Document(
            url="u",
            text="The Mesa site has a rated capacity of 240 megawatt-hours. " * 5,
        )
    ])
    assert answerability([q("What is the Mesa site rated capacity?")], index) == []


def test_answerability_flags_a_question_with_no_content_tokens():
    # "What is the of a?" is all stopwords; it cannot be matched against
    # anything and would otherwise divide by zero.
    index = build_index([Document(url="u", text="Real content about storage. " * 20)])
    assert answerability([q("Is it the one?")], index) == ["q1"]


def test_answerability_against_an_empty_corpus_flags_everything():
    index = build_index([])
    assert answerability([q("Anything at all about storage capacity?")], index) == ["q1"]


def test_min_overlap_is_adjustable():
    index = build_index([
        Document(url="u", text="Storage capacity figures for the Mesa site. " * 10)
    ])
    question = q("What storage capacity did unrelated aviation charters report?")

    # Lenient threshold admits a partial keyword match; strict rejects it.
    lenient = answerability([question], index, min_overlap=0.1)
    strict = answerability([question], index, min_overlap=0.95)
    assert lenient == []
    assert strict == ["q1"]


def test_answerability_reports_every_failing_question_not_just_the_first():
    index = build_index([Document(url="u", text="Storage capacity in Mesa. " * 20)])
    bad = [
        q("What is the capital of Peru?", qid="qa"),
        q("Who won the 1998 world cup final?", qid="qb"),
    ]
    assert answerability(bad, index) == ["qa", "qb"]
