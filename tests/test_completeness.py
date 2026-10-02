from cfbench.completeness import aggregate, completeness, most_missed
from cfbench.schema import AnswerRecord, FailureMode, Question


def test_a_two_part_question_half_answered_scores_half():
    q = Question(id="q1", text="Which company supplies power, and what is its capacity?",
                 failure_modes=(FailureMode.MULTI_HOP,), aspects=("Helion Grid", "240"))
    rec = AnswerRecord(question_id="q1", engine="e",
                       answer_text="The site has a capacity of 240 megawatt-hours.")
    sc = completeness(q, rec)
    assert sc.fraction == 0.5
    assert sc.missed == ("Helion Grid",)
    assert sc.covered == ("240",)
    assert (sc.n_aspects, sc.n_covered) == (2, 1)


def test_an_empty_answer_scores_zero():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("x",))
    assert completeness(q, AnswerRecord(question_id="q1", engine="e", answer_text="")).fraction == 0.0


def test_a_question_with_no_aspects_scores_zero_not_a_divide_by_zero():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,))
    sc = completeness(q, AnswerRecord(question_id="q1", engine="e", answer_text="anything"))
    assert sc.fraction == 0.0
    assert sc.n_aspects == 0


def test_matching_ignores_case_and_punctuation():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("Helion Grid",))
    rec = AnswerRecord(question_id="q1", engine="e", answer_text="helion grid, per the filing.")
    assert completeness(q, rec).fraction == 1.0


def test_a_numeric_aspect_does_not_match_a_longer_number():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.NUMERIC,), aspects=("240",))
    rec = AnswerRecord(question_id="q1", engine="e", answer_text="Capacity reached 1240 units.")
    assert completeness(q, rec).fraction == 0.0


def test_a_numeric_aspect_matches_a_whole_token_next_to_punctuation():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.NUMERIC,), aspects=("240",))
    rec = AnswerRecord(question_id="q1", engine="e", answer_text="Rated at 240, per the filing.")
    assert completeness(q, rec).fraction == 1.0


def test_aggregate_averages_fractions():
    assert aggregate([]) == 0.0

    q_full = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("a",))
    q_half = Question(id="q2", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("a", "b"))
    full = completeness(q_full, AnswerRecord(question_id="q1", engine="e", answer_text="a"))
    half = completeness(q_half, AnswerRecord(question_id="q2", engine="e", answer_text="a"))
    assert aggregate([full, half]) == 0.75


def test_most_missed_ranks_aspects_by_how_often_they_are_absent():
    q1 = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("alpha", "beta"))
    q2 = Question(id="q2", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("alpha",))
    pairs = [
        (q1, AnswerRecord(question_id="q1", engine="e", answer_text="beta only")),
        (q2, AnswerRecord(question_id="q2", engine="e", answer_text="nothing useful")),
    ]
    assert most_missed(pairs) == [("alpha", 2)]


def test_most_missed_respects_its_limit():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,),
                 aspects=("alpha", "beta", "gamma"))
    pairs = [(q, AnswerRecord(question_id="q1", engine="e", answer_text=""))]
    assert len(most_missed(pairs, limit=2)) == 2


def test_question_round_trips_aspects_through_a_dict():
    from cfbench.schema import question_from_dict

    q = question_from_dict({"id": "q1", "text": "t", "failure_modes": ["simple"],
                            "aspects": ["a", "b"]})
    assert q.aspects == ("a", "b")
    assert question_from_dict({"text": "t", "failure_modes": ["simple"]}).aspects == ()
