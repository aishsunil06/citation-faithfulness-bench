# Citation Faithfulness Benchmark: Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the benchmark from a validated instrument on 10 synthetic documents to a real result on a real corpus, with a second measurement axis, relying on a synthetic oracle rather than human annotation.

**Architecture:** Keep the existing pipeline (retrieval -> engine -> claims -> judge -> metrics -> report) and extend it at four points: a corpus ingester that pulls real public documents, a scaled question set authored against that corpus, a broadened fault-injection oracle that is now the sole ground truth, and a second scoring axis for answer completeness.

**Tech Stack:** Python 3.13, standard library only for the core. `pytest` for tests. Corpus text is captured by the operator and ingested from a raw JSONL file, so the library itself gains no network dependency.

**Spec:** This document. Derived from the project README's "Current status and known limitations", each limitation becoming a task.

## Global Constraints

- Core modules (`retrieval`, `claims`, `judge.LexicalJudge`, `metrics`, `report`, `labeling`, `completeness`) stay dependency-free standard library. New third-party imports are confined to optional engines/judges and must degrade with a clear message.
- **No human annotation is required or collected.** Ground truth comes from fault injection. The 25 existing `human:aishwarya` labels are preserved as an auxiliary spot-check but are not the project's validation mechanism and no further human labelling is requested.
- Model labels use a `judge:<model>` labeler. A model must never write a `human:` label.
- Fault injection may only produce cases whose correct verdict is indisputable. Bounds, years, and scope qualifiers are out of bounds.
- Claim and citation ids stay content-addressed so labels survive re-runs.
- Every number in the README must be reproducible by a documented command.
- No `git push` until the whole plan is complete and all tests pass.

## Review Focus

1. **Ingested documents carrying boilerplate or navigation text** — real pages carry menus and footers; a chunk of navigation links retrieved as evidence would silently corrupt every score. Ingestion must strip it and the test must prove it on captured real text. *Covered by Task 1 Step 1.*
2. **A question whose answer is absent from the corpus** — authoring questions separately from documents makes this easy to do by accident, and an unanswerable question makes every engine look unfaithful. `validate` must detect and fail. *Covered by Task 2 Step 1.*
3. **Duplicate or near-duplicate documents** — fetching related pages yields overlapping text, letting an engine cite a different-but-identical source and inflating retrieval. Ingestion must dedupe. *Covered by Task 1 Step 1.*
4. **An empty answer or a question with no declared aspects** — completeness divides by aspect count; both must score 0.0 rather than raising. *Covered by Task 3 Step 1.*
5. **A fault injector that fires on a claim it cannot corrupt indisputably** — the oracle is now the only ground truth, so a single wrong oracle verdict silently corrupts the headline result. Every new injector needs a test proving it declines ambiguous input. *Covered by Task 4 Step 1.*

---

### Task 1: Real corpus ingestion

**Files:**
- Create: `cfbench/ingest.py`
- Modify: `cfbench/cli.py` (add `ingest` subcommand)
- Create: `data/corpus.real.jsonl`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `retrieval.Document`, `retrieval.tokenize`.
- Produces: `clean_text(raw: str) -> str`; `to_document(url: str, title: str, raw: str) -> Document`; `dedupe(docs: Sequence[Document], threshold: float = 0.8) -> tuple[list[Document], list[str]]` returning kept docs and dropped urls; `ingest_rows(rows: Iterable[dict]) -> tuple[list[Document], list[str]]`.

- [ ] **Step 1: Write failing tests in `tests/test_ingest.py`**

```python
def test_clean_text_strips_navigation_and_reference_markers():
    raw = "Jump to content\nMain menu\n\nRevenue rose 12 percent in 2024.[3]\n\nSee also\nReferences\nExternal links"
    out = clean_text(raw)
    assert "Revenue rose 12 percent in 2024." in out
    assert "Jump to content" not in out
    assert "External links" not in out
    assert "[3]" not in out

def test_clean_text_drops_short_lines_and_collapses_whitespace():
    out = clean_text("OK\n\n\nA real sentence that is certainly long enough to keep.\n\nEdit\n")
    assert out == "A real sentence that is certainly long enough to keep."

def test_dedupe_drops_near_identical_documents_keeping_the_first():
    a = Document(url="u1", text="shared body text about storage capacity " * 20)
    b = Document(url="u2", text="shared body text about storage capacity " * 20 + "tail")
    kept, dropped = dedupe([a, b])
    assert [d.url for d in kept] == ["u1"]
    assert dropped == ["u2"]

def test_dedupe_keeps_genuinely_different_documents():
    a = Document(url="u1", text="battery storage capacity megawatt hours " * 20)
    b = Document(url="u2", text="charter aviation fleet regional hubs aircraft " * 20)
    kept, _dropped = dedupe([a, b])
    assert len(kept) == 2

def test_to_document_rejects_a_page_too_short_to_chunk():
    with pytest.raises(ValueError):
        to_document("u", "t", "too short")
```

- [ ] **Step 2: Run tests, verify they fail with ImportError**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL, no module named `cfbench.ingest`

- [ ] **Step 3: Implement `cfbench/ingest.py`**

`clean_text` drops lines under 25 characters, drops lines matching a boilerplate blocklist (`Jump to content`, `Main menu`, `Navigation`, `See also`, `References`, `External links`, `Further reading`, `Retrieved from`, `Categories:`, `Edit`, `Contents`), strips bracketed reference markers matching `\[\d+\]` and `\[citation needed\]`, and collapses runs of whitespace to single spaces. `dedupe` compares Jaccard similarity over `retrieval.tokenize` token sets, keeping the first of any pair at or above `threshold`. `to_document` raises `ValueError` when the cleaned text is under 60 words.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Add the `ingest` CLI subcommand**

`python -m cfbench ingest --input <raw.jsonl> --out data/corpus.real.jsonl` reads `{url, title, raw}` rows, cleans, dedupes, writes `{url, title, text}`, and prints kept/dropped counts plus the total word count.

- [ ] **Step 6: Capture and ingest at least 30 real public documents into `data/corpus.real.jsonl`**

Source: English Wikipedia article prose, selected for factual density (figures, dates, named entities) across at least three domains so questions stratify. Every source URL recorded verbatim in the `url` field.

- [ ] **Step 7: Verify the corpus**

Run: `python -m cfbench validate --corpus data/corpus.real.jsonl --questions data/questions.jsonl`
Expected: no duplicate-url problems and no too-short problems.

- [ ] **Step 8: Commit**

```bash
git add cfbench/ingest.py cfbench/cli.py tests/test_ingest.py data/corpus.real.jsonl
git commit -m "feat: ingest a real document corpus"
```

### Task 2: Scaled, stratified question set

**Files:**
- Create: `data/questions.real.jsonl`
- Modify: `cfbench/cli.py` (`cmd_validate` gains an answerability check)
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `schema.Question`, `retrieval.build_index`, Task 1's corpus.
- Produces: `answerability(questions: Sequence[Question], index: BM25Index, min_overlap: float = 0.25) -> list[str]` returning ids of questions with no plausibly supporting chunk.

- [ ] **Step 1: Write failing tests in `tests/test_validate.py`**

```python
def test_answerability_flags_a_question_the_corpus_cannot_answer():
    index = build_index([Document(url="u", text="Battery storage capacity in Mesa County. " * 10)])
    q = Question(id="q1", text="What is the capital of Peru?", failure_modes=(FailureMode.SIMPLE,))
    assert answerability([q], index) == ["q1"]

def test_answerability_passes_a_question_the_corpus_covers():
    index = build_index([Document(url="u", text="The Mesa site has a rated capacity of 240 megawatt-hours. " * 5)])
    q = Question(id="q1", text="What is the Mesa site rated capacity?", failure_modes=(FailureMode.SIMPLE,))
    assert answerability([q], index) == []

def test_validate_reports_an_unanswerable_question_as_a_problem(capsys, tmp_path):
    # Build a corpus and a question file on disk, run cmd_validate, assert exit 1
    # and the question id in stderr.
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement `answerability` in `cfbench/cli.py` and call it from `cmd_validate`**

Overlap is the fraction of the question's content tokens present in the best-matching chunk, using `retrieval.tokenize`. Report each failing question id as a problem so `validate` exits non-zero.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Author at least 90 questions in `data/questions.real.jsonl`**

Every `FailureMode` carries at least 12 questions. Each question's `notes` records the expected answer and the trap, matching the seed set's style. Questions are authored against specific corpus documents so answerability holds.

- [ ] **Step 6: Verify**

Run: `python -m cfbench validate --questions data/questions.real.jsonl --corpus data/corpus.real.jsonl`
Expected: `no problems found`, every mode at 12 or more.

- [ ] **Step 7: Commit**

### Task 3: Answer-completeness axis

**Files:**
- Create: `cfbench/completeness.py`
- Modify: `cfbench/schema.py` (add `Question.aspects: tuple[str, ...] = ()`), `cfbench/report.py` (new section), `cfbench/cli.py` (`score-completeness` subcommand)
- Test: `tests/test_completeness.py`

**Interfaces:**
- Consumes: `schema.Question`, `schema.AnswerRecord`.
- Produces: `CompletenessScore` dataclass with `n_aspects: int`, `n_covered: int`, `covered: tuple[str, ...]`, `missed: tuple[str, ...]` and property `fraction: float`; `completeness(question: Question, record: AnswerRecord) -> CompletenessScore`; `aggregate(scores: Iterable[CompletenessScore]) -> float`; `most_missed(pairs: Iterable[tuple[Question, AnswerRecord]], limit: int = 10) -> list[tuple[str, int]]`.

Rationale: a faithfulness score cannot see an engine that cites every sentence perfectly and answers the wrong question. `aspects` lists what a complete answer must mention. Coverage is keyword presence, which is crude, and the README must say so.

- [ ] **Step 1: Write failing tests in `tests/test_completeness.py`**

```python
def test_a_two_part_question_half_answered_scores_half():
    q = Question(id="q1", text="Which company supplies power, and what is its capacity?",
                 failure_modes=(FailureMode.MULTI_HOP,), aspects=("Helion Grid", "240"))
    rec = AnswerRecord(question_id="q1", engine="e",
                       answer_text="The site has a capacity of 240 megawatt-hours.")
    sc = completeness(q, rec)
    assert sc.fraction == 0.5
    assert sc.missed == ("Helion Grid",)

def test_an_empty_answer_scores_zero():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("x",))
    assert completeness(q, AnswerRecord(question_id="q1", engine="e", answer_text="")).fraction == 0.0

def test_a_question_with_no_aspects_scores_zero_not_a_divide_by_zero():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,))
    assert completeness(q, AnswerRecord(question_id="q1", engine="e", answer_text="anything")).fraction == 0.0

def test_matching_ignores_case_and_punctuation():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.SIMPLE,), aspects=("Helion Grid",))
    rec = AnswerRecord(question_id="q1", engine="e", answer_text="helion grid, per the filing.")
    assert completeness(q, rec).fraction == 1.0

def test_a_numeric_aspect_does_not_match_a_longer_number():
    q = Question(id="q1", text="?", failure_modes=(FailureMode.NUMERIC,), aspects=("240",))
    rec = AnswerRecord(question_id="q1", engine="e", answer_text="Capacity reached 1240 units.")
    assert completeness(q, rec).fraction == 0.0

def test_aggregate_averages_fractions():
    assert aggregate([]) == 0.0
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement `cfbench/completeness.py` and add `aspects` to `Question`**

`aspects` defaults to `()`. Matching normalises case and strips punctuation. A purely numeric aspect matches on whole-token equality so `240` does not match `1240`; a textual aspect matches as a normalised substring.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Add a `## Answer completeness` report section and a `score-completeness` subcommand**

The section shows per-engine mean coverage and the ten most-missed aspects. It states in one line that coverage is keyword presence, not semantic equivalence.

- [ ] **Step 6: Populate `aspects` for every question in `data/questions.real.jsonl`**

- [ ] **Step 7: Commit**

### Task 4: Broaden the fault-injection oracle

**Files:**
- Modify: `cfbench/engines/faulty.py`
- Test: `tests/test_faults.py`

**Interfaces:**
- Consumes: `schema.AnswerRecord`, `schema.Verdict`, existing `_Base`.
- Produces: two new injectors registered in `FAULTS` — `EntitySwap` (fault name `entity-swap`) and `PolarityFlip` (fault name `polarity-flip`).

Rationale: with human annotation removed, the oracle is the only ground truth, so its coverage determines what the benchmark can measure. Three injectors cover numeric and attribution failures; entity substitution and polarity inversion are the two commonest remaining real citation failures whose correct verdict is still indisputable.

- [ ] **Step 1: Write failing tests**

```python
def test_entity_swap_replaces_a_named_entity_and_flags_contradicted():
    # Claim names an entity present in the cited snippet; swap it for a
    # different entity that also appears in the corpus. Source states the
    # fact about entity A, claim asserts it about entity B.
    ...
    assert result.oracle[0].verdict is Verdict.CONTRADICTED

def test_entity_swap_declines_when_no_alternative_entity_is_available():
    assert EntitySwap(engine_with_one_entity()).inject(Q).oracle == []

def test_polarity_flip_inverts_a_negation_and_flags_contradicted():
    # "found no incidents" -> "found incidents"
    assert result.oracle[0].verdict is Verdict.CONTRADICTED

def test_polarity_flip_declines_a_claim_with_no_polarity_marker():
    assert PolarityFlip(engine()).inject(Q).oracle == []

def test_polarity_flip_declines_a_hedged_claim():
    # "may not have" is modal; inverting it does not produce a clean
    # contradiction, so the injector must decline rather than guess.
    ...
    assert result.oracle == []
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement `EntitySwap` and `PolarityFlip` in `cfbench/engines/faulty.py`**

`EntitySwap` finds capitalised multi-word spans in the claim that also appear in the cited snippet, and substitutes one for a different such span drawn from a non-cited citation on the same record; it declines when no alternative exists. `PolarityFlip` toggles a single unhedged negation marker (`no`, `not`, `never`, `without`) and declines when the claim contains a modal (`may`, `might`, `could`, `appears`) or more than one marker.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Register both in `FAULTS` and confirm `wrap` resolves them**

- [ ] **Step 6: Commit**

### Task 5: Full run and model-judge labelling

**Files:**
- Modify: `runs/answers.jsonl`, `runs/labels.jsonl`, `runs/report.md`

- [ ] **Step 1: Run every engine over the real corpus and question set**

`extractive` and `lossy`, two chunk widths each, plus all five fault injectors.

Run: `python -m cfbench run --engine lossy --questions data/questions.real.jsonl --corpus data/corpus.real.jsonl --chunk-words 90 --chunk-words 250 --fault numeric-drift --fault wrong-source --fault dropped-citation --fault entity-swap --fault polarity-flip`

- [ ] **Step 2: Judge with `lexical`**

- [ ] **Step 3: Produce `judge:claude-opus-5` labels over the prioritised queue**

Dispatched in parallel batches; each batch reads claims against their cited snippets and returns verdicts. Attributed to the model.

- [ ] **Step 4: Verify no model wrote a human label**

Run: `python -c "from cfbench.labeling import load_labels; from collections import Counter; print(Counter(l.labeler for l in load_labels('runs/labels.jsonl')))"`
Expected: exactly one `human:` key, `human:aishwarya`, count 25.

- [ ] **Step 5: Commit the run artifacts**

### Task 6: Analysis and documentation

**Files:**
- Modify: `README.md`
- Create: `docs/FINDINGS.md`

- [ ] **Step 1: Regenerate the report and confirm every section populates**

Run: `python -m cfbench report --out runs/report.md`

- [ ] **Step 2: Write `docs/FINDINGS.md`**

Leaderboard, fault-detection rates per injector per judge, completeness, failure-mode decomposition. Every number carries a confidence interval and the exact command that reproduces it.

- [ ] **Step 3: Rewrite the README's findings and limitations against what is now true**

The limitations must state plainly that validation rests on a synthetic oracle, that no human-agreement figure exists, and that oracle coverage bounds what the benchmark can claim.

- [ ] **Step 4: Full test run**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit, then push once**
