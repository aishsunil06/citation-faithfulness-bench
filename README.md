# Citation Faithfulness Benchmark

Measures whether the source an answer engine cites **actually supports** the
sentence it is attached to.

This is not a hallucination benchmark. The question is narrower and more
tractable: given that an engine said something and pointed at a source, does
that source say it? A true claim attached to an irrelevant source still fails,
because the citation is the product feature being tested.

## Why per-claim, and why judge calibration

Two design choices do most of the work:

**Scoring is per (claim, citation) pair, not per answer.** An engine that makes
three assertions and cites correctly for two scores 2/3. Answer-level scoring
collapses that into a single pass/fail and loses the resolution needed to
compare engines.

**Every judge is calibrated against human labels before its numbers are used.**
An LLM judge whose agreement with humans is unmeasured is not a measurement
instrument. The report refuses to print a judge-derived score without printing
that judge's Cohen's kappa beside it, and a deliberately cheap rule-based judge
ships alongside so you can see how much agreement was available for free.

Kappa rather than raw accuracy, because these verdicts skew heavily toward
`supported`: a judge that always answers `supported` posts high accuracy and a
kappa near zero. There is a test asserting exactly that pathology.

## Install

```bash
python -m pip install -r requirements.txt   # only pytest; the core is stdlib
```

The retrieval, scoring, and rule-based judge are pure standard library. An API
key is needed **only** for the generative engine and the LLM judge.

## Run it

Fully offline, start to finish:

```bash
python -m cfbench run --engine extractive --chunk-words 60 --chunk-words 250
python -m cfbench judge --judge lexical
python -m cfbench report --out runs/report.md
```

Then add human labels, which is what makes the judge numbers mean anything:

```bash
python -m cfbench label --labeler human:yourname --limit 50
python -m cfbench report --out runs/report.md     # now includes kappa
```

With `OPENAI_API_KEY` set, the generative engine and LLM judge become available:

```bash
python -m cfbench run   --engine rag --model gpt-4o-mini --chunk-words 120 --append
python -m cfbench judge --judge llm:gpt-4o-mini
```

Read the disagreements. This is where a score turns into a finding:

```bash
python -m cfbench disagree --judge judge:lexical
```

## Pipeline

```
corpus.jsonl ──► BM25 index ──► engine ──► answer prose + citations
                                              │
                                              ▼
                                   claims.py decomposes into
                                   one claim per sentence
                                              │
                      ┌───────────────────────┴──────────────────┐
                      ▼                                          ▼
            judge (lexical | LLM)                        human labelling
                      │                                          │
                      └──────────────► metrics.agreement ◄───────┘
                                       (accuracy, Cohen's kappa)
                                              │
                                              ▼
                                   leaderboard + failure modes
```

## Failure modes

The question set is stratified so a headline number can be decomposed instead
of averaging over whatever was easy to collect:

| Mode | What it probes |
|---|---|
| `simple` | single fact, single source. The control group. |
| `multi_hop` | answer requires joining two or more sources |
| `numeric` | a figure that must match the source exactly |
| `negation` | the correct answer asserts something is *not* the case |
| `recency` | the answer depends on which snapshot of the world |
| `contested` | sources legitimately disagree, and that disagreement is the fact |

A question may carry several modes, so the buckets overlap by design and do not
partition the claim set.

## The seed dataset is synthetic, on purpose

`data/corpus.jsonl` and `data/questions.jsonl` describe fictional companies
(Northwind Logistics, Helion Grid). Two reasons:

1. The corpus **is** the ground truth, so labelling needs no external fact
   checking and the benchmark cannot silently rot as the real world moves.
2. It lets specific traps be planted deliberately. `Northwind Aviation` exists
   purely as a name-collision decoy with a real-looking revenue figure, so an
   engine citing it for a Northwind Logistics question is caught. Helion's Mesa
   site has one capacity in 2025 and another in 2026, so an undated answer
   cannot be fully supported.

Scaling to a real corpus is the next step, and the loader takes any
`{url, title, text}` JSONL. The synthetic set exists to validate the pipeline,
not to be the published result.

## Current status and known limitations

Working: retrieval, chunk sweeps, extractive and generative engines, claim
decomposition, both judges, scoring, failure-mode decomposition, resumable
human labelling, report rendering. 40 tests, all offline.

Honest limitations:

- **No human labels yet**, so no judge is calibrated and no faithfulness number
  here should be quoted. The report says so rather than hiding it.
- **The chunk sweep is currently uninformative.** The extractive engine quotes
  verbatim from the chunk it cites, so it is near the faithfulness ceiling by
  construction and all chunk widths score identically. The sweep only becomes
  meaningful for the generative engine, which can drift from its source. The
  extractive line is best read as the reference an LLM engine should be measured
  against, not as a result in itself.
- **Claim decomposition is sentence-level.** A sentence asserting two things
  with one citation is scored as one claim, which is generous to the engine.
- **The lexical judge cannot do inference.** It catches numeric and polarity
  mismatches, but a claim correctly entailed by a source in different words
  reads as low overlap and gets marked down. That is the gap the LLM judge is
  meant to close, and the kappa comparison is how you find out whether it does.
- **Judging runs against stored snippets, not live URLs**, so a run stays
  reproducible after the web changes. It also means link rot is out of scope.

## Layout

```
cfbench/
  schema.py        data model, JSONL persistence
  retrieval.py     BM25 + document chunking, no dependencies
  claims.py        answer prose -> atomic claims with citations
  judge.py         LexicalJudge (offline baseline), LLMJudge
  metrics.py       faithfulness scores, agreement, Cohen's kappa
  labeling.py      resumable terminal labelling tool
  report.py        leaderboard, failure modes, calibration
  cli.py           run / judge / label / report / disagree
  engines/
    extractive.py  retrieval-only; the faithfulness reference line
    openai_rag.py  retrieval + LLM that must cite inline
    mock.py        scripted engine for tests
data/              seed corpus and question set
tests/             40 offline tests
```
