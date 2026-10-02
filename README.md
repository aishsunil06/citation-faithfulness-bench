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

The queue is ordered by expected information per label, not document order:

- **Claims with an oracle label are dropped.** Their verdict is known by
  construction, so a human read buys nothing. On the current run that removes
  145 of 240 claims from the queue.
- **Ambiguous claims come first.** A claim whose overlap sits near a judge's
  decision boundary is where judge and human are most likely to diverge, which
  is what calibration needs to measure. Overlap of 0.02 or 0.98 is a case every
  judge already gets right.
- **Modes round-robin**, so a 50-label budget cannot land entirely on `numeric`
  and leave `contested` with nothing.

Pass `--no-priority` for raw document order, which is only useful for auditing
coverage.

With `OPENAI_API_KEY` set, the generative engine and LLM judge become available:

```bash
python -m cfbench run   --engine rag --model gpt-4o-mini --chunk-words 120 --append
python -m cfbench judge --judge llm:gpt-4o-mini
```

Read the disagreements. This is where a score turns into a finding:

```bash
python -m cfbench disagree --judge judge:lexical
```

Validate the dataset and see how many labels a given precision needs:

```bash
python -m cfbench validate
```

### Fault injection

Human labels are slow. But if a *known* corruption is applied to an otherwise
faithful answer, the correct verdict is known by construction, so a judge can
be measured on day one:

```bash
python -m cfbench run --engine extractive --chunk-words 90 --fault numeric-drift --fault wrong-source --fault dropped-citation --fault padded-claim
python -m cfbench judge --judge lexical
python -m cfbench report --out runs/report.md
```

Each injector writes `oracle:<fault>` labels next to the corrupted answers, so
the existing agreement machinery compares judge against oracle with no
special-casing. Four faults ship:

| Fault | Corruption | Correct verdict |
|---|---|---|
| `numeric-drift` | shifts a figure so it no longer matches the source | `unsupported` |
| `wrong-source` | repoints the claim at a different retrieved source | `unsupported` |
| `dropped-citation` | strips the citation, leaving a bare assertion | `uncited` |
| `padded-claim` | appends a confident sentence no source supports | `unsupported` |

This establishes a **floor**, not a substitute for human labelling: injected
faults are unambiguous, and real citation failures are frequently borderline.
A judge that misses synthetic corruption will certainly miss subtler real
failures.

## Pipeline

```
corpus.jsonl ──► BM25 index ──► engine ──► answer prose + citations
                                              │
                                              ▼
                                   claims.py decomposes into
                                   one claim per sentence
                                              │
           ┌──────────────────────┼──────────────────────┐
           ▼                      ▼                      ▼
  judge (lexical|LLM)     human labelling      oracle (fault injection:
           │                      │             verdict known by design)
           │                      │                      │
           └──────────► metrics.agreement ◄──────────────┘
                     kappa  |  fault detection rate
                                  │
                                  ▼
            leaderboard (with bootstrap CIs) + failure modes
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

## Findings so far

Run: 15 questions, 10-document corpus, BM25 with 90-word chunks, `top_k=3`,
scored by the rule-based judge. 45 claims per engine variant.

**Fault detection by the lexical judge:**

| Injected fault | n | detection |
|---|---|---|
| `dropped-citation` | 45 | 100.0% |
| `numeric-drift` | 40 | 100.0% |
| `padded-claim` | 15 | 100.0% |
| `wrong-source` | 45 | **77.8%** |

The informative result is the last row. Token overlap catches a changed number
every time, because the number is simply absent from the cited snippet. It is
substantially weaker on **wrong-source attribution**: it marked 10 of 45
mis-attributed claims as `partial` rather than `unsupported`, because a
topically adjacent wrong source still shares most of the claim's vocabulary.

That is a concrete, falsifiable hypothesis for what an LLM judge has to earn
its cost on: wrong-source attribution, not numeric drift. Numeric checking is
already solved by ten lines of regex.

**Clean-engine baseline:** `extractive-k3-c90` scores 77.8% strict,
95% CI [64.4%, 88.9%]. Every fault-injected variant falls outside that
interval, which is the sanity check that the instrument responds to real
degradation rather than noise.

## Current status and known limitations

Working: retrieval with chunk sweeps, extractive and generative engines, four
fault injectors, claim decomposition, two judges, bootstrap confidence
intervals, failure-mode decomposition, resumable human labelling with
information-ordered queueing, dataset validation, report rendering. 65 tests,
all offline.

Honest limitations:

- **No human labels yet.** Fault injection measures a floor on unambiguous
  cases; it says nothing about borderline ones, which is where real
  disagreement lives. No Cohen's kappa exists yet, and the report refuses to
  print one rather than implying calibration that has not happened.
- **The corpus is 10 synthetic documents.** Results characterise the
  instrument, not the state of real answer engines. Scaling to a real corpus
  is the next step; the loader takes any `{url, title, text}` JSONL.
- **`python -m cfbench validate` reports the sample size needed:** +/-10%
  needs 97 labels, +/-5% needs 385. The current 15-question set cannot support
  narrow claims, and the confidence intervals say so out loud.
- **The chunk sweep is uninformative for the extractive engine**, which quotes
  verbatim from the chunk it cites and so sits near the faithfulness ceiling by
  construction. Chunk geometry should only matter for the generative engine.
  Untested, because that needs an API key.
- **Claim decomposition is sentence-level.** A sentence asserting two things
  under one citation is scored as one claim, which is generous to the engine.
- **The bootstrap resamples claims, not questions.** Claims from the same
  question are correlated, so a question-level cluster bootstrap would give
  wider, more honest intervals.
- **Judging runs against stored snippets, not live URLs**, so runs stay
  reproducible after the web moves. Link rot is out of scope.

## Layout

```
cfbench/
  schema.py        data model, JSONL persistence
  retrieval.py     BM25 + document chunking, no dependencies
  claims.py        answer prose -> atomic claims with citations
  judge.py         LexicalJudge (offline baseline), LLMJudge
  metrics.py       faithfulness scores, agreement, Cohen's kappa
  labeling.py      resumable labelling tool + information-ordered queue
  report.py        leaderboard, failure modes, calibration
  cli.py           run / judge / label / report / disagree / validate
  engines/
    extractive.py  retrieval-only; the faithfulness reference line
    openai_rag.py  retrieval + LLM that must cite inline
    faulty.py      fault injectors with known-correct verdicts
    mock.py        scripted engine for tests
data/              seed corpus and question set
tests/             65 offline tests
```
