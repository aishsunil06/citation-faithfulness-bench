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

- **Ambiguous claims come first.** A claim whose overlap sits near a judge's
  decision boundary is where judge and human are most likely to diverge, which
  is what calibration needs to measure. Overlap of 0.02 or 0.98 is a case every
  judge already gets right.
- **Modes round-robin**, so a 50-label budget cannot land entirely on `numeric`
  and leave `contested` with nothing.
- **A deliberate share of known-verdict claims is retained** (`--oracle-fraction`,
  default 0.35) so the label set has variance. See the warning below.

Pass `--no-priority` for raw document order, which is only useful for auditing
coverage.

> **Label variance is not optional.** An earlier version of this tool excluded
> every known-verdict claim to save effort. Against a verbatim-quoting engine
> those were the only claims that were not trivially supported, so the first
> real labelling session produced 17 labels that were *all* `supported`. Cohen's
> kappa came out **0.000** at 82% raw accuracy, because kappa corrects for
> chance agreement and unanimous labels make chance agreement total. Two fixes
> followed: retain some known-verdict claims, and add `LossyEngine` so the
> engines under test can actually be wrong.

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
| `numeric-drift` | shifts a figure so it no longer matches the source | `contradicted` |
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

## Verdicts

| Verdict | Meaning |
|---|---|
| `supported` | the cited snippet states the claim, numbers and dates included |
| `partial` | relevant but weaker, narrower, or off in a detail |
| `unsupported` | the snippet is about something else, or silent on the point |
| `contradicted` | the snippet addresses this exact point and states the opposite |
| `uncited` | the claim carried no citation at all |

`contradicted` is deliberately separate from `unsupported`. An irrelevant
citation is careless attribution; a source that states the opposite means the
engine had the right evidence in hand and asserted against it. Those are
different product failures with different fixes, and collapsing them hides
which one an engine actually has. Neither earns credit under strict scoring.

`uncited` is likewise kept apart: asserting something with no source at all is
a different failure from asserting it with a bad one.

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
`LossyEngine`, scored by the rule-based judge.

**Fault detection by the lexical judge:**

| Injected fault | n | detection | judge said |
|---|---|---|---|
| `dropped-citation` | 45 | 100.0% | uncited x45 |
| `numeric-drift` | 21 | 100.0% | contradicted x21 |
| `wrong-source` | 45 | **53.3%** | unsupported x24, contradicted x11, partial x10 |

Two results, and the second is the interesting one.

**Exact numeric checking is solved.** Token overlap plus a regex catches every
swapped magnitude, and it does so whether the corruption is wild or subtle: a
missing figure is missing either way. An LLM judge has nothing to add here.

**Wrong-source attribution is where it breaks, in both directions.** Of 45
mis-attributed claims the judge got 24 right, called 10 `partial` because a
topically adjacent source still shares most of the claim's vocabulary, and
called 11 `contradicted` when they were merely unsupported. That last error is
the diagnostic one. Example:

| | |
|---|---|
| claim | Northwind Logistics reported revenue of 412 million dollars in fiscal 2024 |
| cited | In the first quarter of fiscal 2025 Northwind Logistics reported revenue of 118 million dollars |

Both figures are true. The source is about a different period, so it fails to
support the claim rather than contradicting it. Distinguishing *same fact,
different value* from *different scope entirely* needs the reasoning a lexical
rule cannot do, and is the concrete thing an LLM judge has to earn its cost on.

**The oracle itself needed three corrections**, which is worth recording
because it is the main hazard of this technique. Fault injection is only worth
anything while the correct verdict is beyond argument, and the first version
was not:

- Corruption scaled every figure multiplicatively, turning the year 2030 into
  3248. Detectable from implausibility alone, so it measured nothing.
- Bounded quantities were corrupted. "Reaches 3.1 billion **by 2030**" entails
  "reaches 3.1 billion **by 3248**", because a later deadline is a weaker claim
  the source already supports. Labelling that `contradicted` made the oracle
  simply wrong.
- Years were corrupted. "Capacity was 240 as of 2026" cited to a 2025 source is
  `unsupported`, since the source is silent on 2026 rather than denying it.

Injection is now restricted to magnitudes on point values, which cut the sample
from 40 to 21 and is the right trade: 21 indisputable cases beat 40 arguable
ones.

**Judge calibration against human labels (n=24):**

| Judge | n | accuracy | Cohen's kappa | reading |
|---|---|---|---|---|
| `judge:lexical` | 20 | 45.0% | +0.127 | poor |
| `judge:claude-opus-5` | 24 | 87.5% | **+0.711** | substantial |

This is the result the fault-injection work was a proxy for, and it confirms
what fault injection predicted: the lexical baseline is barely better than
chance against a human, and an LLM judge closes most of the gap. The lexical
judge's failure is concentrated exactly where fault injection said it would be,
over-calling `contradicted` on sources that merely cover a different period.

Human-vs-model agreement was perfect on all 17 `unsupported` claims and 4 of 5
`contradicted`. The three disagreements were:

| human | model | case |
|---|---|---|
| `supported` | `partial` | "roughly 120 million" against a source saying 118 million |
| `supported` | `contradicted` | a claim of 2.5 billion against a source saying 3.1 billion |
| `partial` | `unsupported` | FY2024 revenue cited to a Q1 FY2025 source |

The first and third are genuine borderline calls: whether a hedged rounding is
faithful, and whether a right-company-wrong-period citation is weak or simply
absent. Those are the cases a benchmark exists to surface.

**Caveats on this number, which matter more than the number:**

- **n=24 is small.** One label moves kappa by roughly 0.03, so treat 0.711 as
  "substantial, probably" rather than a measurement.
- **One annotator, so there is no inter-annotator agreement.** Without a second
  human there is no way to know how much of the residual disagreement is the
  judge being wrong versus the task being genuinely ambiguous.
- **The model judge is not independent of the data.** The same system wrote the
  corpus, the fault injectors, the lexical judge, and these labels, so shared
  blind spots are likely and this comparison flatters the LLM judge. A judge
  from a different provider, on a corpus it did not author, is the honest
  version of this experiment.

**Clean-engine baseline:** 95% confidence intervals are reported on every
leaderboard score, and every fault-injected variant falls outside the clean
engine's interval.

## Current status and known limitations

Working: retrieval with chunk sweeps, extractive and generative engines, four
fault injectors, claim decomposition, two judges, bootstrap confidence
intervals, failure-mode decomposition, resumable human labelling with
information-ordered queueing, dataset validation, report rendering. 96 tests,
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
  construction. Use `--engine lossy` for labelling and sweeps; `extractive`
  remains useful only as the ceiling reference line.
- **Answer quality is out of scope.** Only the citation is graded. An engine can
  cite every sentence perfectly and still answer the wrong question, or answer
  half of a two-part question, and score 100%. Measuring answer completeness
  alongside faithfulness is the obvious next axis and is not implemented.
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
    lossy.py       paraphrases lossily, for genuinely borderline claims
    mock.py        scripted engine for tests
data/              seed corpus and question set
tests/             96 offline tests
```
