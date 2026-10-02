# Citation Faithfulness Benchmark

Measures whether the source an answer engine cites **actually supports** the
sentence it is attached to.

This is not a hallucination benchmark. The question is narrower and more
tractable: given that an engine said something and pointed at a source, does
that source say it? A true claim attached to an irrelevant source still fails,
because the citation is the product feature being tested.

**[Findings](docs/FINDINGS.md)** — results, with the command that reproduces each.

## Headline results

Over 37 real Wikipedia articles (209,675 words), 106 stratified questions and
3,762 claims:

- **Narrower chunks cite more faithfully**, with non-overlapping confidence
  intervals: 83.8% [79.3, 87.7] at 90-word chunks against 66.0% [61.0, 71.1] at
  250. Chunk width is normally tuned for retrieval hit-rate, where wider often
  wins; measured on faithfulness the pressure runs the other way.
- **The two judges are complementary, not ranked.** A regex catches 95% of
  swapped figures and **1.8%** of inverted negations. The LLM judge caught both
  sampled polarity cases and was marginally *worse* on numerics.
- **Faithfulness and answer completeness are different axes.** A `wrong-source`
  engine scores 7.1% faithfulness and 62.7% completeness: it answers the
  question fully and cites garbage.
- **The synthetic oracle was wrong five times**, each found by reading real
  output rather than by a test. That is the most transferable result here.

## Why per-claim, and why a calibrated judge

Two design choices do most of the work.

**Scoring is per (claim, citation) pair, not per answer.** An engine that makes
three assertions and cites correctly for two scores 2/3. Answer-level scoring
collapses that into pass/fail and loses the resolution needed to compare
engines.

**No judge's number is reported without a measure of that judge's error.** An
LLM judge whose agreement is unmeasured is not a measurement instrument. A
deliberately cheap rule-based judge ships alongside so you can see how much
agreement was available for free, and fault injection supplies a ground truth
that needs no human reading.

Cohen's kappa rather than raw accuracy, because these verdicts skew heavily
toward `supported`: a judge that always answers `supported` posts high accuracy
and a kappa near zero. There is a test asserting exactly that pathology.

## Install

```bash
python -m pip install -r requirements.txt   # only pytest; the core is stdlib
```

Retrieval, scoring, chunking, the rule-based judge and the report are pure
standard library. An API key is needed **only** for the optional generative
engine and LLM judge.

## Run it

Fully offline, start to finish:

```bash
python -m cfbench validate --questions data/questions.real.jsonl --corpus data/corpus.real.jsonl
python -m cfbench run    --engine lossy --questions data/questions.real.jsonl \
                         --corpus data/corpus.real.jsonl --chunk-words 90 --chunk-words 250 \
                         --fault numeric-drift --fault wrong-source \
                         --fault dropped-citation --fault polarity-flip
python -m cfbench judge  --judge lexical --questions data/questions.real.jsonl
python -m cfbench report --questions data/questions.real.jsonl --out runs/report.md
```

Answer coverage, the second axis:

```bash
python -m cfbench score-completeness --questions data/questions.real.jsonl
```

Read the disagreements. This is where a score turns into a finding:

```bash
python -m cfbench disagree --judge judge:lexical
```

### Fault injection

If a *known* corruption is applied to an otherwise faithful answer, the correct
verdict is known by construction, so a judge can be measured with no human
labelling at all. Each injector writes `oracle:<fault>` labels beside the
corrupted answers, so the existing agreement machinery compares judge against
oracle with no special-casing.

| Fault | Corruption | Correct verdict | Fires here? |
|---|---|---|---|
| `numeric-drift` | shifts a magnitude the snippet states | `contradicted` | 349 |
| `wrong-source` | repoints the claim at another retrieved source | `unsupported` | 626 |
| `dropped-citation` | strips the citation | `uncited` | 627 |
| `polarity-flip` | inverts an unhedged negation | `contradicted` | 57 |
| `entity-swap` | substitutes a named entity | `contradicted` | **0** |

**Every injector declines far more often than it fires**, and that is the
design, not a defect. Injection is only worth anything while the verdict is
beyond argument: years, bounded quantities, identifiers, hedged negations and
claims whose cited snippet never stated the original figure are all out of
scope, because each of those produced a *wrong* oracle at some point. See
[FINDINGS §4](docs/FINDINGS.md) for all five.

`entity-swap` fires zero times on encyclopedia prose, which describes one
subject at a time rather than comparing two. It is reported at zero rather than
quietly dropped.

### Corpus capture

The library never touches the network. Capture is an operator step whose output
is a plain `{url, title, raw}` file, so a run is reproducible from the committed
corpus:

```bash
python tools/fetch_wikipedia.py data/corpus.raw.jsonl    # resumable, backs off on 429
python -m cfbench ingest --input data/corpus.raw.jsonl --out data/corpus.real.jsonl
```

`ingest` strips navigation chrome and reference markers, truncates at the first
footer heading, and drops near-duplicates. The footer rule matters more than it
sounds: dropping the "See also" *heading* leaves its body, and a real capture
ended with link headlines welded onto the prose, which a retrieved chunk would
present as evidence.

## Verdicts

| Verdict | Meaning |
|---|---|
| `supported` | the snippet states the claim, numbers and dates included |
| `partial` | relevant but weaker, narrower, or off in a detail; also a claim the source *entails* but states more precisely |
| `unsupported` | the snippet is about something else, or silent on the point |
| `contradicted` | the snippet addresses this exact point and states the opposite |
| `uncited` | the claim carried no citation at all |

`contradicted` is separate from `unsupported` on purpose. An irrelevant
citation is careless attribution; a source stating the opposite means the engine
had the right evidence and asserted against it. Different failures, different
fixes. Neither earns credit under strict scoring.

The distinction that most often goes wrong, in judges and in the oracle alike:
a different figure for a **different period or entity** is `unsupported`, not
`contradicted`. Both statements can be true; the source is silent, not opposed.

## Failure modes

The question set is stratified so a headline number can be decomposed rather
than averaged over whatever was easy to collect.

| Mode | What it probes | n |
|---|---|---|
| `numeric` | a figure that must match the source exactly | 61 |
| `recency` | the answer depends on which snapshot of the world | 27 |
| `negation` | the correct answer asserts something is *not* the case | 25 |
| `simple` | single fact, single source; the control group | 23 |
| `contested` | sources legitimately disagree, and that disagreement is the fact | 23 |
| `multi_hop` | the answer requires joining two or more sources | 21 |

A question may carry several modes, so buckets overlap by design.

## The corpus is real, and its collisions are real

37 English Wikipedia articles across grid storage, semiconductors, transport,
spaceflight, public health and renewables. Pairs were chosen so confusable
sources exist naturally: Panama vs Suez, Rotterdam vs Shanghai, Boeing 787 vs
Airbus A350, Heathrow vs Changi, Hubble vs JWST, Falcon 9 vs Ariane 6, TSMC vs
GlobalFoundries, four vaccines.

Measured afterwards, the most similar document pairs in the corpus are exactly
those intended traps, topping out at Jaccard 0.315 — so no deduplication
triggered, and the traps are genuine rather than contrived.

Several questions exploit real intra-corpus disagreement rather than invented
conflict: pumped-storage efficiency appears as 70-80% in one article and 75-85%
in another; Rotterdam's own rankings are mutually incompatible; the polio
article states the VAPP rate two different ways.

## Pipeline

```
corpus.raw.jsonl --> ingest --> corpus.real.jsonl --> BM25 index
                                                          |
                                                          v
                                            engine --> prose + citations
                                                          |
                                                claims.py decomposes into
                                                one claim per sentence
                                                          |
             +----------------------------+---------------+------------+
             v                            v                           v
    judge (lexical | LLM)        human labelling               oracle (fault
             |                   (optional, none here)        injection: verdict
             |                            |                   known by design)
             +--------------> metrics.agreement <-------------------+
                       kappa  |  per-fault detection rate
                                       |
                                       v
                  leaderboard (bootstrap CIs) + failure modes
                             + answer completeness
```

## Current status and limitations

Working: corpus ingestion, sentence-aware chunking with sweeps, extractive /
lossy / generative engines, five fault injectors, claim decomposition, two
judges, bootstrap confidence intervals, failure-mode decomposition, answer
completeness, information-ordered human labelling, dataset validation with an
answerability check, report rendering. **205 tests, all offline.**

Honest limitations:

- **No human labels.** Validation rests on a synthetic oracle and
  judge-vs-judge comparison. The oracle covers only unambiguous cases, so
  nothing here measures whether a judge matches *human* judgement on the
  borderline cases where real disagreement lives. With one annotator there is
  also no inter-annotator ceiling to measure any judge against. The tooling for
  both exists (`cfbench label`); the labels do not.
- **The model judge is not independent of the data.** The same system authored
  the corpus pipeline, the injectors, the rule-based judge and the model
  verdicts, so shared blind spots are likely and the comparison flatters the
  model. A different provider on a corpus it did not author is the honest
  version.
- **n=54 for the model judge**, including only 2 polarity cases. That result is
  consistent with the 57-injection full-corpus figure but cannot rest on 2.
- **Completeness is keyword presence, not semantic equivalence.** A correct
  paraphrase avoiding the aspect's wording counts as a miss, so figures are a
  lower bound, comparable between engines over identical aspects and never
  readable as an absolute.
- **Claim decomposition is sentence-level.** A sentence asserting two things
  under one citation scores as one claim, which is generous to the engine.
- **The bootstrap resamples claims, not questions.** Claims from one question
  are correlated, so a question-level cluster bootstrap would give wider, more
  honest intervals.
- **Answer quality beyond aspect coverage is out of scope.** An engine can cite
  every sentence perfectly and answer a question nobody asked; completeness
  catches the omission but not the irrelevance.
- **One corpus, one language, one register.** Encyclopedia prose is unusually
  clean and single-subject. The `entity-swap` result suggests register matters.
- **Judging runs against stored snippets, not live URLs**, so runs stay
  reproducible after the web changes. Link rot is out of scope.

## Layout

```
cfbench/
  schema.py        data model, content-addressed ids, JSONL persistence
  ingest.py        corpus cleaning, footer truncation, near-duplicate removal
  retrieval.py     BM25 + sentence-aware chunking, no dependencies
  claims.py        answer prose -> atomic claims with citations
  judge.py         LexicalJudge (offline baseline), LLMJudge
  metrics.py       scores, agreement, Cohen's kappa, bootstrap CIs, sample size
  completeness.py  answer coverage of declared question aspects
  labeling.py      resumable labelling tool + information-ordered queue
  report.py        leaderboard, failure modes, calibration, completeness
  cli.py           ingest / run / judge / label / report / disagree / validate
  engines/
    extractive.py  retrieval-only; the faithfulness reference line
    lossy.py       lossy paraphrase, for genuinely borderline claims
    faulty.py      fault injectors with known-correct verdicts
    openai_rag.py  retrieval + LLM that must cite inline
    mock.py        scripted engine for tests
data/              real corpus and 106-question set, plus the synthetic seed set
tools/             corpus capture, claim sampling for an expensive judge
docs/FINDINGS.md   results with reproduction commands
tests/             205 offline tests
```

Content-addressed claim ids mean labels survive re-runs: the same engine over
the same text reproduces the same ids. That was learned the hard way — random
ids once orphaned a real labelling session, and a later re-run with changed
chunking preserved 54 of 60 model verdicts precisely because ids are derived
from content.
