# Findings

Every figure below is reproducible from the committed corpus, question set and
run artifacts. The commands are given with each result.

## Setup

| | |
|---|---|
| Corpus | 37 English Wikipedia articles, 209,675 words, six domains |
| Questions | 106, stratified over six failure modes, 260 declared aspects |
| Engines | `extractive` and `lossy`, at 90- and 250-word chunk widths |
| Injectors | numeric-drift, wrong-source, dropped-citation, polarity-flip |
| Scale | 1,272 answer records, 3,762 claims, 5,475 labels |
| Judges | `lexical` (rule-based, all claims), `claude-opus-5` (54-claim sample) |

```bash
python -m cfbench validate --questions data/questions.real.jsonl --corpus data/corpus.real.jsonl
python -m cfbench run --engine extractive --questions data/questions.real.jsonl --corpus data/corpus.real.jsonl --chunk-words 90 --chunk-words 250
python -m cfbench run --engine lossy --questions data/questions.real.jsonl --corpus data/corpus.real.jsonl --chunk-words 90 --chunk-words 250 --append --fault numeric-drift --fault wrong-source --fault dropped-citation --fault polarity-flip
python -m cfbench judge --judge lexical --questions data/questions.real.jsonl
python -m cfbench report --questions data/questions.real.jsonl --out runs/report.md
```

## 1. Narrower chunks cite more faithfully

The headline result, and the one the earlier synthetic corpus could not
produce. Confidence intervals are percentile bootstraps over claims.

| Engine | 90-word chunks | 250-word chunks |
|---|---|---|
| `extractive` | **83.8%** [79.3, 87.7] | **66.0%** [61.0, 71.1] |
| `lossy` | **79.6%** [75.1, 84.1] | **63.2%** [58.2, 68.2] |

The intervals do not overlap, so the run separates the two widths rather than
merely ranking them. The effect is about 18 points for both engines, which is
larger than the gap between the engines themselves.

The mechanism is visible in the per-mode breakdown: the 250-word condition
loses most ground on `simple` and `numeric` questions, where the answer is one
sentence and a wider chunk buries it among unrelated ones. A wide chunk gives
retrieval more ways to look relevant while supporting less.

Why this matters beyond the benchmark: chunk width is usually tuned for
retrieval hit-rate, where wider is often better because it is more likely to
contain the answer. Measured on citation faithfulness the pressure runs the
other way, and that trade-off is invisible to a retrieval-only metric.

## 2. Judges are complementary, not ranked

Fault detection by the rule-based judge, over all claims:

| Injected fault | n | detection | what it said instead |
|---|---|---|---|
| `dropped-citation` | 627 | **100.0%** | — |
| `numeric-drift` | 349 | **95.4%** | supported x12, partial x4 |
| `wrong-source` | 626 | **62.5%** | partial x111, contradicted x97, supported x27 |
| `polarity-flip` | 57 | **1.8%** | partial x56 |

On the 21 oracle-labelled claims in the model-judged sample, the two judges
fail in different places:

| Injected fault | `lexical` | `claude-opus-5` |
|---|---|---|
| `dropped-citation` | 4/4 | 4/4 |
| `numeric-drift` | 2/2 | 1/2 |
| `wrong-source` | 12/13 | 12/13 |
| `polarity-flip` | **0/2** | **2/2** |

This is the useful finding. The expected story was "the LLM judge is better";
the measured story is that each is blind where the other sees.

- **Exact numeric checking is solved by a regex**, and the LLM judge is
  marginally *worse* at it, because the rule either finds the figure in the
  snippet or does not, while a model can talk itself into accepting a near
  miss.
- **Polarity is invisible to lexical overlap.** Deleting "not" barely changes
  the token set, so overlap stays high and the judge returns `partial`. It
  scored 1.8% over 57 injections, which is indistinguishable from never
  detecting it. The model judge caught both sampled cases.
- **Wrong-source attribution is hard for both**, at 62.5% for the rule.
  Its errors run in both directions: 111 called `partial` because a topically
  adjacent source shares the claim's vocabulary, and 97 called `contradicted`
  when the source merely covers a different period and is therefore silent
  rather than opposed.

Judge-vs-judge agreement on the 54-claim sample: 74.1% raw, **Cohen's kappa
+0.612**. Substantial, and the disagreements concentrate exactly on the
polarity and scope cases above.

```bash
python tools/sample_claims.py export --n 60 --questions data/questions.real.jsonl --out runs/sample.json
python tools/sample_claims.py import --labeler judge:claude-opus-5 --verdicts runs/verdicts.json
```

## 3. Faithfulness and completeness are genuinely different axes

Answer coverage, scored over the 106 questions that declare aspects:

| Engine | strict faithfulness | answer completeness |
|---|---|---|
| `extractive-k3-c90` | 83.8% | 63.5% |
| `lossy-k3-c90` | 79.6% | 62.7% |
| `lossy-k3-c90+wrong-source` | **7.1%** | **62.7%** |
| `lossy-k3-c90+dropped-citation` | **0.0%** | **62.7%** |

The last two rows are the argument for the second axis. `wrong-source`
destroys faithfulness while leaving completeness untouched, because the answer
text is unchanged and only its citations were repointed. An engine in that
state answers the question fully and cites garbage, and a faithfulness-only
benchmark would report it as catastrophic while a completeness-only benchmark
would report it as fine.

Citation-only faults leaving coverage *identical* is the sanity check that
completeness is not faithfulness in another hat.

```bash
python -m cfbench score-completeness --questions data/questions.real.jsonl
```

## 4. The oracle was wrong five times, and that is the main lesson

Fault injection's whole value is a ground truth nobody can argue with. Five
separate times the oracle asserted something untrue, and each was found by
reading real output rather than by a test:

1. **Implausible corruption.** Scaling every figure multiplicatively turned the
   year 2030 into 3248. Detectable from implausibility alone, so it measured
   nothing about reading a source.
2. **Bounded quantities.** "Reaches 3.1 billion **by 2030**" *entails* "reaches
   3.1 billion **by 3248**": a later deadline is a weaker claim the source
   already supports. The oracle said `contradicted` when the truth was
   `partial`, so a judge "detecting" it was only agreeing with a mistake.
3. **Scope versus value.** "Capacity was 240 as of 2026" cited to a 2025 source
   is `unsupported`, because the source is silent on 2026 rather than denying
   it. Date shifts change scope; only magnitude shifts change value.
4. **Unsupported base claims.** The injector assumed the claim handed to it was
   already faithful. The extractive engine guaranteed that by quoting verbatim;
   the lossy engine does not. Corrupting an already-unsupported claim and
   calling it `contradicted` asserts something false about the source.
5. **Identifiers and malformed numerals.** "Fab 8" became "Fab 7", changing
   which facility the claim was about, and "3,000" became "3,0.0" because the
   number regex split the comma group.

Each fix narrowed what the injector will touch. `numeric-drift` now fires on
magnitudes only, in non-bound contexts, where the cited snippet actually states
the original figure. The sample fell from 40 to 21 on the old synthetic set, and
the trade is correct: 21 indisputable cases are worth more than 40 arguable
ones.

The transferable lesson is that a synthetic oracle is not automatically
trustworthy just because it was constructed. It encodes assumptions about
entailment, scope and reference that are easy to get wrong and invisible until
someone reads the output.

## 5. A correct injector can be inapplicable

`entity-swap` injects **zero** faults on this corpus, across both engines and
chunk widths 90/250/500/900.

Its gate requires the substituted entity to appear in the *cited* snippet with
a differing figure for the same predicate — without that, the source is merely
silent about the substitute and the verdict is `unsupported`, not
`contradicted`. No retrieved snippet in a 37-article Wikipedia corpus contains
two entities sharing a predicate with differing figures, because encyclopedia
prose describes one subject at a time rather than comparing two.

The injector is correct, tested, and reported at zero rather than quietly
omitted. Making it fire would need a corpus of comparative passages, which is a
different dataset, not a threshold change.

## What this does not show

- **No human labels.** Validation rests entirely on a synthetic oracle plus
  judge-vs-judge comparison. The oracle covers only unambiguous cases, so
  nothing here measures whether a judge matches *human* judgement on the
  borderline cases where real disagreement lives, and no inter-annotator
  ceiling exists to measure any judge against.
- **The model judge is not independent of the data.** The same system authored
  the corpus pipeline, the injectors, the rule-based judge and the model
  verdicts. Shared blind spots are likely and the comparison flatters the model
  judge. A judge from a different provider on a corpus it did not author is the
  honest version.
- **n=54 for the model judge**, and 2 polarity cases inside it. The polarity
  result is consistent with the 57-injection full-corpus figure of 1.8% for the
  rule, but 2 cases cannot carry it alone.
- **Completeness is keyword presence.** A correct paraphrase avoiding the
  aspect's wording scores as a miss, so the figures are a lower bound and are
  comparable between engines over identical aspects, never readable as an
  absolute.
- **One corpus, one language, one register.** Encyclopedia prose is unusually
  clean and unusually single-subject. Results on news, filings or forum text
  could differ, and the `entity-swap` finding suggests register matters.
