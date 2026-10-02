"""Command line entry point.

    python -m cfbench run    --engine extractive --chunk-words 90 --chunk-words 300
    python -m cfbench judge  --judge lexical
    python -m cfbench label  --labeler human:aishwarya --limit 50
    python -m cfbench report --out runs/report.md
    python -m cfbench disagree --judge judge:lexical
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .completeness import aggregate, completeness, most_missed
from .engines import ExtractiveEngine, OpenAIRAGEngine
from .engines.lossy import LossyEngine
from .engines.faulty import FAULTS, wrap
from .ingest import ingest_rows
from .judge import get_judge
from .labeling import label_session, load_labels
from .report import build_report, disagreements
from .retrieval import BM25Index, build_index, load_corpus, tokenize
from .schema import (
    Question,
    answer_from_dict,
    question_from_dict,
    read_jsonl,
    write_jsonl,
)

DEFAULT_QUESTIONS = Path("data/questions.jsonl")
DEFAULT_CORPUS = Path("data/corpus.jsonl")
DEFAULT_ANSWERS = Path("runs/answers.jsonl")
DEFAULT_LABELS = Path("runs/labels.jsonl")


def _load_questions(path: Path):
    rows = list(read_jsonl(path))
    if not rows:
        sys.exit(f"no questions found at {path}")
    return [question_from_dict(r) for r in rows]


def _load_answers(path: Path):
    return [answer_from_dict(r) for r in read_jsonl(path)]


# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    corpus = load_corpus(read_jsonl(args.corpus))
    if not corpus:
        sys.exit(f"no corpus documents found at {args.corpus}")

    existing = _load_answers(args.out) if args.append else []
    records = list(existing)
    oracle_labels = []

    for chunk_words in args.chunk_words:
        index = build_index(
            corpus,
            words_per_chunk=chunk_words,
            overlap_words=min(args.overlap, chunk_words - 1),
        )
        if args.engine == "extractive":
            engine = ExtractiveEngine(index=index, top_k=args.top_k)
            engine.name = f"extractive-k{args.top_k}-c{chunk_words}"
        elif args.engine == "lossy":
            engine = LossyEngine(index=index, top_k=args.top_k, seed=args.seed)
            engine.name = f"lossy-k{args.top_k}-c{chunk_words}"
        else:
            if not OpenAIRAGEngine.available():
                sys.exit(
                    "OPENAI_API_KEY is not set. Use --engine extractive to run "
                    "the full pipeline offline."
                )
            engine = OpenAIRAGEngine(
                index=index, model=args.model, top_k=args.top_k
            )
            engine.name = f"rag-{args.model}-k{args.top_k}-c{chunk_words}"

        variants = [(engine, None)] + [
            (wrap(engine, fault, rate=args.fault_rate, seed=args.seed), fault)
            for fault in (args.fault or [])
        ]

        for variant, fault in variants:
            print(
                f"running {variant.name} over {len(questions)} questions "
                f"({len(index.chunks)} chunks)"
            )
            errors = 0
            produced = 0
            # Counted per variant, not over the accumulated list. Summing the
            # whole list reported faults cumulatively across chunk widths, so a
            # two-width sweep claimed more injections than there were claims.
            injected = 0
            for q in questions:
                if fault is None:
                    record = variant.answer(q)
                else:
                    result = variant.inject(q)
                    record, oracle = result.record, result.oracle
                    oracle_labels.extend(oracle)
                    injected += len(oracle)
                if record.error:
                    errors += 1
                    print(f"  ! {q.id}: {record.error}", file=sys.stderr)
                records.append(record)
                produced += len(record.claims)
            note = f", {injected} faults injected" if fault is not None else ""
            print(f"  -> {produced} claims extracted, {errors} errors{note}")

    n = write_jsonl(args.out, records)
    print(f"wrote {n} answer records to {args.out}")

    if oracle_labels:
        existing = [
            lb for lb in load_labels(args.labels)
            if not lb.labeler.startswith("oracle:")
        ]
        write_jsonl(args.labels, existing + oracle_labels)
        print(f"wrote {len(oracle_labels)} oracle labels to {args.labels}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    answers = _load_answers(args.answers)
    if not answers:
        sys.exit(f"no answers at {args.answers}; run `run` first")

    judge = get_judge(args.judge)
    existing = load_labels(args.labels)
    already = {lb.claim_id for lb in existing if lb.labeler == judge.name}

    new = []
    for record in answers:
        for claim in record.claims:
            if claim.id in already:
                continue
            new.append(judge.judge(claim, record.snippets_for(claim)))

    write_jsonl(args.labels, list(existing) + new)
    print(f"{judge.name}: {len(new)} new labels ({len(already)} already present)")
    print(f"wrote {len(existing) + len(new)} labels to {args.labels}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    answers = _load_answers(args.answers)
    if not answers:
        sys.exit(f"no answers at {args.answers}; run `run` first")
    labeler = args.labeler
    if not labeler.startswith("human"):
        labeler = f"human:{labeler}"
    label_session(
        questions,
        answers,
        args.labels,
        labeler,
        limit=args.limit,
        prioritized=not args.no_priority,
        oracle_fraction=args.oracle_fraction,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    answers = _load_answers(args.answers)
    labels = load_labels(args.labels)
    text = build_report(questions, answers, labels)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_disagree(args: argparse.Namespace) -> int:
    answers = _load_answers(args.answers)
    labels = load_labels(args.labels)
    rows = disagreements(labels, answers, args.judge, limit=args.limit)
    if not rows:
        print("no human/judge disagreements found")
        return 0
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def answerability(
    questions: Sequence[Question],
    index: BM25Index,
    min_overlap: float = 0.25,
) -> list[str]:
    """Return ids of questions the corpus cannot plausibly answer.

    An unanswerable question is the most damaging kind of dataset bug here: no
    engine can cite a source that does not exist, so every engine looks
    unfaithful and the failure is attributed to the engines instead of the
    data. Authoring questions separately from documents makes it easy to do by
    accident, so `validate` fails rather than warns.

    The check is retrieval-shaped, not semantic: a question passes if some
    chunk shares at least `min_overlap` of its content tokens. That admits
    questions whose keywords appear without the answer, so this catches
    wholesale absence, not subtle unanswerability.
    """
    unanswerable: list[str] = []
    for question in questions:
        q_tokens = set(tokenize(question.text))
        if not q_tokens:
            unanswerable.append(question.id)
            continue
        best = 0.0
        for chunk, _score in index.search(question.text, top_k=5):
            overlap = len(q_tokens & set(tokenize(chunk.text))) / len(q_tokens)
            best = max(best, overlap)
        if best < min_overlap:
            unanswerable.append(question.id)
    return unanswerable


def cmd_ingest(args: argparse.Namespace) -> int:
    """Clean and dedupe a captured raw corpus into a usable one.

    Capture is an operator step (see tools/fetch_wikipedia.py) so the library
    stays network-free and a run is reproducible from the committed corpus.
    """
    rows = list(read_jsonl(args.input))
    if not rows:
        sys.exit(f"no raw rows found at {args.input}")

    docs, dropped = ingest_rows(rows)
    if not docs:
        sys.exit(f"every row was dropped; nothing written to {args.out}")

    write_jsonl(
        args.out,
        [{"url": d.url, "title": d.title, "text": d.text} for d in docs],
    )
    words = sum(len(d.text.split()) for d in docs)
    print(f"kept {len(docs)} documents ({words:,} words) -> {args.out}")
    if dropped:
        print(f"dropped {len(dropped)} (too thin or near-duplicate):")
        for url in dropped:
            print(f"  - {url}")
    return 0


def cmd_completeness(args: argparse.Namespace) -> int:
    """Score how much of each question an engine's answer actually addressed.

    Separate from faithfulness on purpose: an engine can cite every sentence
    correctly and still answer the wrong question, and the faithfulness score
    cannot see that.
    """
    questions = _load_questions(args.questions)
    answers = _load_answers(args.answers)
    if not answers:
        sys.exit(f"no answers at {args.answers}; run `run` first")

    q_by_id = {q.id: q for q in questions}
    authored = sum(1 for q in questions if q.aspects)
    if not authored:
        sys.exit(
            "no question declares `aspects`, so completeness cannot be scored. "
            "Populate aspects in the question file first."
        )
    print(f"{authored}/{len(questions)} questions declare aspects")

    by_engine: dict[str, list] = {}
    pairs_by_engine: dict[str, list] = {}
    for record in answers:
        question = q_by_id.get(record.question_id)
        if question is None or not question.aspects:
            continue
        by_engine.setdefault(record.engine, []).append(
            completeness(question, record)
        )
        pairs_by_engine.setdefault(record.engine, []).append((question, record))

    print()
    for engine in sorted(by_engine, key=lambda e: -aggregate(by_engine[e])):
        scores = by_engine[engine]
        print(f"{engine:<40} {aggregate(scores):.1%}  (n={len(scores)})")

    print()
    print("most-missed aspects:")
    flat = [p for pairs in pairs_by_engine.values() for p in pairs]
    for aspect, count in most_missed(flat, limit=args.limit):
        print(f"  {count:>4}  {aspect}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Fail loudly on dataset problems that would silently skew results."""
    from collections import Counter

    from .metrics import labels_needed

    questions = _load_questions(args.questions)
    corpus = load_corpus(read_jsonl(args.corpus))
    problems: list[str] = []

    ids = Counter(q.id for q in questions)
    for qid, count in ids.items():
        if count > 1:
            problems.append(f"duplicate question id: {qid} ({count}x)")

    urls = Counter(d.url for d in corpus)
    for url, count in urls.items():
        if count > 1:
            problems.append(f"duplicate corpus url: {url} ({count}x)")

    for d in corpus:
        if len(d.text.split()) < 20:
            problems.append(f"corpus doc too short to chunk usefully: {d.url}")

    mode_counts = Counter(m.value for q in questions for m in q.failure_modes)
    thin = [m for m, c in mode_counts.items() if c < 3]
    for m in thin:
        problems.append(f"failure mode '{m}' has only {mode_counts[m]} questions")

    if corpus:
        index = build_index(corpus, words_per_chunk=90, overlap_words=20)
        for qid in answerability(questions, index):
            problems.append(f"corpus cannot answer question: {qid}")

    no_aspects = [q.id for q in questions if not q.aspects]
    if no_aspects and len(no_aspects) != len(questions):
        problems.append(
            f"{len(no_aspects)} question(s) declare no aspects, so completeness "
            f"cannot be scored for them: {', '.join(no_aspects[:5])}"
            + (" ..." if len(no_aspects) > 5 else "")
        )

    print(f"questions: {len(questions)}   corpus docs: {len(corpus)}")
    print(f"questions with aspects declared: {len(questions) - len(no_aspects)}")
    print("failure-mode coverage:")
    for mode, count in sorted(mode_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {mode:<11} {count}")
    print()
    print("labels needed for a given confidence-interval half-width:")
    for hw in (0.10, 0.07, 0.05):
        print(f"  +/-{hw:.0%}  ->  {labels_needed(hw)} labels")

    if problems:
        print()
        print(f"{len(problems)} problem(s):", file=sys.stderr)
        for prob in problems:
            print(f"  - {prob}", file=sys.stderr)
        return 1
    print()
    print("no problems found")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfbench", description="Citation-faithfulness benchmark"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
        sp.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
        sp.add_argument("--labels", type=Path, default=DEFAULT_LABELS)

    sp = sub.add_parser("run", help="run an engine over the question set")
    sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sp.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    sp.add_argument("--out", type=Path, default=DEFAULT_ANSWERS)
    sp.add_argument(
        "--engine", choices=["extractive", "lossy", "rag"], default="extractive"
    )
    sp.add_argument("--model", default="gpt-4o-mini")
    sp.add_argument("--top-k", type=int, default=3)
    sp.add_argument(
        "--chunk-words",
        type=int,
        action="append",
        help="repeat to sweep chunk sizes, e.g. --chunk-words 90 --chunk-words 300",
    )
    sp.add_argument("--overlap", type=int, default=20)
    sp.add_argument("--append", action="store_true", help="keep existing records")
    sp.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    sp.add_argument(
        "--fault",
        action="append",
        choices=sorted(FAULTS),
        help="also run a fault-injected variant; repeatable",
    )
    sp.add_argument("--fault-rate", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("judge", help="label claims with an automated judge")
    common(sp)
    sp.add_argument("--judge", default="lexical", help="lexical | llm:gpt-4o-mini")
    sp.set_defaults(func=cmd_judge)

    sp = sub.add_parser("label", help="hand-label claims")
    common(sp)
    sp.add_argument("--labeler", required=True, help="e.g. human:aishwarya")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument(
        "--no-priority",
        action="store_true",
        help="label in raw document order instead of by information value",
    )
    sp.add_argument(
        "--oracle-fraction",
        type=float,
        default=0.35,
        help="share of the queue with known verdicts, for label variance",
    )
    sp.set_defaults(func=cmd_label)

    sp = sub.add_parser("report", help="render the leaderboard and calibration")
    common(sp)
    sp.add_argument("--out", type=Path, default=None)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("disagree", help="dump human/judge disagreements")
    common(sp)
    sp.add_argument("--judge", default="judge:lexical")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_disagree)

    sp = sub.add_parser("ingest", help="clean and dedupe a captured raw corpus")
    sp.add_argument("--input", type=Path, default=Path("data/corpus.raw.jsonl"))
    sp.add_argument("--out", type=Path, default=Path("data/corpus.real.jsonl"))
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser(
        "score-completeness", help="score answer coverage of question aspects"
    )
    sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sp.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    sp.add_argument("--limit", type=int, default=10)
    sp.set_defaults(func=cmd_completeness)

    sp = sub.add_parser("validate", help="check dataset integrity")
    sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sp.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    sp.set_defaults(func=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "chunk_words", None) is None and args.cmd == "run":
        args.chunk_words = [90]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
