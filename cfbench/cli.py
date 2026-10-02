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

from .engines import ExtractiveEngine, OpenAIRAGEngine
from .engines.lossy import LossyEngine
from .engines.faulty import FAULTS, wrap
from .judge import get_judge
from .labeling import label_session, load_labels
from .report import build_report, disagreements
from .retrieval import build_index, load_corpus
from .schema import (
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
            for q in questions:
                if fault is None:
                    record = variant.answer(q)
                else:
                    result = variant.inject(q)
                    record, oracle = result.record, result.oracle
                    oracle_labels.extend(oracle)
                if record.error:
                    errors += 1
                    print(f"  ! {q.id}: {record.error}", file=sys.stderr)
                records.append(record)
                produced += len(record.claims)
            note = ""
            if fault is not None:
                injected = sum(
                    1 for lb in oracle_labels if lb.labeler == f"oracle:{fault}"
                )
                note = f", {injected} faults injected"
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

    print(f"questions: {len(questions)}   corpus docs: {len(corpus)}")
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
