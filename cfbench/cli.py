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

    for chunk_words in args.chunk_words:
        index = build_index(
            corpus,
            words_per_chunk=chunk_words,
            overlap_words=min(args.overlap, chunk_words - 1),
        )
        if args.engine == "extractive":
            engine = ExtractiveEngine(index=index, top_k=args.top_k)
            engine.name = f"extractive-k{args.top_k}-c{chunk_words}"
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

        print(
            f"running {engine.name} over {len(questions)} questions "
            f"({len(index.chunks)} chunks)"
        )
        errors = 0
        for q in questions:
            record = engine.answer(q)
            if record.error:
                errors += 1
                print(f"  ! {q.id}: {record.error}", file=sys.stderr)
            records.append(record)
        claims = sum(len(r.claims) for r in records if r.engine == engine.name)
        print(f"  -> {claims} claims extracted, {errors} errors")

    n = write_jsonl(args.out, records)
    print(f"wrote {n} answer records to {args.out}")
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
        questions, answers, args.labels, labeler, limit=args.limit
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
    sp.add_argument("--engine", choices=["extractive", "rag"], default="extractive")
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
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("judge", help="label claims with an automated judge")
    common(sp)
    sp.add_argument("--judge", default="lexical", help="lexical | llm:gpt-4o-mini")
    sp.set_defaults(func=cmd_judge)

    sp = sub.add_parser("label", help="hand-label claims")
    common(sp)
    sp.add_argument("--labeler", required=True, help="e.g. human:aishwarya")
    sp.add_argument("--limit", type=int, default=None)
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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "chunk_words", None) is None and args.cmd == "run":
        args.chunk_words = [90]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
