#!/usr/bin/env python3
"""Export a stratified sample of claims for judging, and import verdicts back.

Why a sample rather than everything: the lexical judge is free and scores every
claim, but an expensive judge cannot. The honest design is a full leaderboard
from the cheap judge plus the expensive judge on a sample large enough to bound
the cheap judge's error. Sampling must be stratified, because a random sample
of a fault-heavy run is mostly injected faults, and the disagreements that
matter live in the clean engines' borderline claims.

Export:
    python tools/sample_claims.py export --n 40 --out runs/sample.json
Import (verdicts as {claim_id: [verdict, rationale]}):
    python tools/sample_claims.py import --labeler judge:claude-opus-5 \
        --verdicts runs/verdicts.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfbench.labeling import load_labels  # noqa: E402
from cfbench.schema import (  # noqa: E402
    Label,
    Verdict,
    answer_from_dict,
    question_from_dict,
    read_jsonl,
    write_jsonl,
)

ANSWERS = Path("runs/answers.jsonl")
LABELS = Path("runs/labels.jsonl")
QUESTIONS = Path("data/questions.real.jsonl")


def cmd_export(args: argparse.Namespace) -> int:
    questions = {q.id: q for q in (question_from_dict(r) for r in read_jsonl(args.questions))}
    answers = [answer_from_dict(r) for r in read_jsonl(args.answers)]
    if not answers:
        sys.exit(f"no answers at {args.answers}")

    labels = load_labels(args.labels)
    oracle = {lb.claim_id for lb in labels if lb.labeler.startswith("oracle:")}

    # Stratify over (engine family, primary failure mode) so the sample spans
    # clean and faulty engines and every mode, rather than piling onto
    # whichever combination produced the most claims.
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for record in answers:
        question = questions.get(record.question_id)
        if question is None:
            continue
        family = "faulty" if "+" in record.engine else "clean"
        mode = question.failure_modes[0].value
        for claim in record.claims:
            buckets[(family, mode)].append((record, claim))

    rng = random.Random(args.seed)
    chosen: list[tuple] = []
    per_bucket = max(1, args.n // max(1, len(buckets)))
    for key in sorted(buckets):
        group = buckets[key]
        rng.shuffle(group)
        chosen.extend(group[:per_bucket])

    rng.shuffle(chosen)
    chosen = chosen[: args.n]

    out = []
    for record, claim in chosen:
        question = questions.get(record.question_id)
        out.append({
            "claim_id": claim.id,
            "engine": record.engine,
            "question": question.text if question else "",
            "failure_modes": [m.value for m in question.failure_modes] if question else [],
            "claim": claim.text,
            "cited_snippets": record.snippets_for(claim),
            "has_oracle": claim.id in oracle,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    n_oracle = sum(1 for row in out if row["has_oracle"])
    print(f"exported {len(out)} claims to {args.out}")
    print(f"  {n_oracle} already have an oracle verdict, {len(out) - n_oracle} do not")
    print(f"  buckets covered: {len(buckets)}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    # This check is first on purpose. Guarding after the file is parsed means a
    # malformed verdicts file crashes before the guard ever runs, which is not
    # a guard at all. Human labels are the only ground truth in this project;
    # a model writing one invalidates every calibration figure derived from it.
    if args.labeler.startswith("human"):
        sys.exit(
            "refusing to import verdicts under a human labeler; human labels "
            "are the ground truth that judges are measured against"
        )

    payload = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        sys.exit("verdicts file must be an object mapping claim_id -> [verdict, rationale]")

    existing = [lb for lb in load_labels(args.labels) if lb.labeler != args.labeler]
    new: list[Label] = []
    bad: list[str] = []
    for claim_id, value in payload.items():
        verdict_raw, _, rationale = (
            (value[0], None, value[1]) if isinstance(value, list) else (value, None, "")
        )
        try:
            verdict = Verdict(str(verdict_raw).strip().lower())
        except ValueError:
            bad.append(f"{claim_id}: unknown verdict {verdict_raw!r}")
            continue
        new.append(
            Label(
                claim_id=claim_id,
                verdict=verdict,
                labeler=args.labeler,
                rationale=str(rationale)[:400],
            )
        )

    if args.labeler.startswith("human"):
        sys.exit(
            "refusing to import model-produced verdicts under a human labeler; "
            "human labels are the ground truth judges are measured against"
        )

    write_jsonl(args.labels, existing + new)
    print(f"imported {len(new)} labels as {args.labeler}")
    if bad:
        print(f"{len(bad)} rejected:", file=sys.stderr)
        for line in bad:
            print(f"  - {line}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="sample_claims")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("export")
    sp.add_argument("--n", type=int, default=40)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--questions", type=Path, default=QUESTIONS)
    sp.add_argument("--answers", type=Path, default=ANSWERS)
    sp.add_argument("--labels", type=Path, default=LABELS)
    sp.add_argument("--out", type=Path, default=Path("runs/sample.json"))
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("import")
    sp.add_argument("--labeler", required=True)
    sp.add_argument("--verdicts", type=Path, required=True)
    sp.add_argument("--labels", type=Path, default=LABELS)
    sp.set_defaults(func=cmd_import)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
