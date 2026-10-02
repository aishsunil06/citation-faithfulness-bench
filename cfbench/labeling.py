"""Terminal tool for hand-labelling (claim, citation) pairs.

Human labels are the ground truth the judges are calibrated against, so this
has to be fast enough to get through hundreds of pairs: single-keystroke
verdicts, resumable, and append-only.

Deliberately *not* shown during labelling: any judge's verdict. Seeing the
machine's answer first anchors the human and inflates the agreement number the
whole benchmark rests on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

from .schema import (
    AnswerRecord,
    Label,
    Question,
    Verdict,
    label_from_dict,
    read_jsonl,
    write_jsonl,
)

KEYMAP = {
    "s": Verdict.SUPPORTED,
    "p": Verdict.PARTIAL,
    "u": Verdict.UNSUPPORTED,
    "n": Verdict.UNCITED,
}

_PROMPT = (
    "[s]upported  [p]artial  [u]nsupported  [n]o-citation  "
    "[k]skip  [q]uit > "
)


def load_labels(path: str | Path) -> list[Label]:
    return [label_from_dict(row) for row in read_jsonl(path)]


def pending_claims(
    answers: Sequence[AnswerRecord],
    existing: Iterable[Label],
    labeler: str,
) -> list[tuple[AnswerRecord, object]]:
    """Claims this labeler has not yet judged, in a stable order."""
    done = {lb.claim_id for lb in existing if lb.labeler == labeler}
    out = []
    for rec in answers:
        for claim in rec.claims:
            if claim.id not in done:
                out.append((rec, claim))
    return out


def label_session(
    questions: Sequence[Question],
    answers: Sequence[AnswerRecord],
    labels_path: str | Path,
    labeler: str,
    *,
    limit: int | None = None,
    prioritized: bool = True,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> int:
    """Run an interactive labelling pass. Returns the number of new labels.

    `prioritized` orders the queue by expected information per label and drops
    claims whose verdict is already known by construction. Pass False for the
    raw document order, which is only useful for auditing coverage.

    `input_fn` and `print_fn` are injected so the loop is testable without a
    terminal.
    """
    labels_path = Path(labels_path)
    existing = load_labels(labels_path)
    if prioritized:
        queue = prioritize(questions, answers, existing, labeler)
    else:
        queue = pending_claims(answers, existing, labeler)
    if limit is not None:
        queue = queue[:limit]

    if not queue:
        print_fn("Nothing left to label for this labeler.")
        return 0

    q_by_id = {q.id: q for q in questions}
    new: list[Label] = []

    print_fn(f"\n{len(queue)} claims to label as '{labeler}'.")
    print_fn("Judge verdicts are hidden on purpose, to keep labels independent.\n")

    for i, (record, claim) in enumerate(queue, 1):
        question = q_by_id.get(record.question_id)
        snippets = record.snippets_for(claim)

        print_fn("=" * 78)
        print_fn(f"[{i}/{len(queue)}]  engine={record.engine}")
        if question is not None:
            modes = ", ".join(m.value for m in question.failure_modes)
            print_fn(f"QUESTION ({modes}): {question.text}")
        print_fn(f"\nCLAIM: {claim.text}\n")

        if not snippets:
            print_fn("  (no citation attached to this claim)")
        for j, snippet in enumerate(snippets, 1):
            print_fn(f"  --- cited source {j} ---")
            print_fn(f"  {snippet.strip()[:900]}")
        print_fn("")

        while True:
            choice = input_fn(_PROMPT).strip().lower()
            if choice == "q":
                _flush(labels_path, existing, new)
                print_fn(f"\nSaved {len(new)} labels. Resume any time.")
                return len(new)
            if choice == "k":
                break
            if choice in KEYMAP:
                rationale = input_fn("  why (optional) > ").strip()
                new.append(
                    Label(
                        claim_id=claim.id,
                        verdict=KEYMAP[choice],
                        labeler=labeler,
                        rationale=rationale,
                    )
                )
                break
            print_fn("  unrecognised key")

    _flush(labels_path, existing, new)
    print_fn(f"\nDone. Saved {len(new)} new labels to {labels_path}")
    return len(new)


def _flush(path: Path, existing: Sequence[Label], new: Sequence[Label]) -> None:
    """Rewrite the store with existing plus new labels.

    Full rewrite rather than append so a partially-written line from an
    interrupted run cannot corrupt the file.
    """
    write_jsonl(path, list(existing) + list(new))


# --------------------------------------------------------------------------
# Label prioritisation
# --------------------------------------------------------------------------
#
# Human labelling is the scarce resource, so queue order matters more than
# queue length. Three rules, in priority order:
#
# 1. Skip claims whose verdict is already known by construction. Injected
#    faults carry an oracle label; asking a human to re-derive it buys nothing
#    and, on the current run, would consume 60% of the budget.
# 2. Prefer ambiguous claims. A claim whose overlap sits near a judge's
#    decision boundary is where judge and human are most likely to diverge,
#    which is exactly what calibration needs to measure. Claims at overlap 0.02
#    or 0.98 are ones every judge already gets right.
# 3. Spread across failure modes. Round-robin so a budget of 50 labels does not
#    land entirely on `numeric`, leaving `contested` with nothing and no
#    per-mode breakdown possible.


def claim_ambiguity(claim, snippets: Sequence[str]) -> float:
    """0 = clear-cut, 1 = sitting exactly on a judge decision boundary."""
    from .judge import LexicalJudge
    from .retrieval import tokenize

    if not snippets or claim.is_uncited:
        return 0.0          # uncited is unambiguous; the oracle handles it

    claim_tokens = set(tokenize(claim.text))
    if not claim_tokens:
        return 0.0

    overlap = max(
        len(claim_tokens & set(tokenize(s))) / len(claim_tokens) for s in snippets
    )

    judge = LexicalJudge()
    boundaries = (judge.partial_threshold, judge.supported_threshold)
    distance = min(abs(overlap - b) for b in boundaries)
    # Normalise by the widest possible distance from any boundary.
    return max(0.0, 1.0 - distance / 0.5)


def prioritize(
    questions: Sequence[Question],
    answers: Sequence[AnswerRecord],
    existing: Sequence[Label],
    labeler: str,
    *,
    exclude_oracle: bool = True,
) -> list[tuple[AnswerRecord, object]]:
    """Order the pending queue by expected information per label."""
    oracle_claims = {
        lb.claim_id for lb in existing if lb.labeler.startswith("oracle:")
    }
    q_by_id = {q.id: q for q in questions}

    pending = pending_claims(answers, existing, labeler)
    if exclude_oracle:
        pending = [
            (rec, claim) for rec, claim in pending
            if claim.id not in oracle_claims
        ]

    # Bucket by failure mode, each bucket sorted by ambiguity descending.
    buckets: dict[str, list[tuple[float, AnswerRecord, object]]] = {}
    for rec, claim in pending:
        question = q_by_id.get(rec.question_id)
        modes = (
            [m.value for m in question.failure_modes] if question else ["unknown"]
        )
        amb = claim_ambiguity(claim, rec.snippets_for(claim))
        # A claim counts toward its rarest mode only, so round-robin stays even.
        key = min(modes)
        buckets.setdefault(key, []).append((amb, rec, claim))

    for group in buckets.values():
        group.sort(key=lambda item: -item[0])

    # Round-robin across modes.
    ordered: list[tuple[AnswerRecord, object]] = []
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    while any(buckets[k] for k in order):
        for key in order:
            if buckets[key]:
                _amb, rec, claim = buckets[key].pop(0)
                ordered.append((rec, claim))
    return ordered
