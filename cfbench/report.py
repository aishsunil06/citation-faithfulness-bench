"""Render the leaderboard, the failure taxonomy, and judge calibration.

Design rule: never print a judge-derived score without printing that judge's
agreement with humans next to it. A faithfulness number from an uncalibrated
judge is not a result, and separating the two invites exactly the misreading
this benchmark exists to prevent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

from .metrics import (
    Agreement,
    agreement,
    bootstrap_ci,
    detection_rates,
    score,
    score_by_failure_mode,
)
from .schema import AnswerRecord, FailureMode, Label, Question, Verdict


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_no data_\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append(
            "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"
        )
    return "\n".join(out) + "\n"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_report(
    questions: Sequence[Question],
    answers: Sequence[AnswerRecord],
    labels: Sequence[Label],
    *,
    title: str = "Citation Faithfulness Benchmark",
) -> str:
    claim_to_question = {
        claim.id: rec.question_id for rec in answers for claim in rec.claims
    }
    claim_to_engine = {
        claim.id: rec.engine for rec in answers for claim in rec.claims
    }

    human = [lb for lb in labels if lb.is_human]
    judges: dict[str, list[Label]] = defaultdict(list)
    oracle: list[Label] = []
    for lb in labels:
        if lb.is_human:
            continue
        if lb.labeler.startswith("oracle:"):
            oracle.append(lb)
        else:
            judges[lb.labeler].append(lb)

    lines: list[str] = [f"# {title}\n"]
    lines.append(
        f"- Questions: **{len(questions)}**\n"
        f"- Engine runs: **{len(answers)}**\n"
        f"- Claims extracted: **{len(claim_to_question)}**\n"
        f"- Human labels: **{len(human)}**\n"
        f"- Judges: **{len(judges)}**\n"
    )

    # ---- judge calibration first, because it gates everything below --------
    lines.append("\n## Judge calibration (vs human labels)\n")
    if not human:
        lines.append(
            "_No human labels yet. Run `label` before trusting any judge score._\n"
        )
        calibration: dict[str, Agreement] = {}
    else:
        calibration = {name: agreement(human, lbs) for name, lbs in judges.items()}
        rows = []
        for name, agr in sorted(
            calibration.items(), key=lambda kv: -kv[1].cohens_kappa
        ):
            rows.append([
                name,
                agr.n_compared,
                _pct(agr.accuracy),
                f"{agr.cohens_kappa:.3f}",
                _kappa_verdict(agr.cohens_kappa),
            ])
        lines.append(_table(
            ["judge", "n", "accuracy", "cohen's kappa", "reading"], rows
        ))
        lines.append(
            "\nKappa is chance-corrected. These verdicts skew heavily toward "
            "`supported`, so a judge that always answers `supported` can post "
            "high accuracy with a kappa near zero. Read the kappa column.\n"
        )

    # ---- leaderboard, per labeler -----------------------------------------
    lines.append("\n## Faithfulness leaderboard\n")
    label_sets: dict[str, list[Label]] = {}
    if human:
        label_sets["human"] = human
    label_sets.update(judges)

    for labeler, lbs in label_sets.items():
        note = ""
        if labeler in calibration:
            note = f" (kappa vs human {calibration[labeler].cohens_kappa:.3f})"
        lines.append(f"\n### Labels: `{labeler}`{note}\n")

        by_engine: dict[str, list[Label]] = defaultdict(list)
        for lb in lbs:
            engine = claim_to_engine.get(lb.claim_id)
            if engine:
                by_engine[engine].append(lb)

        rows = []
        for engine, group in sorted(
            by_engine.items(), key=lambda kv: -score(kv[1]).strict
        ):
            sc = score(group)
            lo, hi = bootstrap_ci(group, "strict")
            rows.append([
                engine,
                sc.n_claims,
                _pct(sc.strict),
                f"[{_pct(lo)}, {_pct(hi)}]",
                _pct(sc.lenient),
                _pct(sc.uncited_rate),
            ])
        lines.append(_table(
            ["engine", "claims", "strict", "95% CI", "lenient", "uncited"], rows
        ))
        lines.append(
            "\nIntervals are a percentile bootstrap over claims. Where two "
            "engines' intervals overlap, this run does not distinguish them.\n"
        )

    # ---- fault-injection validation ---------------------------------------
    if oracle and judges:
        lines.append("\n## Fault-injection validation\n")
        lines.append(
            "Faults were injected into otherwise-faithful answers, so the "
            "correct verdict is known by construction and needs no human "
            "reading. This measures the floor of a judge's sensitivity: a "
            "judge that misses synthetic corruption will certainly miss "
            "subtler real failures. It does not replace human labelling, "
            "because injected faults are unambiguous and real ones often "
            "are not.\n"
        )
        for judge_name, judge_labels in sorted(judges.items()):
            rates = detection_rates(oracle, judge_labels)
            if not rates:
                continue
            lines.append(f"\n### `{judge_name}`\n")
            rows = []
            for fault, det in sorted(rates.items()):
                said = ", ".join(
                    f"{v.value} x{c}"
                    for v, c in sorted(
                        det.confusions.items(), key=lambda kv: -kv[1]
                    )
                )
                rows.append([fault, det.n, det.n_detected, _pct(det.rate), said])
            lines.append(_table(
                ["injected fault", "n", "caught", "detection", "judge said"], rows
            ))

    # ---- failure-mode decomposition ---------------------------------------
    primary = human if human else (next(iter(judges.values())) if judges else [])
    primary_name = "human" if human else (next(iter(judges), "none"))
    if primary:
        lines.append(f"\n## Failure modes (labels: `{primary_name}`)\n")
        by_engine_mode: dict[str, dict[FailureMode, float]] = {}
        for engine in sorted({claim_to_engine[c] for c in claim_to_engine}):
            subset = [
                lb for lb in primary if claim_to_engine.get(lb.claim_id) == engine
            ]
            if not subset:
                continue
            modes = score_by_failure_mode(subset, claim_to_question, questions)
            by_engine_mode[engine] = {m: s.strict for m, s in modes.items()}

        all_modes = [m for m in FailureMode if any(
            m in v for v in by_engine_mode.values()
        )]
        rows = []
        for engine, modes in by_engine_mode.items():
            rows.append(
                [engine] + [
                    _pct(modes[m]) if m in modes else "-" for m in all_modes
                ]
            )
        lines.append(_table(
            ["engine"] + [m.value for m in all_modes], rows
        ))
        lines.append(
            "\nColumns are strict faithfulness within each failure mode. A "
            "question tagged with two modes counts in both, so columns do not "
            "partition the claim set.\n"
        )

        lines.append("\n## Verdict mix\n")
        rows = []
        for engine in sorted(by_engine_mode):
            subset = [
                lb for lb in primary if claim_to_engine.get(lb.claim_id) == engine
            ]
            counts = Counter(lb.verdict for lb in subset)
            rows.append([engine] + [counts.get(v, 0) for v in Verdict])
        lines.append(_table(["engine"] + [v.value for v in Verdict], rows))

    return "\n".join(lines)


def _kappa_verdict(k: float) -> str:
    if k < 0.2:
        return "poor"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "near-perfect"


def disagreements(
    labels: Sequence[Label],
    answers: Sequence[AnswerRecord],
    judge_name: str,
    limit: int = 20,
) -> list[dict]:
    """Claims where a judge and a human disagreed, for error analysis.

    This is the list that turns a score into a finding: reading the top
    disagreements is how the judge prompt gets fixed and how the failure
    taxonomy gets written.
    """
    claim_text = {c.id: c.text for rec in answers for c in rec.claims}
    human = {lb.claim_id: lb for lb in labels if lb.is_human}
    judge = {lb.claim_id: lb for lb in labels if lb.labeler == judge_name}

    out: list[dict] = []
    for claim_id in sorted(set(human) & set(judge)):
        if human[claim_id].verdict is judge[claim_id].verdict:
            continue
        out.append({
            "claim_id": claim_id,
            "claim": claim_text.get(claim_id, ""),
            "human": human[claim_id].verdict.value,
            "judge": judge[claim_id].verdict.value,
            "judge_rationale": judge[claim_id].rationale,
            "human_rationale": human[claim_id].rationale,
        })
        if len(out) >= limit:
            break
    return out
