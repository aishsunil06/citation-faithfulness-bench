"""Automated judges that assign a Verdict to a (claim, cited-snippet) pair.

Two judges ship deliberately:

* `LexicalJudge` - no API key, no network. A rule-based baseline built from
  token overlap plus numeric and negation checks. It exists to be *beaten*:
  reporting an LLM judge's kappa without a cheap baseline hides how much of
  the agreement was available for free.
* `LLMJudge` - asks a model, returns structured output. Stronger, costs money,
  and its agreement with humans must be measured before its numbers are
  trusted.

Neither is authoritative. Human labels are the ground truth; judges are
instruments calibrated against them (see metrics.agreement).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from .retrieval import tokenize
from .schema import Claim, Label, Verdict

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")
_NEGATION = frozenset({
    "not", "no", "never", "none", "neither", "nor", "without", "cannot",
    "didn't", "doesn't", "isn't", "wasn't", "weren't", "won't", "hasn't",
    "haven't", "unable", "failed", "declined", "denied", "rejected",
})


class Judge(Protocol):
    name: str

    def judge(self, claim: Claim, snippets: list[str]) -> Label: ...


def _normalise_number(tok: str) -> str:
    """Normalise so '12%', '12.0' and '12' compare equal."""
    tok = tok.rstrip("%")
    try:
        val = float(tok)
    except ValueError:
        return tok
    return str(int(val)) if val == int(val) else str(val)


def _numbers(text: str) -> set[str]:
    return {_normalise_number(m.group()) for m in _NUMBER.finditer(text)}


def _has_negation(text: str) -> bool:
    return bool(_NEGATION & set(tokenize(text, drop_stopwords=False)))


@dataclass
class LexicalJudge:
    """Token-overlap baseline with numeric and negation guards.

    Overlap is measured as the fraction of the *claim's* content tokens present
    in the snippet, not Jaccard: a long snippet that happens to contain every
    word of a short claim is exactly the supported case, and Jaccard would
    penalise it for length.
    """

    supported_threshold: float = 0.75
    partial_threshold: float = 0.40
    name: str = "judge:lexical"

    def judge(self, claim: Claim, snippets: list[str]) -> Label:
        if claim.is_uncited or not snippets:
            return Label(
                claim_id=claim.id,
                verdict=Verdict.UNCITED,
                labeler=self.name,
                rationale="claim carried no resolvable citation",
            )

        claim_tokens = set(tokenize(claim.text))
        if not claim_tokens:
            return Label(
                claim_id=claim.id,
                verdict=Verdict.PARTIAL,
                labeler=self.name,
                rationale="claim had no content tokens to compare",
            )

        best_overlap = 0.0
        best_snippet = ""
        for snippet in snippets:
            overlap = len(claim_tokens & set(tokenize(snippet))) / len(claim_tokens)
            if overlap > best_overlap:
                best_overlap, best_snippet = overlap, snippet

        joined = " ".join(snippets)

        # A number asserted in the claim but absent from every snippet is the
        # single most reliable unsupported signal, so it overrides overlap.
        claim_nums = _numbers(claim.text)
        snippet_nums = _numbers(joined)
        missing_nums = claim_nums - snippet_nums
        if claim_nums and missing_nums:
            # High lexical overlap plus a swapped figure means the source is
            # talking about this very fact and stating a different number, i.e.
            # contradiction rather than mere absence of support.
            contradicts = (
                best_overlap >= self.partial_threshold and bool(snippet_nums)
            )
            return Label(
                claim_id=claim.id,
                verdict=(
                    Verdict.CONTRADICTED if contradicts else Verdict.UNSUPPORTED
                ),
                labeler=self.name,
                rationale=(
                    f"numbers {sorted(missing_nums)} in the claim appear in no "
                    f"cited snippet (overlap {best_overlap:.2f})"
                ),
            )

        # Polarity mismatch: one side negates and the other does not.
        if _has_negation(claim.text) != _has_negation(best_snippet):
            return Label(
                claim_id=claim.id,
                verdict=Verdict.PARTIAL,
                labeler=self.name,
                rationale=(
                    f"negation polarity differs between claim and snippet "
                    f"(overlap {best_overlap:.2f})"
                ),
            )

        if best_overlap >= self.supported_threshold:
            verdict = Verdict.SUPPORTED
        elif best_overlap >= self.partial_threshold:
            verdict = Verdict.PARTIAL
        else:
            verdict = Verdict.UNSUPPORTED

        return Label(
            claim_id=claim.id,
            verdict=verdict,
            labeler=self.name,
            rationale=f"content-token overlap {best_overlap:.2f}",
        )


JUDGE_SYSTEM_PROMPT = """\
You assess whether a cited source supports a claim. You are grading the \
citation, not the claim's truth: a true claim cited to an irrelevant source is \
UNSUPPORTED.

Reply with JSON only: {"verdict": "...", "rationale": "..."}

verdict must be exactly one of:
- "supported": the snippet states the claim, including any numbers, dates, and
  entities. Paraphrase is fine; changed facts are not.
- "partial": the snippet is relevant but weaker, narrower, or differs in a
  detail such as a number, date, scope, or qualifier.
- "unsupported": the snippet simply does not support the claim. It is about
  something else, or is silent on the point.
- "contradicted": the snippet addresses this exact point and states the
  opposite. Use this when the snippet gives a different number, date, or
  polarity for the very fact the claim asserts. Prefer this over
  "unsupported" whenever the source actively disagrees rather than being
  merely irrelevant.

Do not grade whether the claim answers any particular question. Only whether
the cited snippet supports the sentence as written.

rationale: one sentence, naming the specific mismatch if there is one.
"""


@dataclass
class LLMJudge:
    """LLM-backed judge. Requires OPENAI_API_KEY in the environment."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_snippet_chars: int = 4000
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"judge:{self.model}"

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "LLMJudge needs the openai package: pip install openai"
            ) from exc
        if not self.available():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use LexicalJudge for offline runs."
            )
        return OpenAI()

    def _user_prompt(self, claim: Claim, snippets: list[str]) -> str:
        blocks = "\n\n".join(
            f"[source {i}]\n{s[:self.max_snippet_chars]}"
            for i, s in enumerate(snippets, 1)
        )
        return f"CLAIM:\n{claim.text}\n\nCITED SOURCES:\n{blocks}"

    def judge(self, claim: Claim, snippets: list[str]) -> Label:
        if claim.is_uncited or not snippets:
            return Label(
                claim_id=claim.id,
                verdict=Verdict.UNCITED,
                labeler=self.name,
                rationale="claim carried no resolvable citation",
            )

        response = self._client().chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": self._user_prompt(claim, snippets)},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return self._parse(claim, raw)

    def _parse(self, claim: Claim, raw: str) -> Label:
        try:
            data = json.loads(raw)
            verdict = Verdict(str(data["verdict"]).strip().lower())
            rationale = str(data.get("rationale", ""))[:500]
        except (json.JSONDecodeError, KeyError, ValueError):
            # A malformed judge response is recorded as PARTIAL rather than
            # dropped, so the run stays complete and the failure is visible in
            # the rationale column instead of silently shrinking the sample.
            return Label(
                claim_id=claim.id,
                verdict=Verdict.PARTIAL,
                labeler=self.name,
                rationale=f"unparseable judge response: {raw[:200]!r}",
            )
        return Label(
            claim_id=claim.id,
            verdict=verdict,
            labeler=self.name,
            rationale=rationale,
        )


def get_judge(name: str) -> Judge:
    """Resolve a judge by CLI name."""
    if name in ("lexical", "baseline"):
        return LexicalJudge()
    if name.startswith("llm"):
        _, _, model = name.partition(":")
        return LLMJudge(model=model or "gpt-4o-mini")
    raise ValueError(f"unknown judge {name!r}; try 'lexical' or 'llm:gpt-4o-mini'")
