"""Citation-faithfulness benchmark for answer engines.

Measures whether the source an answer engine cites actually supports the
sentence it is attached to, scored per (claim, citation) pair and decomposed
by failure mode.
"""

__version__ = "0.1.0"

from .schema import (
    AnswerRecord,
    Citation,
    Claim,
    FailureMode,
    Label,
    Question,
    Verdict,
)

__all__ = [
    "AnswerRecord",
    "Citation",
    "Claim",
    "FailureMode",
    "Label",
    "Question",
    "Verdict",
    "__version__",
]
