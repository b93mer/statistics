# regime_boss_template/result.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .categories import RegimeCategory


@dataclass
class RegimeResult:
    """
    Output of a single regime classification pass.

    Fields
    ------
    category : RegimeCategory
        The classified regime for this observation.

    confidence : float
        Classifier confidence, in [0.0, 1.0].
        0.0 = no signal / UNDEFINED.
        1.0 = maximum confidence (use sparingly).

    reasoning : dict
        Structured explanation of the classification.
        Always contains three keys:

        primary_signal : Any
            The single strongest contributing feature or signal name.
            Keep this human-readable — it's your audit trail.

        contributing_signals : List[Any]
            Supporting signals that influenced the result, in
            descending order of contribution. Can be empty.

        confidence_factors : Dict[str, float]
            Named breakdown of what drove the confidence score.
            Example: {"signal_strength": 0.8, "window_agreement": 0.6}
            Values are not required to sum to 1.0.

    Notes
    -----
    For miu's port: primary_signal is the natural hook for logging
    which ERIP dimension drove the classification. confidence_factors
    maps directly to Bayesian posterior weights if you want the reasoning
    dict to double as a calibration artifact.
    """

    category: RegimeCategory
    confidence: float
    reasoning: Dict[str, Any] = field(default_factory=lambda: {
        "primary_signal": None,
        "contributing_signals": [],
        "confidence_factors": {},
    })

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        # Ensure reasoning always has all three required keys.
        self.reasoning.setdefault("primary_signal", None)
        self.reasoning.setdefault("contributing_signals", [])
        self.reasoning.setdefault("confidence_factors", {})

    @classmethod
    def undefined(cls, reason: str = "insufficient_data") -> "RegimeResult":
        """Convenience constructor for blocked / no-data cases."""
        return cls(
            category=RegimeCategory.UNDEFINED,
            confidence=0.0,
            reasoning={
                "primary_signal": reason,
                "contributing_signals": [],
                "confidence_factors": {},
            },
        )
