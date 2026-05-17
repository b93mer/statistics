# regime_boss_template/observer.py
from __future__ import annotations

from abc import ABC, abstractmethod

from .categories import RegimeCategory
from .result import RegimeResult


class RegimeObserver(ABC):
    """
    Event hook for regime transitions.

    Not storage — this is where you wire into an external event stream.

    For miu's port: override on_regime_change to call MMA's emit_event.
    The observer is decoupled from the classifier so you can attach
    multiple observers (logging, event bus, metrics) independently.

    Usage
    -----
    class MMAObserver(RegimeObserver):
        def on_regime_change(self, before, after, result):
            mma.emit_event("regime_shift", {
                "from": before.value,
                "to": after.value,
                "confidence": result.confidence,
                "signal": result.reasoning["primary_signal"],
            })

    boss.add_observer(MMAObserver())
    """

    @abstractmethod
    def on_regime_change(
        self,
        category_before: RegimeCategory,
        category_after: RegimeCategory,
        result: RegimeResult,
    ) -> None:
        """
        Called when the classified regime changes between evaluations.

        Parameters
        ----------
        category_before : RegimeCategory
            The regime category from the previous evaluation.
        category_after : RegimeCategory
            The newly classified regime category.
        result : RegimeResult
            Full result for the new regime — confidence and reasoning
            are available here for downstream consumers.
        """
        ...


class LoggingObserver(RegimeObserver):
    """
    Minimal concrete observer — prints regime changes to stdout.

    Replace with your own observer. This exists so the template
    runs without requiring any external dependencies.
    """

    def on_regime_change(
        self,
        category_before: RegimeCategory,
        category_after: RegimeCategory,
        result: RegimeResult,
    ) -> None:
        print(
            f"[RegimeObserver] {category_before.value} → {category_after.value}"
            f" | conf={result.confidence:.3f}"
            f" | signal={result.reasoning.get('primary_signal')}"
        )
