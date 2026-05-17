# regime_boss_template/categories.py
from __future__ import annotations

from enum import Enum


class RegimeCategory(Enum):
    """
    Domain-neutral regime categories.

    Map these to your domain's semantics — e.g. for miu's port:
        STABLE        → sustained coherent relational state
        VOLATILE      → high-variance / emotionally turbulent state
        TRANSITIONING → active shift between states (in-progress)
        DECAYING      → previously ordered state breaking down
        EMERGING      → new pattern forming, direction not yet confirmed
        UNDEFINED     → insufficient data, blocked, or warmup period

    Keep to 5–7 max. Resist adding domain-specific names here —
    domain semantics belong in the subclass that implements classify().
    """
    STABLE        = "STABLE"
    VOLATILE      = "VOLATILE"
    TRANSITIONING = "TRANSITIONING"
    DECAYING      = "DECAYING"
    EMERGING      = "EMERGING"
    UNDEFINED     = "UNDEFINED"
