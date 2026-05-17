# regime_boss_template/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .categories import RegimeCategory


@dataclass
class RegimeState:
    """
    Rolling state for the regime classifier.

    All fields are intentionally generic — no domain-specific attributes.
    Extend this dataclass if your classifier needs additional history.

    Fields
    ------
    current_category : RegimeCategory
        The most recent classified category. Starts as UNDEFINED.

    category_age : int
        Number of consecutive observations in the current category.
        Resets to 0 on every category change.

    enter_conf : int
        Consecutive observations supporting entry into a new category.
        Used for hysteresis — don't switch until this reaches threshold.

    exit_conf : int
        Consecutive observations supporting exit from current category.
        Used for hysteresis — don't exit until this reaches threshold.

    signal_history : List[float]
        Rolling window of scalar signal values (domain-defined).
        Use for trend detection, divergence baseline, etc.
        Managed externally by RegimeBoss.update().

    category_history : List[str]
        Rolling window of category labels (as strings) for walk-forward
        evaluation and compare_windows baseline.

    last_window_id : str
        Dedup guard. Stores the identifier of the last processed window.
        Prevents double-counting when the same window is evaluated twice
        (e.g. in a polling loop that runs faster than window advancement).

    last_epoch_id : str
        Epoch/session boundary tracker. When this changes, rolling histories
        are cleared. Set this to your day/session/conversation identifier.

    Notes
    -----
    For miu's port: last_epoch_id maps naturally to conversation session IDs.
    signal_history is the right place to buffer ERIP vector norms or scalar
    projections for divergence baseline computation.
    """

    current_category: RegimeCategory = RegimeCategory.UNDEFINED
    category_age: int = 0

    # Hysteresis counters
    enter_conf: int = 0
    exit_conf: int = 0

    # Rolling history
    signal_history: List[float] = field(default_factory=list)
    category_history: List[str] = field(default_factory=list)

    # Dedup and epoch tracking
    last_window_id: str = ""
    last_epoch_id: str = ""

    def reset_for_new_epoch(self) -> None:
        """
        Clear rolling histories on epoch/session boundary.

        Call this (or set epoch_id in RegimeBoss.update()) whenever
        your time window resets — new day, new session, new conversation.

        Intentionally does NOT reset current_category — the classifier
        should make that decision based on fresh data, not the reset.
        """
        self.signal_history.clear()
        self.category_history.clear()
        self.enter_conf = 0
        self.exit_conf  = 0
        self.category_age = 0
        self.last_window_id = ""

    def append_signal(self, value: float, max_len: int = 100) -> None:
        """Append to signal_history with automatic truncation."""
        self.signal_history.append(value)
        if len(self.signal_history) > max_len:
            self.signal_history = self.signal_history[-max_len:]

    def append_category(self, category: RegimeCategory, max_len: int = 50) -> None:
        """Append to category_history with automatic truncation."""
        self.category_history.append(category.value)
        if len(self.category_history) > max_len:
            self.category_history = self.category_history[-max_len:]
