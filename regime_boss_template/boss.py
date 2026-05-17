# regime_boss_template/boss.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from .categories import RegimeCategory
from .config import RegimeBossConfig
from .observer import RegimeObserver
from .result import RegimeResult
from .state import RegimeState
from .utils import compare_windows, walk_forward_train, clamp01


class RegimeBoss(ABC):
    """
    Domain-neutral regime classifier skeleton.

    How to use
    ----------
    1. Subclass RegimeBoss.
    2. Implement classify(features: np.ndarray) → RegimeResult.
       This is where your model logic lives.
    3. Implement train(X_train, X_test) → Any.
       Sklearn-shaped — plug in your own model, this stays abstract.
    4. Optionally attach RegimeObserver instances for event hooks.
    5. Call update(features) in your main loop for streaming use,
       or call classify(features) directly for one-shot use.

    walk_forward_train and compare_windows are provided as utilities.
    Override them only if you need custom loop or divergence logic.

    Port plan (miu)
    ---------------
    - Input:    np.ndarray of ERIP vectors, shape (T, F) with sliding time window.
    - Observer: override on_regime_change to call mma.emit_event().
    - Taxonomy: RegimeCategory values map to emotional/relational state shifts.
                Rename in your subclass — the enum stays generic here.
    - Walk-forward: identical skeleton; plug your memory windows into train().
    - Divergence: compare_windows() defaults to symmetric KL.
                  Swap method="cosine" for direction-sensitive ERIP comparison.

    State is NOT shared across instances. Each RegimeBoss instance owns
    its own RegimeState — instantiate one per entity you're classifying.
    """

    def __init__(
        self,
        config: Optional[RegimeBossConfig] = None,
        observers: Optional[List[RegimeObserver]] = None,
    ) -> None:
        self.config    = config or RegimeBossConfig()
        self._observers: List[RegimeObserver] = list(observers or [])
        self._state    = RegimeState()

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def classify(self, features: np.ndarray) -> RegimeResult:
        """
        Classify the current regime from a feature vector or window.

        Parameters
        ----------
        features : np.ndarray
            Current feature input. Shape: (F,) for a single vector,
            or (T, F) for a window of T timesteps.

        Returns
        -------
        RegimeResult
            Category, confidence in [0, 1], and reasoning dict.

        Notes
        -----
        This is the ONLY required method. Everything else is optional
        scaffolding. Start here.
        """
        ...

    @abstractmethod
    def train(self, X_train: np.ndarray, X_test: np.ndarray) -> Any:
        """
        Train or update the internal model.

        Called by walk_forward_train() as the window slides.
        Also callable directly.

        Parameters
        ----------
        X_train : np.ndarray
            Training window. Shape (T, F). This is window_n.
        X_test : np.ndarray
            Test observation(s). Shape (1, F) or (T2, F). This is window_n+1.

        Returns
        -------
        Any
            Classifier-defined. Return a score, fitted model, or None.
            walk_forward_train() collects whatever you return here.

        Notes
        -----
        Leave the model implementation abstract. sklearn estimators,
        PyTorch modules, Gaussian processes — all plug in here.
        The signature is the contract; the internals are yours.
        """
        ...

    # ── Provided utilities ────────────────────────────────────────────────────

    def walk_forward_train(
        self,
        features: np.ndarray,
        *,
        min_train_size: Optional[int] = None,
        step_size: int = 1,
    ) -> List[tuple]:
        """
        Walk-forward loop: train on window_n, test on window_n+1.

        Loop logic only — model lives in self.train().
        See utils.walk_forward_train for full documentation.

        Parameters
        ----------
        features : np.ndarray
            Full feature array. Shape (T, F).
        min_train_size : int, optional
            Minimum window_n size. Defaults to config.wf_min_train_size.
        step_size : int
            Window advance per iteration. 1 = fully overlapping.

        Returns
        -------
        List of (train_end_idx, test_idx, train_result) tuples.
        """
        size = min_train_size if min_train_size is not None else self.config.wf_min_train_size
        return walk_forward_train(features, self, min_train_size=size, step_size=step_size)

    def compare_windows(
        self,
        window_n: np.ndarray,
        window_n_plus_1: np.ndarray,
        method: str = "kl",
    ) -> float:
        """
        Statistical divergence between two feature windows.

        Higher = more different = more likely a regime boundary exists.

        Parameters
        ----------
        window_n : np.ndarray
            Past window.
        window_n_plus_1 : np.ndarray
            Future window.
        method : str
            "kl" | "cosine" | "euclidean". Default: "kl".

        Returns
        -------
        float
            Divergence score. Compare against config.divergence_threshold.
        """
        return compare_windows(window_n, window_n_plus_1, method=method)

    # ── Observer management ───────────────────────────────────────────────────

    def add_observer(self, observer: RegimeObserver) -> None:
        """Register an observer. Multiple observers are supported."""
        self._observers.append(observer)

    def remove_observer(self, observer: RegimeObserver) -> None:
        """Unregister an observer."""
        self._observers.remove(observer)

    def _notify_observers(
        self,
        before: RegimeCategory,
        after: RegimeCategory,
        result: RegimeResult,
    ) -> None:
        """Fire on_regime_change on all registered observers."""
        for obs in self._observers:
            try:
                obs.on_regime_change(before, after, result)
            except Exception as exc:
                # Observers must not crash the classifier.
                # Log and continue.
                print(f"[RegimeBoss] Observer {type(obs).__name__} raised: {exc}")

    # ── State management ──────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """
        Reset rolling state. Call on epoch/session boundary.

        Alternatively, pass epoch_id to update() and state resets
        automatically when the epoch changes.
        """
        self._state.reset_for_new_epoch()

    @property
    def current_category(self) -> RegimeCategory:
        """The most recently classified category."""
        return self._state.current_category

    @property
    def state(self) -> RegimeState:
        """Direct access to internal state. Read-only by convention."""
        return self._state

    # ── Core update loop ──────────────────────────────────────────────────────

    def update(
        self,
        features: np.ndarray,
        *,
        window_id: str = "",
        epoch_id: str = "",
    ) -> RegimeResult:
        """
        Classify and update state. Fires observers on regime change.

        This is the main entry point for streaming / live use.
        For batch or one-shot use, call classify() directly.

        Parameters
        ----------
        features : np.ndarray
            Current feature vector or window.
        window_id : str
            Optional dedup key for this observation window (e.g. timestamp).
            When provided, duplicate window_ids are skipped — safe to call
            in a polling loop faster than your window advances.
        epoch_id : str
            Optional epoch/session identifier (e.g. date, conversation ID).
            When this changes, rolling state resets automatically.

        Returns
        -------
        RegimeResult
            Classification result for this step.
        """
        # Epoch boundary reset
        if epoch_id and epoch_id != self._state.last_epoch_id:
            self._state.last_epoch_id = epoch_id
            self._state.reset_for_new_epoch()

        # Dedup guard — skip if window hasn't advanced
        if window_id and window_id == self._state.last_window_id:
            return RegimeResult(
                category=self._state.current_category,
                confidence=0.0,
                reasoning={
                    "primary_signal": "duplicate_window",
                    "contributing_signals": [],
                    "confidence_factors": {},
                },
            )
        if window_id:
            self._state.last_window_id = window_id

        # Classify
        result = self.classify(features)

        # Apply confidence gate — downgrade to UNDEFINED if below floor
        if result.confidence < self.config.min_confidence and result.category != RegimeCategory.UNDEFINED:
            result = RegimeResult.undefined(
                reason=f"confidence_below_floor:{result.confidence:.3f}"
            )

        # Observer dispatch on category change
        before = self._state.current_category
        if result.category != before:
            self._notify_observers(before, result.category, result)
            self._state.current_category = result.category
            self._state.category_age     = 0
        else:
            self._state.category_age += 1

        # Update rolling history
        self._state.append_category(result.category, max_len=self.config.transition_window * 4)
        scalar = float(np.mean(np.abs(features))) if features.size > 0 else 0.0
        self._state.append_signal(scalar, max_len=self.config.stability_window * 4)

        return result
