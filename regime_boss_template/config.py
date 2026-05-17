# regime_boss_template/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RegimeBossConfig:
    """
    All tunable thresholds in one place. Nothing hardcoded in the classifier.

    Pass a custom instance to RegimeBoss.__init__ to override any value.
    Tune per domain — defaults are deliberately conservative starting points,
    not prescriptions.

    Rolling window sizes
    --------------------
    stability_window : int
        Number of observations used to assess signal stability.
    transition_window : int
        Number of observations used for transition detection.
    max_state_age : int
        Maximum observations before forcing a re-evaluation regardless
        of hysteresis. Prevents stale regime lock.

    Hysteresis
    ----------
    hysteresis_enter : int
        Consecutive agreeing signals required before entering a new regime.
        Higher = slower to enter, more stable. Lower = more responsive.
    hysteresis_exit : int
        Consecutive counter-signals required before exiting a regime.
        Higher = more persistent, less whipsaw. Lower = faster exit.

    Confidence
    ----------
    min_confidence : float
        Below this threshold → result treated as UNDEFINED regardless of category.
    high_confidence : float
        Above this → strong signal. Used internally to gate some transitions.

    Divergence
    ----------
    divergence_threshold : float
        compare_windows() score above this triggers a regime-shift flag.
        Scale depends on chosen method (kl / cosine / euclidean).

    Walk-forward
    ------------
    wf_min_train_size : int
        Minimum window_n length before the first walk-forward training step.

    Extras
    ------
    extras : dict
        Arbitrary user-defined parameters. Use this for domain-specific
        thresholds that don't belong in the base config.
        Example: {"erip_dim_weights": [0.4, 0.3, 0.3], "min_coherence": 0.6}
    """

    # Rolling windows
    stability_window: int   = 8
    transition_window: int  = 12
    max_state_age: int      = 50

    # Hysteresis
    hysteresis_enter: int   = 2
    hysteresis_exit: int    = 3

    # Confidence gates
    min_confidence: float   = 0.40
    high_confidence: float  = 0.75

    # Divergence
    divergence_threshold: float = 0.30

    # Walk-forward
    wf_min_train_size: int  = 30

    # User-defined overflow
    extras: Dict[str, Any]  = field(default_factory=dict)
