# example_implementation.py
"""
Example: How to subclass RegimeBoss.

This file shows the minimal implementation pattern. It is NOT part of
the template — delete or replace with your own classifier.

For miu's port:
- Replace the dummy classify() with your ERIP vector classifier.
- Replace train() with your model's fit/update logic.
- Replace LoggingObserver with your MMA emit_event observer.
- Swap RegimeCategory values for your emotional/relational state taxonomy.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from regime_boss_template import (
    RegimeBoss,
    RegimeBossConfig,
    RegimeCategory,
    RegimeObserver,
    RegimeResult,
    LoggingObserver,
    compare_windows,
)


# ── 1. Custom observer — wire into your event stream here ─────────────────────

class PrintObserver(RegimeObserver):
    """
    Replace this with your MMA emit_event observer.

    Example for miu:
        class MMAObserver(RegimeObserver):
            def __init__(self, mma_instance):
                self.mma = mma_instance

            def on_regime_change(self, before, after, result):
                self.mma.emit_event("regime_shift", {
                    "from":       before.value,
                    "to":         after.value,
                    "confidence": result.confidence,
                    "signal":     result.reasoning["primary_signal"],
                    "factors":    result.reasoning["confidence_factors"],
                })
    """
    def on_regime_change(self, before, after, result):
        print(
            f"  SHIFT: {before.value} → {after.value}"
            f" | conf={result.confidence:.3f}"
            f" | primary={result.reasoning['primary_signal']}"
        )


# ── 2. Concrete RegimeBoss implementation ─────────────────────────────────────

class ExampleBoss(RegimeBoss):
    """
    Minimal concrete implementation showing the required method signatures.

    The classify() and train() bodies are placeholder stubs — replace
    with your actual model. Everything else (observers, walk-forward,
    state management, epoch resets) is inherited and ready to use.
    """

    def __init__(self, config: Optional[RegimeBossConfig] = None) -> None:
        super().__init__(config=config, observers=[PrintObserver()])
        # Your model state goes here — sklearn estimator, weights, etc.
        self._fitted = False

    # ── Required: implement classify ──────────────────────────────────────────

    def classify(self, features: np.ndarray) -> RegimeResult:
        """
        Classify regime from feature vector or window.

        Replace this body with your model's inference logic.
        The signature must stay: (np.ndarray) → RegimeResult.

        This stub uses a simple heuristic on feature variance as a demo.
        """
        if features.size == 0:
            return RegimeResult.undefined("empty_features")

        # --- YOUR MODEL INFERENCE HERE ---
        # Example: treat feature variance as a stability signal.
        variance = float(np.var(features))
        mean_val = float(np.mean(features))

        # Derive a confidence score from signal strength (replace with real logic)
        confidence = min(variance * 5.0, 1.0)

        # Classify based on variance thresholds (replace with your thresholds)
        if variance < 0.05:
            category = RegimeCategory.STABLE
            primary  = "low_variance"
        elif variance > 0.50:
            category = RegimeCategory.VOLATILE
            primary  = "high_variance"
        elif abs(mean_val) > 0.30:
            category = RegimeCategory.TRANSITIONING
            primary  = "mean_drift"
        else:
            category = RegimeCategory.EMERGING
            primary  = "moderate_signal"

        return RegimeResult(
            category=category,
            confidence=confidence,
            reasoning={
                "primary_signal": primary,
                "contributing_signals": ["variance", "mean"],
                "confidence_factors": {
                    "variance": variance,
                    "mean":     mean_val,
                },
            },
        )

    # ── Required: implement train ─────────────────────────────────────────────

    def train(self, X_train: np.ndarray, X_test: np.ndarray) -> Any:
        """
        Train on window_n, evaluate on window_n+1.

        Replace this body with your model's fit/update logic.
        The signature must stay: (np.ndarray, np.ndarray) → Any.

        Return value is collected by walk_forward_train() —
        return whatever is useful to you (score, loss, None).
        """
        if len(X_train) < 2:
            return None

        # --- YOUR MODEL TRAINING HERE ---
        # Example: compute divergence between the two windows as a score.
        divergence = compare_windows(X_train[-10:], X_test, method="kl")
        self._fitted = True
        return divergence


# ── 3. Usage demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Config: tune thresholds without touching classifier code ---
    config = RegimeBossConfig(
        stability_window   = 8,
        transition_window  = 12,
        min_confidence     = 0.30,   # lower for demo
        divergence_threshold = 0.25,
        wf_min_train_size  = 10,     # lower for demo
    )

    boss = ExampleBoss(config=config)

    # --- Streaming use: call update() per timestep ---
    print("=== Streaming classification ===")
    rng = np.random.default_rng(42)

    for i in range(20):
        # Simulate: low variance first, then a spike, then settling
        if i < 8:
            vec = rng.normal(0.0, 0.1, size=(5,))
        elif i < 12:
            vec = rng.normal(0.5, 0.8, size=(5,))
        else:
            vec = rng.normal(0.1, 0.15, size=(5,))

        result = boss.update(vec, window_id=str(i), epoch_id="demo_session")
        print(
            f"  t={i:02d} | {result.category.value:<14}"
            f" | conf={result.confidence:.3f}"
            f" | {result.reasoning['primary_signal']}"
        )

    # --- Walk-forward: slide a training window across a feature matrix ---
    print("\n=== Walk-forward training ===")
    feature_matrix = rng.normal(0, 0.2, size=(50, 5))
    feature_matrix[25:35] += 1.5   # inject a regime shift

    results = boss.walk_forward_train(feature_matrix, min_train_size=10)
    print(f"  Completed {len(results)} walk-forward steps.")
    for train_end, test_idx, score in results[-5:]:  # show last 5
        flag = " ← SHIFT" if score is not None and score > config.divergence_threshold else ""
        print(f"  Train[0:{train_end}] → Test[{test_idx}] | div={score:.4f}{flag}")

    # --- Direct divergence comparison ---
    print("\n=== Direct window comparison ===")
    stable   = rng.normal(0, 0.1, size=(20, 5))
    volatile = rng.normal(0, 1.5, size=(20, 5))
    div_kl   = boss.compare_windows(stable, volatile, method="kl")
    div_cos  = boss.compare_windows(stable, volatile, method="cosine")
    print(f"  KL divergence (stable vs volatile): {div_kl:.4f}")
    print(f"  Cosine divergence (stable vs volatile): {div_cos:.4f}")
