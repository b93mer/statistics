# regime_boss_template/utils.py
from __future__ import annotations

from typing import Any, List, Optional, Protocol, Tuple

import numpy as np


# ── Divergence ────────────────────────────────────────────────────────────────

def compare_windows(
    window_n: np.ndarray,
    window_n_plus_1: np.ndarray,
    method: str = "kl",
) -> float:
    """
    Compute statistical divergence between two feature windows.

    Higher score = more different = more likely a regime boundary exists
    between these two windows.

    Parameters
    ----------
    window_n : np.ndarray
        Past window (train). Shape (T, F) or (F,).
    window_n_plus_1 : np.ndarray
        Future window (test). Shape (T, F) or (F,).
    method : str
        Divergence metric. One of:
          "kl"        — Symmetric KL divergence on normalized distributions.
                        Range: [0, inf). Scale: ~0 = identical, >0.5 = clear shift.
          "cosine"    — 1 - cosine_similarity on mean vectors.
                        Range: [0, 2]. 0 = same direction, 1 = orthogonal.
          "euclidean" — L2 distance between mean vectors.
                        Range: [0, inf). Scale is feature-dependent.
        Default: "kl".

    Returns
    -------
    float
        Divergence score. Compare against RegimeBossConfig.divergence_threshold.

    Notes
    -----
    For miu's port: swap in your own divergence function here.
    The contract is (np.ndarray, np.ndarray) → float.
    KL is a reasonable default for normalized ERIP vector distributions.
    Cosine is cleaner when direction matters more than magnitude.
    """
    if method == "kl":
        return _kl_divergence(window_n, window_n_plus_1)
    elif method == "cosine":
        return _cosine_divergence(window_n, window_n_plus_1)
    elif method == "euclidean":
        return _euclidean_divergence(window_n, window_n_plus_1)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose from: 'kl', 'cosine', 'euclidean'"
        )


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric KL divergence (Jensen-Shannon style) on flattened, normalized arrays."""
    p_flat = np.asarray(p, dtype=float).flatten()
    q_flat = np.asarray(q, dtype=float).flatten()
    # Shift to non-negative then normalize to distributions
    p_flat = p_flat - p_flat.min() + eps
    q_flat = q_flat - q_flat.min() + eps
    p_flat = p_flat / p_flat.sum()
    q_flat = q_flat / q_flat.sum()
    # Pad to equal length
    n = max(len(p_flat), len(q_flat))
    p_flat = np.pad(p_flat, (0, n - len(p_flat)), constant_values=eps)
    q_flat = np.pad(q_flat, (0, n - len(q_flat)), constant_values=eps)
    kl_pq = np.sum(p_flat * np.log(p_flat / (q_flat + eps)))
    kl_qp = np.sum(q_flat * np.log(q_flat / (p_flat + eps)))
    return float((kl_pq + kl_qp) / 2.0)


def _cosine_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity on mean vectors. Range [0, 2]."""
    a_mean = np.asarray(a, dtype=float).mean(axis=0) if np.asarray(a).ndim > 1 else np.asarray(a, dtype=float)
    b_mean = np.asarray(b, dtype=float).mean(axis=0) if np.asarray(b).ndim > 1 else np.asarray(b, dtype=float)
    denom = np.linalg.norm(a_mean) * np.linalg.norm(b_mean)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(a_mean, b_mean) / denom)


def _euclidean_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between mean feature vectors."""
    a_mean = np.asarray(a, dtype=float).mean(axis=0) if np.asarray(a).ndim > 1 else np.asarray(a, dtype=float)
    b_mean = np.asarray(b, dtype=float).mean(axis=0) if np.asarray(b).ndim > 1 else np.asarray(b, dtype=float)
    return float(np.linalg.norm(a_mean - b_mean))


# ── Walk-forward ──────────────────────────────────────────────────────────────

class _TrainableProtocol(Protocol):
    def train(self, X_train: np.ndarray, X_test: np.ndarray) -> Any: ...


def walk_forward_train(
    features: np.ndarray,
    classifier: _TrainableProtocol,
    *,
    min_train_size: int = 30,
    step_size: int = 1,
) -> List[Tuple[int, int, Any]]:
    """
    Walk-forward training loop. Train on window_n, evaluate on window_n+1.

    This is the loop logic only — model implementation lives in the classifier.

    Parameters
    ----------
    features : np.ndarray
        Full feature array. Shape (T, F) — T timesteps, F features.
    classifier : object
        Any object with a .train(X_train, X_test) method.
        sklearn estimators work. Custom models work if they follow
        the same signature. RegimeBoss subclasses work directly.
    min_train_size : int
        Minimum timesteps in window_n before the first evaluation.
        Default: 30.
    step_size : int
        How many steps to advance the window per iteration.
        1 = fully overlapping. n = non-overlapping.

    Returns
    -------
    List of (train_end_idx, test_idx, train_result) tuples.
        train_end_idx : int
            Index of last training observation.
        test_idx : int
            Index of test observation (train_end_idx + 1).
        train_result : Any
            Whatever classifier.train() returned (score, None, etc).

    Notes
    -----
    For miu's port: window_n is your memory window. window_n_plus_1 is
    the next timestep you're predicting into. The loop doesn't care what
    the classifier does internally — it just slides the window and calls train().

    Example
    -------
    results = walk_forward_train(erip_vectors, my_boss, min_train_size=20)
    for train_end, test_idx, score in results:
        print(f"Train [0:{train_end}] → Test [{test_idx}] | score={score}")
    """
    T = len(features)
    if T < min_train_size + 1:
        return []

    results: List[Tuple[int, int, Any]] = []
    for train_end in range(min_train_size, T - 1, step_size):
        window_n        = features[:train_end]
        window_n_plus_1 = features[train_end: train_end + 1]
        result = classifier.train(window_n, window_n_plus_1)
        results.append((train_end, train_end + 1, result))

    return results


# ── Misc ──────────────────────────────────────────────────────────────────────

def clamp01(x: float) -> float:
    """Clamp x to [0.0, 1.0]."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
