# regime_boss_template/__init__.py
"""
RegimeBoss Template
===================
Domain-neutral regime classification skeleton.

Public API
----------
RegimeBoss       — Abstract base classifier. Subclass and implement classify() + train().
RegimeCategory   — Domain-neutral enum. Map to your domain's state semantics.
RegimeResult     — Output dataclass: category, confidence, reasoning dict.
RegimeObserver   — Abstract event hook. Override on_regime_change().
LoggingObserver  — Minimal concrete observer (stdout). Replace with your own.
RegimeBossConfig — All thresholds in one place. Nothing hardcoded.
RegimeState      — Rolling state dataclass. Extended if needed.
compare_windows  — Statistical divergence between two feature windows.
walk_forward_train — Walk-forward loop skeleton. Model lives in your subclass.

Quick start
-----------
    import numpy as np
    from regime_boss_template import RegimeBoss, RegimeCategory, RegimeResult

    class MyBoss(RegimeBoss):
        def classify(self, features: np.ndarray) -> RegimeResult:
            # Your model here
            ...
        def train(self, X_train, X_test):
            # Your training logic here
            ...

    boss = MyBoss()
    result = boss.update(features, epoch_id="session_001")
"""

from .boss       import RegimeBoss
from .categories import RegimeCategory
from .config     import RegimeBossConfig
from .observer   import RegimeObserver, LoggingObserver
from .result     import RegimeResult
from .state      import RegimeState
from .utils      import compare_windows, walk_forward_train

__all__ = [
    "RegimeBoss",
    "RegimeCategory",
    "RegimeBossConfig",
    "RegimeObserver",
    "LoggingObserver",
    "RegimeResult",
    "RegimeState",
    "compare_windows",
    "walk_forward_train",
]
