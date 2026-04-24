"""SCCAF-style cluster-separability score — sklearn LR CV.

Evaluated a Rust port via linfa-logistic in an earlier pass (see
ROADMAP). Result: numerically consistent with sklearn (|Δ| < 0.005) but
~1.8× SLOWER because scipy's lbfgs is LAPACK-backed and linfa's pure
Rust argmin implementation is not. Since SCCAF is already a minor share
of per-route Phase-2 wall (~13 s at 222 k), the Rust port was rejected.

Ported from v1 ``scatlas.metrics.sccaf_accuracy``.
"""
from __future__ import annotations

import numpy as np


def sccaf_accuracy(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    test_size: float = 0.2,
    n_splits: int = 3,
    random_state: int = 0,
    max_iter: int = 200,
) -> float:
    """SCCAF-style cluster-separability score.

    Score 1.0 = clusters are linearly separable on this embedding; < 0.7
    typically means clusters are over-fragmented.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    labels = np.asarray(cluster_labels)
    if len(np.unique(labels)) < 2:
        return 1.0
    X = np.ascontiguousarray(embedding, dtype=np.float32)

    _, counts = np.unique(labels, return_counts=True)
    k = min(n_splits, int(counts.min()))
    if k < 2:
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(
            X, labels, test_size=test_size, random_state=random_state,
            stratify=None,
        )
        clf = LogisticRegression(max_iter=max_iter, n_jobs=-1).fit(Xtr, ytr)
        return float(clf.score(Xte, yte))

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(np.mean(scores))
