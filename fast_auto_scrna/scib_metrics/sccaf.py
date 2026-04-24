"""SCCAF-style cluster-separability score.

Current: 3-fold CV accuracy of sklearn ``LogisticRegression`` on the
embedding. A Rust SCCAF kernel is on the roadmap (see ROADMAP ≫
"SCCAF-Rust" milestone) — when it lands, this module will dispatch to
``fast_auto_scrna._native.sccaf.lr_cv_accuracy(...)`` and fall back to
sklearn only if the extension isn't installed.

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
    # Rust dispatch hook — activated by the SCCAF-Rust milestone.
    try:
        from fast_auto_scrna._native import sccaf as _native_sccaf  # type: ignore
        if hasattr(_native_sccaf, "lr_cv_accuracy"):
            return float(_native_sccaf.lr_cv_accuracy(
                np.ascontiguousarray(embedding, dtype=np.float32),
                np.ascontiguousarray(_encode(cluster_labels), dtype=np.int32),
                n_splits=int(n_splits),
                max_iter=int(max_iter),
                seed=int(random_state),
            ))
    except (ImportError, AttributeError):
        pass

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


def _encode(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.dtype.kind in {"i", "u"}:
        return arr.astype(np.int32)
    _, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int32)
