"""Harmony 2 on a PCA embedding.

Ported from v1 ``scatlas.ext.harmony`` at V2-P2.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def harmony(
    adata,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    key_added: str = "X_pca_harmony",
    n_clusters: int | None = None,
    theta: float = 2.0,
    sigma: float = 0.1,
    lambda_: float | None = 1.0,
    alpha: float = 0.2,
    max_iter: int = 10,
    max_iter_cluster: int = 20,
    epsilon_cluster: float = 1e-3,
    epsilon_harmony: float = 1e-2,
    block_size: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Run Harmony 2.0 on ``adata.obsm[use_rep]``.

    Writes corrected embedding to ``adata.obsm[key_added]`` and run
    metadata to ``adata.uns['harmony']``.

    Set ``lambda_=None`` for dynamic-lambda mode (matches
    ``RunHarmony(lambda=NULL)``).
    """
    from fast_auto_scrna._native import harmony as _native_harmony

    if use_rep not in adata.obsm:
        raise KeyError(
            f"{use_rep!r} not in adata.obsm — run PCA first or pass use_rep"
        )
    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key!r} not in adata.obs")

    pca_arr = np.ascontiguousarray(adata.obsm[use_rep], dtype=np.float32)

    batch_raw = adata.obs[batch_key].to_numpy()
    code_map: dict[int, Any] = {}
    if np.asarray(batch_raw).dtype.kind in {"i", "u"}:
        batch_codes = np.asarray(batch_raw, dtype=np.int32)
    else:
        uniq, codes = np.unique(batch_raw, return_inverse=True)
        batch_codes = codes.astype(np.int32)
        code_map = {int(i): uniq[i] for i in range(len(uniq))}

    if len(np.unique(batch_codes)) < 2:
        raise ValueError(
            f"batch_key {batch_key!r} has only "
            f"{len(np.unique(batch_codes))} unique value(s); Harmony "
            "needs ≥ 2 batches to integrate."
        )

    z_corr, r, y, obj, converged = _native_harmony.harmony_integrate(
        pca_arr, batch_codes, n_clusters,
        float(theta), float(sigma),
        None if lambda_ is None else float(lambda_),
        float(alpha),
        int(max_iter), int(max_iter_cluster),
        float(epsilon_cluster), float(epsilon_harmony),
        float(block_size), int(seed),
    )

    adata.obsm[key_added] = z_corr
    adata.uns["harmony"] = {
        "key_added": key_added,
        "use_rep": use_rep,
        "batch_key": batch_key,
        "batch_code_map": code_map,
        "objective_harmony": np.asarray(obj),
        "converged_at_iter": converged,
        "params": {
            "n_clusters": int(n_clusters) if n_clusters is not None else None,
            "theta": float(theta),
            "sigma": float(sigma),
            "lambda": None if lambda_ is None else float(lambda_),
            "alpha": float(alpha),
            "lambda_mode": "dynamic" if lambda_ is None else "fixed",
            "max_iter": int(max_iter),
            "max_iter_cluster": int(max_iter_cluster),
            "epsilon_cluster": float(epsilon_cluster),
            "epsilon_harmony": float(epsilon_harmony),
            "block_size": float(block_size),
            "seed": int(seed),
        },
    }
    return {
        "z_corrected": z_corr,
        "r": r,
        "y": y,
        "objective_harmony": np.asarray(obj),
        "converged_at_iter": converged,
        "batch_code_map": code_map,
    }
