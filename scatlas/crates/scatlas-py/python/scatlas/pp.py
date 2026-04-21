"""scatlas.pp — preprocessing kernels (randomized PCA, HVG, etc.).

scanpy-compatible wrappers over Rust randomized SVD. Current scope:
  * ``pca`` — randomized truncated SVD on sparse CSR or dense X;
    matches scanpy's ``pp.pca(adata, zero_center=False)`` behavior on
    sparse inputs.

Mean-centered PCA (zero_center=True) is not yet implemented — on sparse
inputs it would require densification or implicit-centering matmul; for
typical log-normed 10x data the ``zero_center=False`` path is the
scanpy default anyway.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scatlas._scatlas_native import pp as _rust_pp

__all__ = ["pca", "suggest_n_comps"]


def suggest_n_comps(
    singular_values: np.ndarray,
    n_rows: int,
    n_cols: int,
    *,
    margin: int = 5,
    min_comps: int = 15,
    max_comps: int = 50,
) -> dict:
    """Suggest an optimal number of PCs from a scree curve.

    Primary: Gavish-Donoho 2014 MSE-optimal hard threshold
    (`τ* = ω(β)·median(sv)`, β = min(N,G)/max(N,G)). Returns a dict
    with the Gavish-Donoho count, a perpendicular-line elbow
    cross-check, and a final `suggested_n_comps` (GD + margin, clamped).
    """
    sv = np.ascontiguousarray(singular_values, dtype=np.float32)
    return _rust_pp.suggest_n_comps_py(
        sv, int(n_rows), int(n_cols),
        margin=int(margin), min_comps=int(min_comps), max_comps=int(max_comps),
    )


def _ensure_csr_u64_u32(X) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Normalize scipy.sparse input to (indptr: u64, indices: u32, data: f32).

    scipy uses i32 indptr/indices by default but u32 indices is sufficient
    for G ≤ 4B features; u64 indptr handles nnz > 2B.
    """
    import scipy.sparse as sp

    if not sp.issparse(X):
        raise TypeError("expected scipy.sparse matrix")
    X = X.tocsr()
    n_rows, n_cols = X.shape
    indptr = np.asarray(X.indptr, dtype=np.uint64)
    indices = np.asarray(X.indices, dtype=np.uint32)
    data = np.ascontiguousarray(X.data, dtype=np.float32)
    return indptr, indices, data, (n_rows, n_cols)


def pca(
    adata,
    n_comps: int | str = 30,
    *,
    zero_center: bool = False,
    layer: str | None = None,
    use_highly_variable: bool = False,
    n_oversamples: int = 10,
    n_power_iter: int = 7,
    random_state: int = 0,
    auto_max_comps: int = 60,
    auto_margin: int = 5,
    auto_min_comps: int = 15,
    auto_max_clip: int = 50,
    copy: bool = False,
):
    """Randomized truncated SVD over AnnData, writing ``obsm['X_pca']``.

    Parameters
    ----------
    adata
        AnnData-like object (``.X``, ``.obs``, ``.obsm``, ``.uns``, ``.varm``).
    n_comps
        Number of principal components to return. Pass ``"auto"`` to
        run PCA with ``auto_max_comps`` components, then select the
        final count via Gavish-Donoho 2014 optimal hard threshold +
        margin (see ``scatlas.pp.suggest_n_comps``).
    zero_center
        Must be ``False`` — mean-centering on sparse inputs isn't yet
        implemented.
    layer
        If set, use ``adata.layers[layer]`` instead of ``adata.X``.
    use_highly_variable
        Restrict to ``adata.var['highly_variable']`` genes if set.
    n_oversamples, n_power_iter, random_state
        Randomized-SVD knobs. Our defaults (7 power iterations) are 2
        higher than sklearn's because we use QR re-orthogonalization
        (vs sklearn's LU), which converges a bit slower per iteration.
    auto_max_comps, auto_margin, auto_min_comps, auto_max_clip
        Only used when ``n_comps="auto"``. First runs PCA with
        ``auto_max_comps`` components, then sets
        ``final = clamp(GD_count + auto_margin, auto_min_comps, auto_max_clip)``.
    copy
        If ``True``, return a modified copy instead of mutating.

    Returns
    -------
    result : dict
        Keys ``embedding``, ``components``, ``singular_values``,
        ``explained_variance``, ``explained_variance_ratio``,
        ``n_comps`` (the effective count actually used). Also
        writes ``adata.obsm['X_pca']``, ``varm['PCs']``, ``uns['pca']``.
    """
    import scipy.sparse as sp

    if zero_center:
        raise NotImplementedError(
            "zero_center=True not yet supported in scatlas.pp.pca — use "
            "zero_center=False (matches scanpy's sparse default)."
        )

    if copy:
        adata = adata.copy()

    X = adata.layers[layer] if layer is not None else adata.X

    if use_highly_variable:
        if "highly_variable" not in adata.var.columns:
            raise ValueError(
                "use_highly_variable=True but `adata.var['highly_variable']` missing"
            )
        mask = adata.var["highly_variable"].to_numpy().astype(bool)
        X = X[:, mask]

    # --- Resolve n_comps (handle "auto") ---
    auto_mode = isinstance(n_comps, str) and n_comps == "auto"
    target_n = auto_max_comps if auto_mode else int(n_comps)

    if sp.issparse(X):
        indptr, indices, data, (n_rows, n_cols) = _ensure_csr_u64_u32(X)
        if target_n > min(n_rows, n_cols):
            target_n = min(n_rows, n_cols)
        embedding, components, s, ev, evr = _rust_pp.pca_csr(
            indptr, indices, data,
            n_rows, n_cols,
            n_comps=target_n,
            n_oversamples=n_oversamples,
            n_power_iter=n_power_iter,
            seed=int(random_state),
        )
    else:
        dense = np.ascontiguousarray(np.asarray(X), dtype=np.float32)
        n_rows, n_cols = dense.shape
        if target_n > min(n_rows, n_cols):
            target_n = min(n_rows, n_cols)
        embedding, components, s, ev, evr = _rust_pp.pca_dense(
            dense,
            n_comps=target_n,
            n_oversamples=n_oversamples,
            n_power_iter=n_power_iter,
            seed=int(random_state),
        )

    # --- Auto-select: compute Gavish-Donoho + slice ---
    suggestion = None
    if auto_mode:
        suggestion = suggest_n_comps(
            s, n_rows, n_cols,
            margin=auto_margin,
            min_comps=auto_min_comps,
            max_comps=auto_max_clip,
        )
        k = int(suggestion["suggested_n_comps"])
        # Slice the (oversized) result down to the recommended rank.
        embedding = np.ascontiguousarray(embedding[:, :k])
        components = np.ascontiguousarray(components[:k, :])
        s = s[:k]
        ev = ev[:k]
        evr = evr[:k]

    adata.obsm["X_pca"] = embedding
    adata.varm_key = None  # scanpy uses adata.varm['PCs'] = components.T
    if "highly_variable" in adata.var.columns and use_highly_variable:
        # Scatter components back into full gene space for scanpy compat
        full = np.zeros((n_comps, adata.n_vars), dtype=np.float32)
        full[:, mask] = components
        adata.varm["PCs"] = full.T
    else:
        adata.varm["PCs"] = components.T

    effective_n = embedding.shape[1]
    uns_entry = {
        "singular_values": s,
        "variance": ev,
        "variance_ratio": evr,
        "params": {
            "zero_center": False,
            "use_highly_variable": bool(use_highly_variable),
            "n_comps": effective_n,
            "n_comps_requested": "auto" if auto_mode else int(n_comps),
            "n_oversamples": int(n_oversamples),
            "n_power_iter": int(n_power_iter),
            "random_state": int(random_state),
        },
    }
    if suggestion is not None:
        uns_entry["auto"] = {
            "n_comps_gavish_donoho": int(suggestion["n_comps_gavish_donoho"]),
            "n_comps_elbow": int(suggestion["n_comps_elbow"]),
            "suggested_n_comps": int(suggestion["suggested_n_comps"]),
            "gd_threshold": float(suggestion["gd_threshold"]),
            "sv_median": float(suggestion["sv_median"]),
            "beta": float(suggestion["beta"]),
        }
    adata.uns["pca"] = uns_entry

    return {
        "embedding": embedding,
        "components": components,
        "singular_values": s,
        "explained_variance": ev,
        "explained_variance_ratio": evr,
        "n_comps": effective_n,
        "auto": uns_entry.get("auto"),
    }
