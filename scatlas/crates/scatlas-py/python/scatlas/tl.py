"""scatlas.tl — tool kernels (UMAP, leiden-wrapper, ...).

Current scope:
  * ``umap`` — Rust layout optimization over BBKNN connectivities,
    matches scanpy.tl.umap semantics (writes ``obsm['X_umap']``).
"""
from __future__ import annotations

import numpy as np

from scatlas._scatlas_native import tl as _rust_tl

__all__ = ["umap", "fit_ab"]


def fit_ab(min_dist: float = 0.5, spread: float = 1.0) -> tuple[float, float]:
    """Fit (a, b) for the UMAP low-dim kernel ``1 / (1 + a·x^(2b))``
    to the piecewise target defined by ``(min_dist, spread)``.

    Matches :func:`umap.umap_.find_ab_params`.
    """
    return _rust_tl.fit_ab(float(min_dist), float(spread))


def _pick_init(adata, init: str, n_components: int, random_state: int) -> np.ndarray:
    """Pick an initial low-dim embedding for UMAP.

    Matches umap-learn convention: rescale so that `max(|x|) = 10`, then
    add small Gaussian noise (σ = 1e-4) to break ties. This scale is
    chosen so the SGD starts near the target [-10, 10] range — without
    it, embeddings drift for many epochs before reaching scale.
    """
    n = adata.n_obs
    rng = np.random.default_rng(int(random_state))
    if init == "random":
        return (20.0 * rng.random((n, n_components)).astype(np.float32) - 10.0).astype(
            np.float32, copy=False
        )
    if init in ("pca", "X_pca"):
        if "X_pca" not in adata.obsm:
            raise ValueError(
                "init='pca' but adata.obsm['X_pca'] missing — run scatlas.pp.pca first"
            )
        pc = np.ascontiguousarray(
            adata.obsm["X_pca"][:, :n_components], dtype=np.float32
        )
        # Match umap-learn `noisy_scale_coords`: rescale to max(|x|)=10,
        # then Gaussian noise σ=1e-4. Equivalent starting point so SGD
        # trajectories are comparable.
        max_abs = float(np.abs(pc).max())
        if max_abs > 0:
            pc = pc * (10.0 / max_abs)
        pc = pc + (1e-4 * rng.standard_normal(pc.shape)).astype(np.float32)
        return np.ascontiguousarray(pc.astype(np.float32))
    raise ValueError(f"unknown init '{init}' — use 'random' or 'pca'")


def umap(
    adata,
    *,
    neighbors_key: str = "bbknn",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    n_epochs: int | None = None,
    init: str = "pca",
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    learning_rate: float = 1.0,
    random_state: int = 0,
    copy: bool = False,
):
    """Compute UMAP layout from a pre-computed connectivity graph.

    Expects ``adata.obsp['<neighbors_key>_connectivities']`` as the
    symmetric fuzzy-simplicial-set output (scatlas BBKNN writes
    ``bbknn_connectivities`` when ``with_connectivities=True``).

    Writes ``adata.obsm['X_umap']`` and diagnostic info to
    ``adata.uns['umap']``.

    Parameters match scanpy.tl.umap where applicable. ``init='pca'``
    uses the first ``n_components`` dims of ``adata.obsm['X_pca']``
    (with scaling + noise) — fast and usually better than random.
    """
    import scipy.sparse as sp

    if copy:
        adata = adata.copy()

    conn_key = f"{neighbors_key}_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"adata.obsp['{conn_key}'] missing — run scatlas.ext.bbknn "
            f"with `with_connectivities=True` first"
        )

    C = adata.obsp[conn_key]
    if not sp.issparse(C):
        C = sp.csr_matrix(C)
    C = C.tocsr()
    n = C.shape[0]
    if n != adata.n_obs:
        raise ValueError(
            f"connectivities shape {C.shape} doesn't match adata.n_obs={adata.n_obs}"
        )

    indptr = np.asarray(C.indptr, dtype=np.uint64)
    indices = np.asarray(C.indices, dtype=np.uint32)
    data = np.ascontiguousarray(C.data, dtype=np.float32)

    init_emb = _pick_init(adata, init, n_components, random_state)

    embedding, a, b, n_epochs_used = _rust_tl.umap_layout(
        indptr, indices, data,
        n, init_emb,
        n_components=int(n_components),
        n_epochs=(None if n_epochs is None else int(n_epochs)),
        min_dist=float(min_dist),
        spread=float(spread),
        negative_sample_rate=int(negative_sample_rate),
        repulsion_strength=float(repulsion_strength),
        learning_rate=float(learning_rate),
        seed=int(random_state),
    )

    adata.obsm["X_umap"] = embedding
    adata.uns["umap"] = {
        "params": {
            "neighbors_key": neighbors_key,
            "min_dist": float(min_dist),
            "spread": float(spread),
            "n_components": int(n_components),
            "n_epochs": int(n_epochs_used),
            "init": init,
            "negative_sample_rate": int(negative_sample_rate),
            "repulsion_strength": float(repulsion_strength),
            "learning_rate": float(learning_rate),
            "random_state": int(random_state),
        },
        "a": float(a),
        "b": float(b),
    }
    return embedding
