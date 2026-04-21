"""scatlas.tl — tool kernels (UMAP, leiden-wrapper, ...).

Current scope:
  * ``umap`` — Rust layout optimization over BBKNN connectivities,
    matches scanpy.tl.umap semantics (writes ``obsm['X_umap']``).
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as _sp

from scatlas._scatlas_native import tl as _rust_tl

__all__ = ["umap", "fit_ab"]


def fit_ab(min_dist: float = 0.5, spread: float = 1.0) -> tuple[float, float]:
    """Fit (a, b) for the UMAP low-dim kernel ``1 / (1 + a·x^(2b))``
    to the piecewise target defined by ``(min_dist, spread)``.

    Matches :func:`umap.umap_.find_ab_params`.
    """
    return _rust_tl.fit_ab(float(min_dist), float(spread))


def _noisy_scale_coords(
    coords: np.ndarray, rng: np.random.Generator,
    max_coord: float = 10.0, noise: float = 1e-4,
) -> np.ndarray:
    """Match :func:`umap.umap_.noisy_scale_coords`: rescale to ``max|x|=10``
    and add Gaussian noise ``σ=1e-4`` to break eigenvector ties."""
    max_abs = float(np.abs(coords).max())
    if max_abs > 0:
        coords = coords * (max_coord / max_abs)
    coords = coords + (noise * rng.standard_normal(coords.shape)).astype(np.float32)
    return np.ascontiguousarray(coords.astype(np.float32))


def _spectral_init(graph, n_components: int, random_state: int) -> np.ndarray | None:
    """Spectral layout = bottom ``n_components`` non-trivial eigenvectors of the
    normalized Laplacian L = I - D^(-1/2) W D^(-1/2).

    Delegates to :func:`umap.spectral.spectral_layout` when umap-learn is
    importable (bit-exact parity with umap-learn's default init). Returns
    ``None`` on failure so the caller can fall back.
    """
    try:
        from umap.spectral import spectral_layout as _umap_spectral
    except ImportError:
        return None

    try:
        rng = np.random.default_rng(int(random_state))
        emb = _umap_spectral(
            data=None,
            graph=graph.astype(np.float64, copy=False),
            dim=n_components,
            random_state=rng,
        )
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:  # ARPACK divergence, etc.
        warnings.warn(
            f"spectral init failed ({type(e).__name__}: {e}); falling back to PCA init",
            RuntimeWarning, stacklevel=2,
        )
        return None


def _pick_init(
    adata, init: str, n_components: int, random_state: int,
    graph=None,
) -> np.ndarray:
    """Pick an initial low-dim embedding for UMAP.

    After producing the raw init, all branches apply ``noisy_scale_coords``:
    rescale so ``max|x|=10`` and add ``σ=1e-4`` Gaussian noise — matching
    umap-learn so the SGD starts at a comparable scale.

    Branches:
      * ``random`` — uniform on ``[-10, 10]``.
      * ``pca`` — first ``n_components`` PCs. NOTE: because we rescale by
        a single ``max|x|`` across both dims, if PC1 variance ≫ PC2 this
        produces a highly elongated init that SGD cannot untangle in 200
        epochs. Use ``spectral`` unless you know your PCs are balanced.
      * ``spectral`` (default) — bottom eigenvectors of the normalized
        Laplacian of the connectivity graph. Matches umap-learn's default
        init; the only init that reliably avoids dimensional collapse on
        trajectory-like data (development, differentiation).
    """
    n = adata.n_obs
    rng = np.random.default_rng(int(random_state))

    if init == "random":
        return (20.0 * rng.random((n, n_components)).astype(np.float32) - 10.0).astype(
            np.float32, copy=False
        )

    if init == "spectral":
        if graph is None:
            raise ValueError("init='spectral' requires the connectivity graph")
        emb = _spectral_init(graph, n_components, random_state)
        if emb is not None:
            return _noisy_scale_coords(emb, rng)
        # fall through to PCA fallback
        init = "pca"
        warnings.warn(
            "spectral init unavailable (umap-learn missing or solver failed); "
            "using PCA init as fallback",
            RuntimeWarning, stacklevel=2,
        )

    if init in ("pca", "X_pca"):
        if "X_pca" not in adata.obsm:
            raise ValueError(
                "init='pca' but adata.obsm['X_pca'] missing — run scatlas.pp.pca first"
            )
        pc = np.ascontiguousarray(
            adata.obsm["X_pca"][:, :n_components], dtype=np.float32
        )
        return _noisy_scale_coords(pc, rng)

    raise ValueError(f"unknown init '{init}' — use 'spectral', 'pca', or 'random'")


def umap(
    adata,
    *,
    neighbors_key: str = "bbknn",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    n_epochs: int | None = None,
    init: str = "spectral",
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    learning_rate: float = 1.0,
    random_state: int = 0,
    single_thread: bool = False,
    copy: bool = False,
):
    """Compute UMAP layout from a pre-computed connectivity graph.

    Expects ``adata.obsp['<neighbors_key>_connectivities']`` as the
    symmetric fuzzy-simplicial-set output (scatlas BBKNN writes
    ``bbknn_connectivities`` when ``with_connectivities=True``).

    Writes ``adata.obsm['X_umap']`` and diagnostic info to
    ``adata.uns['umap']``.

    Parameters match scanpy.tl.umap. ``init='spectral'`` (default) uses
    the Laplacian bottom eigenvectors of the connectivity graph — the
    same init umap-learn uses by default, and the only one that avoids
    dimensional collapse on trajectory-like data. Requires umap-learn
    to be importable (transitive scanpy dep); falls back to PCA init
    with a warning otherwise.

    ``single_thread=True`` disables the rayon Hogwild parallelism in the
    Rust SGD kernel. Use for parity testing against umap-learn — the
    serial kernel is deterministic at fixed ``random_state``.
    """
    if copy:
        adata = adata.copy()

    conn_key = f"{neighbors_key}_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"adata.obsp['{conn_key}'] missing — run scatlas.ext.bbknn "
            f"with `with_connectivities=True` first"
        )

    C = adata.obsp[conn_key]
    if not _sp.issparse(C):
        C = _sp.csr_matrix(C)
    C = C.tocsr()
    n = C.shape[0]
    if n != adata.n_obs:
        raise ValueError(
            f"connectivities shape {C.shape} doesn't match adata.n_obs={adata.n_obs}"
        )

    indptr = np.asarray(C.indptr, dtype=np.uint64)
    indices = np.asarray(C.indices, dtype=np.uint32)
    data = np.ascontiguousarray(C.data, dtype=np.float32)

    init_emb = _pick_init(adata, init, n_components, random_state, graph=C)

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
        single_thread=bool(single_thread),
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
            "single_thread": bool(single_thread),
        },
        "a": float(a),
        "b": float(b),
    }
    return embedding
