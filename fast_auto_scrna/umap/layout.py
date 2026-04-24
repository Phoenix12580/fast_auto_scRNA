"""UMAP layout from pre-computed connectivities. Ported from v1 ``scatlas.tl.umap``."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as _sp


def fit_ab(min_dist: float = 0.5, spread: float = 1.0) -> tuple[float, float]:
    """Fit ``(a, b)`` for the UMAP low-dim kernel ``1 / (1 + a·x^(2b))``.

    Matches :func:`umap.umap_.find_ab_params`.
    """
    from fast_auto_scrna._native import umap as _native_umap
    return _native_umap.fit_ab(float(min_dist), float(spread))


def _noisy_scale_coords(
    coords: np.ndarray, rng: np.random.Generator,
    max_coord: float = 10.0, noise: float = 1e-4,
) -> np.ndarray:
    """Match :func:`umap.umap_.noisy_scale_coords`: rescale so ``max|x|=10``
    and add σ=1e-4 Gaussian noise to break eigenvector ties.
    """
    max_abs = float(np.abs(coords).max())
    if max_abs > 0:
        coords = coords * (max_coord / max_abs)
    coords = coords + (noise * rng.standard_normal(coords.shape)).astype(np.float32)
    return np.ascontiguousarray(coords.astype(np.float32))


def _spectral_init(graph, n_components: int, random_state: int):
    """Spectral layout = bottom ``n_components`` non-trivial eigenvectors of
    the normalized Laplacian. Delegates to umap-learn for bit-exact parity.
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
    except Exception as e:
        warnings.warn(
            f"spectral init failed ({type(e).__name__}: {e}); falling back to PCA init",
            RuntimeWarning, stacklevel=2,
        )
        return None


def _pick_init(
    adata, init: str, n_components: int, random_state: int,
    graph=None,
) -> np.ndarray:
    """Pick an initial low-dim embedding for UMAP."""
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
        init = "pca"
        warnings.warn(
            "spectral init unavailable; using PCA init as fallback",
            RuntimeWarning, stacklevel=2,
        )

    if init in ("pca", "X_pca"):
        if "X_pca" not in adata.obsm:
            raise ValueError(
                "init='pca' but adata.obsm['X_pca'] missing — run pca first"
            )
        pc = np.ascontiguousarray(
            adata.obsm["X_pca"][:, :n_components], dtype=np.float32
        )
        return _noisy_scale_coords(pc, rng)

    raise ValueError(f"unknown init {init!r} — use 'spectral', 'pca', or 'random'")


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
    symmetric fuzzy-simplicial-set output. Writes ``adata.obsm['X_umap']``.
    """
    from fast_auto_scrna._native import umap as _native_umap

    if copy:
        adata = adata.copy()

    conn_key = f"{neighbors_key}_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"adata.obsp['{conn_key}'] missing — run integration first"
        )

    C = adata.obsp[conn_key]
    if not _sp.issparse(C):
        C = _sp.csr_matrix(C)
    C = C.tocsr()
    n = C.shape[0]
    if n != adata.n_obs:
        raise ValueError(
            f"connectivities shape {C.shape} ≠ adata.n_obs={adata.n_obs}"
        )

    indptr = np.asarray(C.indptr, dtype=np.uint64)
    indices = np.asarray(C.indices, dtype=np.uint32)
    data = np.ascontiguousarray(C.data, dtype=np.float32)

    init_emb = _pick_init(adata, init, n_components, random_state, graph=C)

    embedding, a, b, n_epochs_used = _native_umap.umap_layout(
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
