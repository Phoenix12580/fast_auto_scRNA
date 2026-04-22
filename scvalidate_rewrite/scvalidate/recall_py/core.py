"""recall orchestration — the outer loop that reduces resolution until no
cluster pair is knockoff-indistinguishable.

R: ``FindClustersRecall`` (recall_main.R L215–L377).

Design choice for Python: we replace Seurat's (Normalize, FindVariableFeatures,
ScaleData, RunPCA, FindNeighbors, FindClusters) pipeline with the scanpy
equivalents. Functionally equivalent but with scanpy defaults, which may yield
minor numerical differences — the verdict-level consistency target (≥0.95) is
the pass criterion, not bit-identical clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from scvalidate.recall_py.knockoff import (
    generate_knockoff_matrix,
    knockoff_threshold_offset1,
)

try:
    import scvalidate_rust as _rust
    _RUST_WILCOXON = _rust.wilcoxon_ranksum_matrix
except ImportError:
    _RUST_WILCOXON = None


@dataclass
class RecallResult:
    """Output of :func:`find_clusters_recall`.

    Attributes
    ----------
    labels
        Cluster label per cell (int array, length n_cells).
    resolution
        Final Leiden resolution at which no cluster pair was merged.
    n_iterations
        Number of outer-loop iterations executed.
    per_cluster_pass
        Dict mapping cluster_id → bool recall-gate pass. A cluster "passes" if,
        for every other cluster, the knockoff filter selected ≥1 real gene with
        W ≥ t at the given FDR. A cluster fails if **any** pair produced 0
        selections at the final (converged) resolution — which, after
        convergence, should be empty by construction, hence all pass=True.
        The dict is retained for downstream fuse() compatibility.
    """

    labels: np.ndarray
    resolution: float
    n_iterations: int
    per_cluster_pass: dict[int, bool]
    # v1 additions:
    resolution_trajectory: list[float] = field(default_factory=list)
    k_trajectory: list[int] = field(default_factory=list)
    converged: bool = False


def _scanpy_preprocess_and_cluster(
    counts_gxc: np.ndarray,
    resolution: float,
    dims: int,
    n_variable_features: int,
    algorithm: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Run normalize → log1p → HVG → scale → PCA → neighbors → leiden.

    Returns ``(cluster_labels, n_clusters)``.

    We construct a minimal AnnData internally so the orchestrator can be fed
    a raw counts matrix without requiring callers to set up an AnnData.
    """
    import scanpy as sc
    from anndata import AnnData

    # scanpy expects cells × genes
    adata = AnnData(counts_gxc.T.astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Select up to n_variable_features HVGs (falls back to all genes when
    # fewer are available, e.g. small synthetic tests).
    n_hvg = min(n_variable_features, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    # PCA: cap n_comps to what the data supports
    n_pcs = min(dims, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(adata, n_pcs=n_pcs, random_state=seed)

    if algorithm == "leiden":
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        labels = adata.obs["leiden"].astype(int).to_numpy()
    elif algorithm == "louvain":
        sc.tl.louvain(adata, resolution=resolution, random_state=seed)
        labels = adata.obs["louvain"].astype(int).to_numpy()
    else:
        raise ValueError(f"algorithm must be 'leiden' or 'louvain', got {algorithm!r}")

    return labels, int(labels.max()) + 1


def _wilcoxon_per_gene(
    counts_gxc_log: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> np.ndarray:
    """Per-gene Wilcoxon rank-sum p-values between two cell groups.

    Vectorized implementation: rank across both groups per gene, compute U, then
    normal-approximation two-sided p-value.

    When ``scvalidate_rust`` is available, delegates to the Rust kernel
    (rayon-parallel, 50–100× faster on recall's hot path). Otherwise falls
    back to the pure-Python scipy path below, which is the reference
    implementation and the source of truth for parity tests.
    """
    if _RUST_WILCOXON is not None:
        # Rust kernel accepts both f32 and f64 via dtype dispatch, and ArrayView
        # handles strided input. Upcast only if dtype is neither — keeps the
        # large log_counts matrix in its original precision, avoiding 20-40 GB
        # copies per pair at 157k scale.
        x = counts_gxc_log
        if x.dtype != np.float64 and x.dtype != np.float32:
            x = x.astype(np.float64, copy=False)
        m1 = mask1 if mask1.dtype == bool else mask1.astype(bool)
        m2 = mask2 if mask2.dtype == bool else mask2.astype(bool)
        return np.asarray(_RUST_WILCOXON(x, m1, m2))

    from scipy.stats import rankdata

    g_all = counts_gxc_log[:, mask1 | mask2]
    # Build a boolean mask over the concatenated cells indicating group 1.
    combined_mask = np.zeros(mask1.sum() + mask2.sum(), dtype=bool)
    # Concatenation order inside g_all matches the column selection above —
    # cells with mask1 first, then mask2, BUT only when they're ordered so.
    # boolean indexing preserves original order, so we need to map carefully:
    # reconstruct the cell-index order from the boolean mask:
    orig_indices = np.where(mask1 | mask2)[0]
    combined_mask = mask1[orig_indices]

    n1 = int(combined_mask.sum())
    n2 = int((~combined_mask).sum())
    if n1 == 0 or n2 == 0:
        return np.ones(counts_gxc_log.shape[0])

    # rankdata over axis=1 for each gene
    ranks = np.apply_along_axis(rankdata, 1, g_all)
    r1 = ranks[:, combined_mask].sum(axis=1)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    # Normal approximation (ignore ties correction — good enough for FDR
    # calibration since the knockoff procedure is rank-based itself)
    mu_u = n1 * n2 / 2.0
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u1 - mu_u) / sigma_u
    # Two-sided p-value
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(np.abs(z)))
    # Floor zero p-values (log10 would -> inf); mirror Seurat's ε
    p = np.clip(p, 1e-300, 1.0)
    return p


def _compute_W_for_pair(
    log_counts_gxc: np.ndarray,
    n_real: int,
    labels: np.ndarray,
    c1: int,
    c2: int,
) -> np.ndarray:
    """Compute knockoff W = -log10(p_real) - -log10(p_knockoff) per real gene.

    R: ``compute_knockoff_filter`` (recall_main.R L103–L173). Assumes
    ``log_counts_gxc`` is the augmented (real + knockoff) log-normalized matrix
    with the first ``n_real`` rows being the real genes.
    """
    mask1 = labels == c1
    mask2 = labels == c2
    pvals = _wilcoxon_per_gene(log_counts_gxc, mask1, mask2)

    p_real = pvals[:n_real]
    p_knock = pvals[n_real:]
    # Knockoffs should equal n_real in count by construction
    if len(p_knock) != n_real:
        raise ValueError(
            f"Knockoff row count ({len(p_knock)}) != real ({n_real}) — "
            f"matrix augmentation went wrong."
        )
    return -np.log10(p_real) - (-np.log10(p_knock))


def find_clusters_recall(
    counts_gxc,
    resolution_start: float = 0.8,
    reduction_percentage: float = 0.2,
    dims: int = 10,
    algorithm: str = "leiden",
    null_method: str = "ZIP",
    n_variable_features: int = 2000,
    fdr: float = 0.05,
    max_iterations: int = 20,
    seed: int | None = 0,
    verbose: bool = True,
    backend: str = "auto",
    scratch_dir: Path | None = None,
    oom_threshold_cells: int = 30_000,
) -> RecallResult:
    """Iteratively reduce resolution until no cluster pair is null-dominated.

    R: ``FindClustersRecall`` (recall_main.R L215–L377).

    Parameters
    ----------
    counts_gxc
        Genes × cells raw counts (dense or sparse).
    resolution_start
        Starting Leiden resolution.
    reduction_percentage
        Fraction by which to shrink resolution each retry (R default 0.2 →
        ``res_new = res * 0.8``).
    dims
        Number of PCs to use for the neighbor graph.
    algorithm
        "leiden" (default) or "louvain".
    null_method
        "ZIP" (default) or "NB" — the null distribution for knockoff draws.
    n_variable_features
        HVG count for the Python scanpy wrapper (R uses VariableFeatures).
    fdr
        Nominal FDR for Barber-Candès knockoff threshold (R default 0.05).
    max_iterations
        Safety cap on the outer loop. Default 20.

        Divergence from R: the upstream R ``FindClustersRecall`` has no
        iteration cap and runs until no pair is knockoff-indistinguishable.
        On large, homogeneous datasets this can collapse to a single
        cluster (epithelia 10k bench: R → 1 cluster, Py with
        ``max_iterations=20`` → 21 clusters). The Python cap is deliberate:
        biological ARI vs ground-truth subtype on that bench was 0.26
        (Py) vs 0.00 (R), so the early stop is a net positive. If you
        need exact R parity for regression testing, pass a large value
        (e.g. 100).
    seed
        Reproducibility seed.
    backend
        "auto" (default), "dense", or "oom". Auto picks based on n_cells vs
        oom_threshold_cells. Dense uses the in-memory path. Oom uses the
        anndata-oom backed path (Task 5+).
    scratch_dir
        Path | None. Scratch directory for the oom backend (ignored for dense).
    oom_threshold_cells
        Cell count threshold for auto backend selection. Default 30_000 —
        below this, in-memory processing is faster than disk I/O; above it,
        oom's memory savings outweigh the ~40s first-read overhead (empirical,
        hardware-dependent — see spec §6 risk table).

    Returns
    -------
    :class:`RecallResult`
    """
    # -- backend resolution --
    if backend == "auto":
        n_cells = counts_gxc.shape[1] if hasattr(counts_gxc, "shape") else None
        backend = "oom" if (n_cells is not None and n_cells >= oom_threshold_cells) else "dense"
    if backend not in ("dense", "oom"):
        raise ValueError(f"backend must be 'auto'/'dense'/'oom', got {backend!r}")

    if backend == "oom":
        from scvalidate.recall_py._oom_backend import find_clusters_recall_oom
        return find_clusters_recall_oom(
            counts_gxc,
            resolution_start=resolution_start,
            reduction_percentage=reduction_percentage,
            dims=dims,
            algorithm=algorithm,
            null_method=null_method,
            n_variable_features=n_variable_features,
            fdr=fdr,
            max_iterations=max_iterations,
            seed=seed,
            verbose=verbose,
            scratch_dir=scratch_dir,
        )

    # -- dense path (existing code below, unchanged) --
    if sp.issparse(counts_gxc):
        counts = np.asarray(counts_gxc.todense())
    else:
        counts = np.asarray(counts_gxc)
    counts = counts.astype(np.int64)
    n_genes = counts.shape[0]

    # 1. Generate knockoffs and augment: [real | knockoff] rows, 2G × n_cells.
    # For memory efficiency at 157k (where int64 augmented = 40 GB, log_counts
    # f64 = 40 GB, total > 100 GB w/ temps), we downcast to int32/f32 for the
    # large intermediates. Safe: per-gene per-cell counts are well under 2^31;
    # Rust Wilcoxon dispatches on dtype and accepts f32.
    knock_i64 = generate_knockoff_matrix(
        counts, null_method=null_method, seed=seed, verbose=verbose
    )
    augmented = np.concatenate(
        [counts.astype(np.int32, copy=False), knock_i64.astype(np.int32, copy=False)],
        axis=0,
    )
    del knock_i64

    # 2. Pre-compute log-normalized augmented matrix for W statistics.
    # Library-size normalize to 1e4 then log1p — matches scanpy semantics.
    # Build log_counts as f32 in-place to halve the 2G × n footprint.
    col_sums = augmented.sum(axis=0, dtype=np.int64)
    col_sums = np.where(col_sums == 0, 1.0, col_sums).astype(np.float32)
    log_counts = np.empty(augmented.shape, dtype=np.float32)
    np.multiply(augmented, np.float32(1e4), out=log_counts, casting="unsafe")
    log_counts /= col_sums
    np.log1p(log_counts, out=log_counts)

    resolution = float(resolution_start)
    n_iter = 0
    last_labels: np.ndarray | None = None
    resolution_trajectory: list[float] = []
    k_trajectory: list[int] = []
    converged = False

    while n_iter < max_iterations:
        n_iter += 1
        resolution_trajectory.append(resolution)
        if verbose:
            print(f"[recall] iter {n_iter}: resolution={resolution:.4f}")

        labels, k = _scanpy_preprocess_and_cluster(
            augmented,
            resolution=resolution,
            dims=dims,
            n_variable_features=2 * n_variable_features,
            algorithm=algorithm,
            seed=seed if seed is not None else 0,
        )
        last_labels = labels
        k_trajectory.append(k)

        if k < 2:
            if verbose:
                print(f"[recall] single cluster — stopping")
            converged = True
            break

        # 3. For every (i < j) pair: compute W → check selection count.
        found_merged_pair = False
        for i in range(k):
            for j in range(i):
                w = _compute_W_for_pair(log_counts, n_genes, labels, i, j)
                t = knockoff_threshold_offset1(w, fdr=fdr)
                n_selected = int((w >= t).sum()) if np.isfinite(t) else 0
                if n_selected == 0:
                    found_merged_pair = True
                    if verbose:
                        print(f"[recall]   pair ({i},{j}) 0 selected → reduce res")
                    break
            if found_merged_pair:
                break

        if not found_merged_pair:
            converged = True
            if verbose:
                print(f"[recall] converged at k={k} clusters")
            break

        resolution = (1 - reduction_percentage) * resolution

    assert last_labels is not None
    unique = np.unique(last_labels)
    per_cluster_pass = {int(c): True for c in unique}

    return RecallResult(
        labels=last_labels,
        resolution=resolution,
        n_iterations=n_iter,
        per_cluster_pass=per_cluster_pass,
        resolution_trajectory=resolution_trajectory,
        k_trajectory=k_trajectory,
        converged=converged,
    )
