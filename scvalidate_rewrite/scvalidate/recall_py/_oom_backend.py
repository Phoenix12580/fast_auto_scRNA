"""anndata-oom backed path for find_clusters_recall."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata
from anndata.io import write_elem
import pandas as pd
import h5py


def _write_augmented_h5ad(
    real_gxc: np.ndarray,
    knock_gxc: np.ndarray,
    out_path: Path,
    chunk_cells: int = 5000,
) -> None:
    """Write [real | knockoff] stacked gene-wise to disk as AnnData cells × (2G).

    Streams N cells in chunks so RAM peak ≈ 2G × chunk_cells × 4B.
    Stored transposed to match scanpy's cells × genes convention.
    Single-pass write: h5py streams X; anndata.io.write_elem attaches obs/var
    without ever loading X back into memory.
    """
    if real_gxc.shape != knock_gxc.shape:
        raise ValueError(
            f"real_gxc and knock_gxc must have the same shape, got "
            f"{real_gxc.shape} vs {knock_gxc.shape}"
        )
    G, N = real_gxc.shape
    # Build var table labelling real/knockoff rows.
    # IMPORTANT: indices MUST be dtype='object' or pandas 2.x + pyarrow can
    # infer ArrowStringArray, which anndata.write_h5ad cannot serialize
    # (raises IORegistryError on write). Explicit dtype='object' below.
    var = pd.DataFrame(
        {"is_knockoff": [False] * G + [True] * G},
        index=pd.Index(
            [f"g{i}" for i in range(G)] + [f"k{i}" for i in range(G)],
            dtype="object",
        ),
    )
    obs = pd.DataFrame(
        index=pd.Index([f"c{i}" for i in range(N)], dtype="object"),
    )

    # Single-pass write: stream X via h5py, attach obs/var via write_elem.
    # No Pass 2 read — avoids materialising (N × 2G × 4B) in RAM.
    try:
        with h5py.File(out_path, "w") as f:
            # Root attrs required by the AnnData on-disk format
            f.attrs["encoding-type"] = "anndata"
            f.attrs["encoding-version"] = "0.1.0"

            # X: chunked streaming write
            dset = f.create_dataset(
                "X", shape=(N, 2 * G), dtype=np.float32,
                chunks=(min(chunk_cells, N), 2 * G),
                compression=None,
            )
            dset.attrs["encoding-type"] = "array"
            dset.attrs["encoding-version"] = "0.2.0"
            for start in range(0, N, chunk_cells):
                end = min(start + chunk_cells, N)
                chunk = np.empty((end - start, 2 * G), dtype=np.float32)
                chunk[:, :G] = real_gxc[:, start:end].T.astype(np.float32, copy=False)
                chunk[:, G:] = knock_gxc[:, start:end].T.astype(np.float32, copy=False)
                dset[start:end, :] = chunk

            # obs + var via anndata.io.write_elem (encodes proper AnnData metadata)
            write_elem(f, "obs", obs)
            write_elem(f, "var", var)
            # Empty auxiliary groups for AnnData format completeness
            write_elem(f, "uns", {})
            write_elem(f, "obsm", {})
            write_elem(f, "varm", {})
            write_elem(f, "obsp", {})
            write_elem(f, "varp", {})
            write_elem(f, "layers", {})
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise


def _wilcoxon_pair_chunked_dense(
    log_counts_gxc: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    chunk: int = 2000,
) -> np.ndarray:
    """Chunked per-gene Wilcoxon on a dense ndarray, for parity testing.

    Real production path uses _wilcoxon_pair_chunked_oom (reads h5ad chunks).
    Both call the same Rust kernel.
    """
    from scvalidate.recall_py.core import _wilcoxon_per_gene
    G = log_counts_gxc.shape[0]
    out = np.empty(G, dtype=np.float64)
    for start in range(0, G, chunk):
        end = min(start + chunk, G)
        out[start:end] = _wilcoxon_per_gene(
            log_counts_gxc[start:end], mask1, mask2,
        )
    return out


def _wilcoxon_pair_chunked_oom(
    log_counts_adata,         # backed AnnData (cells x 2G)
    mask1: np.ndarray,        # cell-level boolean
    mask2: np.ndarray,
    chunk_genes: int = 2000,
) -> np.ndarray:
    """Stream gene chunks from a backed AnnData, run Rust Wilcoxon, concat p."""
    from scvalidate.recall_py.core import _wilcoxon_per_gene
    n_vars = log_counts_adata.n_vars  # == 2G
    out = np.empty(n_vars, dtype=np.float64)
    for start in range(0, n_vars, chunk_genes):
        end = min(start + chunk_genes, n_vars)
        # Pull dense slab genes[start:end] for all cells (cells x chunk).
        # Use sl_X[:] (BackedArray.__getitem__(slice(None))) to materialise —
        # np.asarray() on a _SubsetBackedArray hangs because the class has no
        # __array__ method and numpy's iteration fallback is unusably slow.
        sl_X = log_counts_adata[:, start:end].X
        if hasattr(sl_X, "toarray"):
            sl_dense = sl_X.toarray()
        elif hasattr(sl_X, "__getitem__"):
            sl_dense = sl_X[:]  # BackedArray[:] calls _read_rows(0, n_rows) → ndarray
        else:
            sl_dense = np.array(sl_X)
        # Convert to genes x cells for _wilcoxon_per_gene convention
        sl_gxc = np.asarray(sl_dense, dtype=np.float32).T
        out[start:end] = _wilcoxon_per_gene(sl_gxc, mask1, mask2)
    return out


def find_clusters_recall_oom(
    counts_gxc,
    *,
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
    scratch_dir=None,
):
    """anndata-oom backed find_clusters_recall.

    Keeps the augmented matrix on disk (scratch h5ad). Preprocess,
    PCA, neighbors computed once outside the while loop. Inside the loop
    only sc.tl.leiden re-runs. Wilcoxon streams gene chunks from disk.

    anndataoom 0.1.3 API differences from the plan's pseudo-code:
    - No oom.normalize_total / oom.log1p — use oom.chunked_normalize_total
      and oom.chunked_log1p (in-place mutations on AnnDataOOM).
    - HVG: oom.chunked_highly_variable_genes_pearson (not scanpy HVG on backed).
    - Scale: oom.chunked_scale stores result in adata.layers['scaled'].
    - PCA: oom.chunked_pca returns (X_pca, components, variance_ratio); we
      store X_pca in adata.obsm['X_pca'] then call sc.pp.neighbors.
    - Wilcoxon reads from a separate log-normalized view (pre-scale) to avoid
      distorted p-values from z-scored data.
    """
    import tempfile
    import anndataoom as oom
    import scanpy as sc

    from scvalidate.recall_py.knockoff import (
        generate_knockoff_matrix,
        knockoff_threshold_offset1,
    )
    from scvalidate.recall_py.core import RecallResult

    if sp.issparse(counts_gxc):
        counts = np.asarray(counts_gxc.todense())
    else:
        counts = np.asarray(counts_gxc)
    counts = counts.astype(np.int32)
    G, N = counts.shape

    # -- scratch --
    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
        tmpdir = Path(tmpdir)
        aug_path = tmpdir / "augmented.h5ad"

        if verbose:
            print(f"[recall/oom] generating knockoffs (G={G}, N={N}) ...")
        knock = generate_knockoff_matrix(
            counts, null_method=null_method, seed=seed, verbose=verbose,
        ).astype(np.int32)
        _write_augmented_h5ad(counts, knock, aug_path, chunk_cells=5000)
        del knock, counts  # reclaim RAM before oom.read

        if verbose:
            print(f"[recall/oom] oom.read + chunked preprocess ...")
        aug = oom.read(str(aug_path))  # backed AnnDataOOM, cells x 2G

        # In-place normalize + log1p (lazy TransformedBackedArray — no copy)
        oom.chunked_normalize_total(aug, target_sum=1e4)
        oom.chunked_log1p(aug)

        # HVG on augmented — allow 2 * n_variable_features because half are knockoffs.
        # Use oom's Pearson-residual chunked HVG (scanpy HVG on backed AnnData
        # would attempt to materialise X, which is too large).
        n_hvg = min(2 * n_variable_features, aug.n_vars)
        oom.chunked_highly_variable_genes_pearson(aug, n_top_genes=n_hvg)
        hvg_mask = aug.var["highly_variable"].to_numpy().astype(bool)
        hvg_indices = np.where(hvg_mask)[0]

        # Keep the full log-normalized backed object for Wilcoxon streaming.
        # Wilcoxon must run over ALL 2G genes (real + knockoff) to compute
        # W = -log10(p_real) - (-log10(p_knock)) per gene, mirroring the dense path.
        # 'aug' is already log-normalized (in-place above) — keep reference alive.
        aug_log_full = aug  # alias for clarity; same object

        # HVG-subsetted view for PCA/neighbors only.
        aug_scale = aug.subset(var_indices=hvg_indices, inplace=False)
        oom.chunked_scale(aug_scale, max_value=10)

        n_pcs = min(dims, len(hvg_indices) - 1, N - 1)
        if verbose:
            print(f"[recall/oom] chunked PCA (n_pcs={n_pcs}) ...")
        X_pca, _components, _var_ratio = oom.chunked_pca(
            aug_scale, n_comps=n_pcs, random_state=seed if seed is not None else 0,
        )

        # Build a minimal in-memory AnnData for neighbors + leiden
        # (only X_pca is needed — n_cells x n_pcs, small enough for RAM)
        from anndata import AnnData

        obs_names = [f"c{i}" for i in range(N)]
        adata_pca = AnnData(
            X=np.zeros((N, 1), dtype=np.float32),  # placeholder X (not used after neighbors)
        )
        adata_pca.obs_names = obs_names
        adata_pca.obsm["X_pca"] = X_pca.astype(np.float32)

        if verbose:
            print(f"[recall/oom] computing neighbors ...")
        sc.pp.neighbors(adata_pca, n_pcs=n_pcs, use_rep="X_pca", random_state=seed)

        # While loop: only leiden + chunked Wilcoxon on log-normalized backed data
        resolution = float(resolution_start)
        n_iter = 0
        last_labels = None
        resolution_trajectory: list[float] = []
        k_trajectory: list[int] = []
        converged = False
        n_real = G

        while n_iter < max_iterations:
            n_iter += 1
            resolution_trajectory.append(resolution)
            if verbose:
                print(f"[recall/oom] iter {n_iter}: res={resolution:.4f}")

            if algorithm == "leiden":
                sc.tl.leiden(
                    adata_pca, resolution=resolution, random_state=seed,
                    flavor="igraph", n_iterations=2, directed=False,
                )
                labels = adata_pca.obs["leiden"].astype(int).to_numpy()
            elif algorithm == "louvain":
                sc.tl.louvain(adata_pca, resolution=resolution, random_state=seed)
                labels = adata_pca.obs["louvain"].astype(int).to_numpy()
            else:
                raise ValueError(f"algorithm {algorithm!r}")
            last_labels = labels
            k = int(labels.max()) + 1
            k_trajectory.append(k)

            if k < 2:
                if verbose:
                    print(f"[recall/oom] single cluster — stop")
                converged = True
                break

            found_merged = False
            for i in range(k):
                for j in range(i):
                    m1 = labels == i
                    m2 = labels == j
                    # Stream Wilcoxon over the full log-normalized backed data (2G genes).
                    # p_real = first G p-values; p_knock = last G p-values.
                    # Mirrors the dense path: pvals[:n_real] / pvals[n_real:].
                    pvals = _wilcoxon_pair_chunked_oom(aug_log_full, m1, m2)
                    p_real = pvals[:n_real]
                    p_knock = pvals[n_real:]
                    w = -np.log10(p_real) - (-np.log10(p_knock))
                    t = knockoff_threshold_offset1(w, fdr=fdr)
                    n_sel = int((w >= t).sum()) if np.isfinite(t) else 0
                    if n_sel == 0:
                        found_merged = True
                        if verbose:
                            print(f"[recall/oom]   pair ({i},{j}) 0 sel")
                        break
                if found_merged:
                    break

            if not found_merged:
                converged = True
                if verbose:
                    print(f"[recall/oom] converged at k={k}")
                break
            resolution = (1 - reduction_percentage) * resolution

        assert last_labels is not None
        per_cluster_pass = {int(c): True for c in np.unique(last_labels)}
        return RecallResult(
            labels=last_labels,
            resolution=resolution,
            n_iterations=n_iter,
            per_cluster_pass=per_cluster_pass,
            resolution_trajectory=resolution_trajectory,
            k_trajectory=k_trajectory,
            converged=converged,
        )
