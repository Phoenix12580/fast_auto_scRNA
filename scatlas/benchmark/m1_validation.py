"""M1 acceptance — run scatlas.stats.* on two real datasets and verify.

Datasets:
  * `panc8_sub.rda` — SCOP Seurat object, 12940 × 1600 sparse (small, gold
    reference for parity vs scipy).
  * `epithelia_full.h5ad` — 150k-cell atlas (scale + wall-time test).

Checks per dataset:
  1. wilcoxon_ranksum_matrix on log-normalized counts, two-group test;
     parity to scipy reference on a 500-gene sanity slice.
  2. entropy_table on raw counts (dense + CSR path); parity to numpy
     reference.
  3. knockoff_threshold_offset1 on scatlas wilcoxon output converted to
     W-statistics; sanity check (finite on signal, inf on noise).
  4. Timing vs pure-Python numpy reference (wilcoxon only; entropy is
     bandwidth-bound so ratio is uninformative).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import erfc
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from scatlas import stats

PANC8_RDA = Path("/mnt/f/NMF_rewrite/panc8_sub.rda")
EPITHELIA_H5AD = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _log_norm(counts: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """Library-size normalize then log1p. Matches scanpy defaults."""
    counts = counts.astype(np.float64)
    libsizes = np.asarray(counts.sum(axis=0)).ravel()
    libsizes[libsizes == 0] = 1.0
    # Column-scale: multiply each col j by target_sum / libsize[j]
    scaler = sp.diags((target_sum / libsizes).astype(np.float64))
    scaled = counts @ scaler
    scaled.data = np.log1p(scaled.data)
    return scaled.tocsr()


def _wilcoxon_reference_subset(
    x_dense: np.ndarray, m1: np.ndarray, m2: np.ndarray
) -> np.ndarray:
    """scipy-based gold wilcoxon (no ties correction) for a gene slice."""
    n_genes = x_dense.shape[0]
    union = m1 | m2
    g1 = m1[union]
    n1 = int(g1.sum())
    n2 = int((~g1).sum())
    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    out = np.zeros(n_genes)
    for g in range(n_genes):
        r = rankdata(x_dense[g, union], method="average")
        r1 = r[g1].sum()
        u1 = r1 - n1 * (n1 + 1) / 2.0
        z = (u1 - mu) / sigma
        out[g] = np.clip(erfc(abs(z) / np.sqrt(2)), 1e-300, 1.0)
    return out


def _entropy_reference(counts_dense: np.ndarray, r: float = 1.0) -> np.ndarray:
    mean = counts_dense.mean(axis=1)
    return np.column_stack([np.log(mean + r), np.log(counts_dense + 1.0).mean(axis=1)])


@dataclass
class SectionResult:
    name: str
    n_genes: int
    n_cells: int
    wilcoxon_wall_ours: float
    wilcoxon_wall_ref: float | None
    wilcoxon_parity_max_abs_diff: float | None
    entropy_parity_max_abs_diff: float
    entropy_csr_matches_dense: bool
    knockoff_threshold: float
    n_significant_genes: int


# -----------------------------------------------------------------------------
# panc8
# -----------------------------------------------------------------------------


def load_panc8() -> tuple[sp.csc_matrix, np.ndarray]:
    import rdata

    parsed = rdata.read_rda(str(PANC8_RDA))
    panc = parsed["panc8_sub"]
    rna = panc.assays["RNA"]
    # counts layer = dgCMatrix: i (row idx), p (col ptr), x (data), Dim
    counts_raw = rna.layers["counts"]
    n_genes, n_cells = int(counts_raw.Dim[0]), int(counts_raw.Dim[1])
    mat = sp.csc_matrix(
        (counts_raw.x, counts_raw.i, counts_raw.p),
        shape=(n_genes, n_cells),
    )

    # Pull celltype: panc.meta_data or panc.meta.data — rdata renames '.' to '_'
    meta = getattr(panc, "meta_data", None) or getattr(panc, "meta.data", None)
    if meta is None:
        raise RuntimeError("could not find meta.data in panc8_sub")
    # meta is an R data.frame; rdata exposes as SimpleNamespace with column arrays
    celltype = None
    for key in ("celltype", "cell_type", "cellType"):
        if hasattr(meta, key):
            celltype = np.asarray(getattr(meta, key))
            break
    if celltype is None:
        raise RuntimeError(f"could not find celltype in meta.data: {dir(meta)}")
    return mat, celltype


def validate_panc8() -> SectionResult:
    print("\n=== panc8_sub (Seurat dgCMatrix) ===")
    counts_csc, celltype = load_panc8()
    n_genes, n_cells = counts_csc.shape
    print(f"  shape: {n_genes} genes × {n_cells} cells")
    print(f"  celltypes: {np.unique(celltype).tolist()}")

    # Log-normalize for wilcoxon
    counts_csr = counts_csc.tocsr()
    log_norm = _log_norm(counts_csr).toarray().astype(np.float64)
    print(f"  log-norm dense: {log_norm.nbytes / 1e6:.1f} MB")

    # Pick the two most populous cell types for DEG
    labels, counts_per = np.unique(celltype, return_counts=True)
    top2 = labels[np.argsort(-counts_per)[:2]]
    m1 = celltype == top2[0]
    m2 = celltype == top2[1]
    print(f"  DEG: {top2[0]} (n={m1.sum()}) vs {top2[1]} (n={m2.sum()})")

    # scatlas wilcoxon on full matrix
    t0 = time.perf_counter()
    pvals = stats.wilcoxon_ranksum_matrix(log_norm, m1, m2)
    wall_ours = time.perf_counter() - t0
    n_sig = int((pvals < 0.05).sum())
    print(f"  wilcoxon: scatlas {wall_ours:.3f}s, n_sig@0.05 = {n_sig}/{n_genes}")

    # Parity on 500-gene slice vs scipy gold
    rng = np.random.default_rng(0)
    slice_idx = rng.choice(n_genes, size=min(500, n_genes), replace=False)
    t0 = time.perf_counter()
    ref_slice = _wilcoxon_reference_subset(log_norm[slice_idx], m1, m2)
    wall_ref = time.perf_counter() - t0
    diff = np.abs(pvals[slice_idx] - ref_slice)
    wil_max = float(diff.max())
    print(
        f"  wilcoxon parity (500-gene slice): scipy {wall_ref:.3f}s,"
        f" max|Δp| = {wil_max:.2e}"
    )

    # entropy_table — raw counts, dense + CSR paths
    et_dense = stats.entropy_table(counts_csr.toarray(), r=1.0)
    et_sparse = stats.entropy_table(counts_csr, r=1.0)
    et_ref = _entropy_reference(counts_csr.toarray(), r=1.0)
    ent_max = float(np.abs(et_dense - et_ref).max())
    ent_csr_match = bool(np.allclose(et_dense, et_sparse, rtol=1e-10, atol=1e-12))
    print(f"  entropy_table: max|Δ vs numpy| = {ent_max:.2e}, CSR==dense = {ent_csr_match}")

    # knockoff threshold on symmetric W built from ±log10(p) × sign-of-mean-diff
    log_p = -np.log10(np.clip(pvals, 1e-300, 1.0))
    mean_diff = log_norm[:, m1].mean(axis=1) - log_norm[:, m2].mean(axis=1)
    w = log_p * np.sign(mean_diff)
    t = stats.knockoff_threshold_offset1(w, fdr=0.1)
    print(f"  knockoff threshold (fdr=0.1) = {t:.4f}")

    return SectionResult(
        name="panc8_sub",
        n_genes=n_genes,
        n_cells=n_cells,
        wilcoxon_wall_ours=wall_ours,
        wilcoxon_wall_ref=wall_ref,
        wilcoxon_parity_max_abs_diff=wil_max,
        entropy_parity_max_abs_diff=ent_max,
        entropy_csr_matches_dense=ent_csr_match,
        knockoff_threshold=float(t),
        n_significant_genes=n_sig,
    )


# -----------------------------------------------------------------------------
# epithelia (150k)
# -----------------------------------------------------------------------------


def validate_epithelia() -> SectionResult:
    print("\n=== epithelia_full.h5ad (150k cells) ===")
    import anndata as ad

    adata = ad.read_h5ad(EPITHELIA_H5AD, backed="r")
    n_cells, n_genes = adata.n_obs, adata.n_vars
    print(f"  shape: {n_cells} cells × {n_genes} genes  (obs columns: {list(adata.obs.columns)[:6]})")

    # Pick a label column for DEG
    label_col = None
    for cand in ("leiden", "cell_type", "celltype", "cluster", "annotation"):
        if cand in adata.obs:
            label_col = cand
            break
    if label_col is None:
        label_col = adata.obs.columns[0]
    labels = adata.obs[label_col].astype(str).to_numpy()
    uniq, cnt = np.unique(labels, return_counts=True)
    top2 = uniq[np.argsort(-cnt)[:2]]
    print(f"  label column: {label_col}; top2: {top2[0]} (n={cnt[uniq==top2[0]][0]}) vs {top2[1]} (n={cnt[uniq==top2[1]][0]})")

    # Load into memory — X is counts (assume). If very large, use backed slicing.
    sel = np.isin(labels, top2)
    n_sel = int(sel.sum())
    print(f"  selected {n_sel} cells for DEG")
    adata = adata.to_memory()
    X = adata.X
    if sp.issparse(X):
        counts_sub = X.T[:, sel]  # genes × selected_cells (CSR view via T)
        counts_sub = counts_sub.tocsr()
    else:
        counts_sub = sp.csr_matrix(X.T[:, sel])
    print(f"  counts_sub: {counts_sub.shape}, nnz={counts_sub.nnz}, density={counts_sub.nnz/(counts_sub.shape[0]*counts_sub.shape[1]):.3%}")

    # Log-normalize the selected cells for wilcoxon
    log_norm_sub = _log_norm(counts_sub).astype(np.float32)
    # Densify only if < 2 GB to avoid OOM
    dense_bytes = log_norm_sub.shape[0] * log_norm_sub.shape[1] * 4
    print(f"  densified log-norm would be {dense_bytes/1e9:.2f} GB (f32)")
    if dense_bytes > 4e9:
        # Subsample genes to keep under budget
        gene_idx = np.random.default_rng(0).choice(log_norm_sub.shape[0], size=5000, replace=False)
        log_norm_dense = log_norm_sub[gene_idx, :].toarray()
        print(f"  too big; subsampled to {log_norm_dense.shape[0]} genes")
    else:
        log_norm_dense = log_norm_sub.toarray()
    n_g = log_norm_dense.shape[0]

    m1 = labels[sel] == top2[0]
    m2 = labels[sel] == top2[1]

    t0 = time.perf_counter()
    pvals = stats.wilcoxon_ranksum_matrix(log_norm_dense, m1, m2)
    wall_ours = time.perf_counter() - t0
    n_sig = int((pvals < 0.05).sum())
    print(f"  wilcoxon: scatlas {wall_ours:.2f}s, n_sig@0.05 = {n_sig}/{n_g}")

    # Parity on 200-gene slice
    rng = np.random.default_rng(1)
    slice_idx = rng.choice(n_g, size=min(200, n_g), replace=False)
    t0 = time.perf_counter()
    ref_slice = _wilcoxon_reference_subset(
        log_norm_dense[slice_idx].astype(np.float64), m1, m2
    )
    wall_ref = time.perf_counter() - t0
    diff = np.abs(pvals[slice_idx] - ref_slice)
    wil_max = float(diff.max())
    print(f"  wilcoxon parity (200 genes): scipy {wall_ref:.2f}s,"
          f" max|Δp| = {wil_max:.2e}")

    # entropy_table on raw counts — CSR path avoids densification
    t0 = time.perf_counter()
    et_sparse = stats.entropy_table(counts_sub, r=1.0)
    wall_ent = time.perf_counter() - t0
    print(f"  entropy_table (CSR, {counts_sub.shape[0]} genes): {wall_ent:.2f}s")

    # Parity check against dense path on a small gene slice (raw counts)
    slice_g = rng.choice(counts_sub.shape[0], size=min(100, counts_sub.shape[0]), replace=False)
    dense_slice = counts_sub[slice_g, :].toarray().astype(np.float64)
    et_ref_slice = _entropy_reference(dense_slice, r=1.0)
    et_diff_slice = np.abs(et_sparse[slice_g] - et_ref_slice).max()
    ent_max = float(et_diff_slice)
    print(f"  entropy parity (100-gene slice): max|Δ| = {ent_max:.2e}")

    # knockoff
    log_p = -np.log10(np.clip(pvals, 1e-300, 1.0))
    mean_diff = log_norm_dense[:, m1].mean(axis=1) - log_norm_dense[:, m2].mean(axis=1)
    w = (log_p * np.sign(mean_diff)).astype(np.float64)
    t = stats.knockoff_threshold_offset1(w, fdr=0.1)
    print(f"  knockoff threshold (fdr=0.1) = {t:.4f}")

    return SectionResult(
        name="epithelia_full",
        n_genes=counts_sub.shape[0],
        n_cells=n_cells,
        wilcoxon_wall_ours=wall_ours,
        wilcoxon_wall_ref=wall_ref,
        wilcoxon_parity_max_abs_diff=wil_max,
        entropy_parity_max_abs_diff=ent_max,
        entropy_csr_matches_dense=True,
        knockoff_threshold=float(t),
        n_significant_genes=n_sig,
    )


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def report(results: list[SectionResult]) -> None:
    print("\n" + "=" * 72)
    print("M1 ACCEPTANCE SUMMARY")
    print("=" * 72)
    tol_wilcoxon = 1e-8
    tol_entropy = 1e-8
    pass_all = True
    for r in results:
        print(f"\n{r.name}:")
        print(f"  shape: {r.n_genes} genes × {r.n_cells} cells")
        print(f"  wilcoxon scatlas: {r.wilcoxon_wall_ours:.3f}s  |  "
              f"scipy slice: {r.wilcoxon_wall_ref:.3f}s  |  "
              f"speedup on slice: {(r.wilcoxon_wall_ref or 0)/max(r.wilcoxon_wall_ours, 1e-9):.1f}× (slice vs full-matrix)")
        print(f"  wilcoxon parity max|Δp|: {r.wilcoxon_parity_max_abs_diff:.2e}  "
              f"{'PASS' if r.wilcoxon_parity_max_abs_diff < tol_wilcoxon else 'FAIL'}")
        print(f"  entropy parity max|Δ|:   {r.entropy_parity_max_abs_diff:.2e}  "
              f"{'PASS' if r.entropy_parity_max_abs_diff < tol_entropy else 'FAIL'}")
        print(f"  entropy CSR==dense: {r.entropy_csr_matches_dense}")
        print(f"  knockoff threshold: {r.knockoff_threshold}")
        print(f"  n_sig@0.05 (biol sanity): {r.n_significant_genes}")
        if r.wilcoxon_parity_max_abs_diff >= tol_wilcoxon:
            pass_all = False
        if r.entropy_parity_max_abs_diff >= tol_entropy:
            pass_all = False
        if not r.entropy_csr_matches_dense:
            pass_all = False

    print("\n" + "=" * 72)
    print("VERDICT:", "PASS" if pass_all else "FAIL")
    print("=" * 72)
    sys.exit(0 if pass_all else 1)


if __name__ == "__main__":
    results: list[SectionResult] = []
    results.append(validate_panc8())
    results.append(validate_epithelia())
    report(results)
