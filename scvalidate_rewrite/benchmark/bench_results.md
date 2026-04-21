# scvalidate benchmark — pancreas_sub (1000 cells, 15998 genes)

Dataset: `pancreas_sub.rda` (Seurat v5 Assay5), single-batch UMI data,
8 leiden clusters, SubCellType ground-truth with 8 levels.

## Current status (v0.3-dev, post-loess + eigsh + matr.filter fixes)

Wall clock, single-thread on Windows 10 Pro:

| stage | start of session | after fixes | Δ |
|---|---:|---:|---|
| load | 0.3s | 0.3s | — |
| baseline leiden (scanpy) | 19.8s | 20.0s | — |
| recall | 188s | 194s | — |
| sc-SHC (testClusters) | 109s | 107s | 1.1× |
| ROGUE + markers | **217s** | **1.8s** | **120×** |
| **total** | **534s** | **323s** | **1.7×** |

Reference: R `scSHC::testClusters` alone = 260s (single-thread with Windows
`mclapply` monkey-patch; R's ROGUE adds ~3–5s per cluster).

## Accuracy vs R originals

### sc-SHC — `ARI = 1.0000` (perfect match)

- Python merges 8 → 7 clusters. R's `scSHC::testClusters` does the same.
- Crosstab is a pure permutation matrix.
- ARI(Python scshc_merged, SubCellType) = 0.5926 == ARI(R scshc_R, SubCellType).
- Gate-level consistency: same accept/merge decision for all 7 pairs tested.

Two port bugs found and fixed in `scshc_py/core.py::fit_model_batch`:

1. **RSpectra vs `eigh` semantics.** R's `RSpectra::eigs_sym(k=30)` returns the
   top-k eigenvalues by |λ|, so when `rhos` (a log-covariance that is not PSD)
   carries large *negative* eigenvalues, R keeps only the positives and the
   effective rank is often << 30. The Python port previously took
   top-30 *algebraic* positives, filling slots with small positives that R
   would have skipped → implied covariance inflated ~40% → null
   LogNormal-Poisson draws ran hot → null Ward stats 2-3× R's → uniform
   p ≈ 1 (every observed split looked null-like).

2. **`sfsmisc::posdefify` rescale step.** posdefify clips negative
   eigenvalues to ε *and* rescales so `diag(result) == diag(input)`. The
   initial Python port did only the clip; without the rescale the on-gene
   diagonal drifted after eigenvalue clipping and variances shifted.

After the fix: Python `on_cov_sqrt` Frobenius norm matches R exactly (29.04),
null Ward stat mean 144 (R: 149), p-value on root split 0.0000 (R: 0.0000).

### ROGUE — Pearson r = 0.993, Spearman r = 0.976

Per-leiden-cluster ROGUE (after adding `filter_matrix`):

| cluster | cells | Python | R | abs Δ |
|---:|---:|---:|---:|---:|
| 0 | 209 | 0.803 | 0.804 | 0.001 |
| 1 | 162 | 0.753 | 0.753 | 0.000 |
| 2 | 153 | 0.766 | 0.770 | 0.004 |
| 3 |  92 | 0.843 | 0.839 | 0.004 |
| 4 | 174 | 0.723 | 0.743 | 0.020 |
| 5 |  96 | 0.760 | 0.760 | 0.000 |
| 6 |  93 | 0.734 | 0.734 | 0.000 |
| 7 |  21 | 0.889 | 0.889 | 0.000 |

Mean abs diff = **0.0035** (was 0.022 before filter), max = 0.020 (cluster 4).
5 of 8 clusters match R to ≤ 0.001. Spearman 0.976, Pearson 0.993.

Two fixes gave this parity:

1. **Loess surface**: removed `lo.control.surface = "direct"` in
   `_loess_predict`. R's `loess()` defaults to `surface = "interpolate"`
   (fast cell-grid approximation); the port previously overrode to
   `"direct"` (exact per-point regression, ~100× slower) without justification.
   The interpolate path now clamps `x_new` to the fit range to avoid skmisc's
   "Extrapolation not allowed" error at boundary points — this matches R's
   boundary-extension behavior. **Speedup: 217s → 1.8s (120×).**

2. **`matr.filter` before `SE_fun`**: R's benchmark pipeline runs
   `matr.filter(min.cells=10, min.genes=10)` before entropy calculation.
   The Python pipeline was skipping this → a few hundred low-coverage genes
   leaked into the SE fit, uniformly depressing all ROGUE scores by ~2%.
   Adding the filter call drops mean abs diff from 0.022 to 0.0035.

### recall — no direct R parity yet, but biological validation holds

`lcrawlab/recall` R install blocked by `scDesign3` (Bioconductor dep).
Python flags leiden-1 (162 cells) and leiden-2 (153 cells) as
knockoff-null-indistinguishable (REJECT). Biological check against the
SubCellType annotation shows both REJECTed clusters are transcriptionally
**mixed**:

- leiden 1 (162 cells): 91 Ngn3-high-EP + 71 Pre-endocrine
- leiden 2 (153 cells): 80 Endocrine + 73 Pre-endocrine

So recall is correctly flagging clusters with no coherent cell-type
identity — the REJECTs are biologically justified, not a false positive.
Direct R parity check still useful as a regression guard; unblock options:
- fork `recall` and strip `scDesign3` (only the copula branch uses it, and
  we explicitly do not port copula), or
- accept the biological check as the accuracy validation for v0.3.

## Final verdict table (Python)

| cluster | recall_pass | scSHC_p | ROGUE (Py / R) | n_markers | verdict |
|---:|:---:|---:|:---:|---:|:---|
| 0 | ✓ | 0.000 | 0.80 / 0.80 | 1094 | HIGH |
| 1 | ✗ | 0.000 | 0.73 / 0.75 |  268 | REJECT (recall) |
| 2 | ✗ | 0.000 | 0.75 / 0.77 |  226 | REJECT (recall) |
| 3 | ✓ | 0.000 | 0.82 / 0.84 |  625 | HIGH |
| 4 | ✓ | 0.000 | 0.71 / 0.74 |  332 | HIGH |
| 5 | ✓ | 0.000 | 0.73 / 0.76 |  408 | HIGH |
| 6 | ✓ | 0.000 | 0.72 / 0.73 | 3029 | HIGH |
| 7 | ✓ | 0.000 | 0.85 / 0.89 |   55 | HIGH |

## Remaining hotspots

cProfile breakdown post-fix:

1. **sc-SHC `scipy.linalg.eigh`**: 71s of 110s (64%) — 131 calls on
   2000×2000 gram, one per null draw. Potential Rust targets: Lanczos
   via `faer` (matches R's RSpectra) for subset-by-index. Or: reduce
   null-draw count from the adaptive 10→50 schedule if early-exit p-values
   remain decisive.

2. **recall 194s**: not yet profiled cleanly (AnnData dtype mismatch in the
   harness). Main algorithmic cost is likely knockoff NB sampling + a full
   scanpy PCA/KNN/Leiden pass on 2G × N genes. Rust candidates: NB/ZIP
   sampling loops.

3. **scanpy baseline leiden (20s)**: already C-backed, not a Rust target.

Current total 330s vs R testClusters alone at 260s is the wrong comparison
(R doesn't include ROGUE/markers/recall in that 260s). A full
R-side scSHC + recall + ROGUE end-to-end benchmark is still TODO.
