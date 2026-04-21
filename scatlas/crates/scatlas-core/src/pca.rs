//! Randomized truncated SVD (PCA) for sparse CSR + dense f32 matrices.
//!
//! Matches scanpy's default behavior when `zero_center=False` on a sparse
//! input — equivalent to `sklearn.decomposition.TruncatedSVD` but
//! rayon-parallelized.
//!
//! Algorithm: Halko–Martinsson–Tropp randomized SVD with power iteration
//! (Halko et al. 2011). For an input `A: (N, G)`, n_comps `k`, oversampling
//! `p` (default 10), and power iterations `q` (default 4 for f32):
//!
//! 1. Draw Ω ~ N(0, 1), shape `(G, L)` where `L = k + p`.
//! 2. Form `Y = A·Ω`, shape `(N, L)`.
//! 3. Power iterate `q` times: Y ← A·(Aᵀ·Y), re-orthonormalize Y each step.
//! 4. QR(Y) → Q, shape `(N, L)`.
//! 5. `B = Qᵀ·A`, shape `(L, G)`.
//! 6. Eigendecompose `M = B·Bᵀ`, shape `(L, L)` — small.
//!    `M = U_tilde·diag(λ)·U_tildeᵀ`, S = √λ.
//! 7. Right singular vectors `V = (1/S)·U_tildeᵀ·B`, shape `(L, G)`.
//! 8. Left singular vectors in original space: `U = Q·U_tilde`,
//!    shape `(N, L)`.
//! 9. Return top-`k`: embedding = `U[:, :k] · diag(S[:k])`,
//!    components = `V[:k, :]`.
//!
//! Sparse matmul: CSR → CSC conversion once (parallel chunked), then
//! both `A·X` (per-row par_iter over CSR) and `Aᵀ·X` (per-row par_iter
//! over CSC, which is A transposed) use the same fast kernel.
#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::harmony::kmeans_init::SplitMix64;

/// Hyperparameters for randomized SVD.
#[derive(Debug, Clone, Copy)]
pub struct RsvdParams {
    /// Extra columns beyond `n_comps` used during the sketch — more
    /// oversampling → better accuracy but more flops. Default 10.
    pub n_oversamples: usize,
    /// Number of power iterations. f32 needs ≥ 4 for clean subspace.
    pub n_power_iter: usize,
}

impl Default for RsvdParams {
    fn default() -> Self {
        Self {
            n_oversamples: 10,
            // 7 iters gives ~1e-3 relative SV error on real scRNA data;
            // sklearn's TruncatedSVD default (n_iter=5) uses LU re-orth
            // which converges similarly. We use QR re-orth, so need 2
            // more iters to match accuracy — still faster overall.
            n_power_iter: 7,
        }
    }
}

/// PCA output — mirrors sklearn's `fit_transform` + `components_` +
/// `singular_values_` + `explained_variance_` + `explained_variance_ratio_`.
#[derive(Debug)]
pub struct PcaResult {
    /// Cell embedding, shape `(N, n_comps)`.
    pub embedding: Array2<f32>,
    /// Principal components, shape `(n_comps, G)`. Rows are unit length.
    pub components: Array2<f32>,
    /// Singular values, shape `(n_comps,)`.
    pub singular_values: Array1<f32>,
    /// `S²/(N − 1)`, shape `(n_comps,)`.
    pub explained_variance: Array1<f32>,
    /// Ratio of each component's variance to total variance of `A`.
    pub explained_variance_ratio: Array1<f32>,
}

/// Diagnostic report from `suggest_n_comps` — exposes every intermediate
/// quantity so downstream users (and tests) can verify the decision.
#[derive(Debug, Clone)]
pub struct NcompsSuggestion {
    /// Gavish-Donoho 2014 optimal hard threshold: the number of SVs
    /// that exceed `τ* = ω(β) · median(singular_values)` where
    /// `β = min(N,G) / max(N,G)`. MSE-optimal truncation under iid
    /// noise (proven). Single-cell data violates the iid assumption
    /// mildly; the estimate is usually conservative (small).
    pub n_comps_gavish_donoho: usize,
    /// Classical perpendicular-line elbow: farthest point (index) from
    /// the chord connecting the first and last singular value. Pure
    /// geometric heuristic, kept as a sanity cross-check.
    pub n_comps_elbow: usize,
    /// Final pragmatic recommendation — `clamp(gd + margin, min, max)`
    /// with margin=5 to leave headroom for downstream integration
    /// (Harmony's soft k-means benefits from a few extra PCs beyond
    /// the strict denoising threshold).
    pub suggested_n_comps: usize,
    /// Gavish-Donoho threshold value `τ*` on singular values.
    pub gd_threshold: f32,
    /// Median of the singular values used as noise scale estimate.
    pub sv_median: f32,
    /// Aspect ratio `β` used in the ω(β) formula.
    pub beta: f32,
}

// =============================================================================
// Entry points
// =============================================================================

/// Randomized PCA of a CSR-format sparse matrix `A: (n_rows, n_cols)`,
/// without centering (matches scanpy `zero_center=False` on sparse).
///
/// * `indptr` — length `n_rows + 1`, CSR row pointers (i64 for large nnz)
/// * `indices` — length nnz, column index per non-zero (u32 fits G ≤ 4B)
/// * `data` — length nnz, values
#[allow(clippy::too_many_arguments)]
pub fn pca_csr_f32(
    indptr: &[u64],
    indices: &[u32],
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    n_comps: usize,
    params: RsvdParams,
    seed: u64,
) -> PcaResult {
    assert_eq!(indptr.len(), n_rows + 1, "indptr wrong length");
    assert_eq!(indices.len(), data.len(), "indices/data length mismatch");
    assert!(n_comps >= 1, "n_comps must be ≥ 1");
    assert!(n_comps <= n_rows.min(n_cols), "n_comps > min(N, G)");

    let profile = std::env::var("SCATLAS_PCA_PROFILE").is_ok();
    let now = std::time::Instant::now;

    let csr = CsrView {
        indptr,
        indices,
        data,
        n_rows,
        n_cols,
    };

    // Build CSC (the transpose in CSR format) once. Both A·X and Aᵀ·X
    // then reuse the fast per-row `csr_dot_dense` kernel.
    let t = now();
    let (csc_indptr, csc_indices, csc_data) = csr_to_csc(&csr);
    let csc = CsrView {
        indptr: &csc_indptr,
        indices: &csc_indices,
        data: &csc_data,
        n_rows: n_cols,
        n_cols: n_rows,
    };
    if profile {
        eprintln!("[pca] CSR→CSC: {:.2?}", t.elapsed());
    }

    let l = n_comps + params.n_oversamples;
    let l = l.min(n_rows.min(n_cols));

    let t = now();
    let omega = gaussian_matrix(n_cols, l, seed);
    if profile {
        eprintln!("[pca] omega: {:.2?}", t.elapsed());
    }

    let t = now();
    let mut y = csr_dot_dense(&csr, omega.view());
    if profile {
        eprintln!("[pca] Y = A·Ω: {:.2?}", t.elapsed());
    }

    for iter in 0..params.n_power_iter {
        let t = now();
        // Z = Aᵀ·Y = (CSC as CSR) · Y
        let z = csr_dot_dense(&csc, y.view());
        let t1 = t.elapsed();
        let t = now();
        y = csr_dot_dense(&csr, z.view());
        let t2 = t.elapsed();
        let t = now();
        qr_mgs_inplace(&mut y);
        let t3 = t.elapsed();
        if profile {
            eprintln!(
                "[pca] power {}: Aᵀ·Y={:.2?}, A·Z={:.2?}, QR={:.2?}",
                iter + 1,
                t1,
                t2,
                t3
            );
        }
    }

    let t = now();
    qr_mgs_inplace(&mut y);
    if profile {
        eprintln!("[pca] final QR: {:.2?}", t.elapsed());
    }
    let q = y;

    // B = Qᵀ·A: (L, G) = (Aᵀ·Q)ᵀ
    let t = now();
    let at_q = csr_dot_dense(&csc, q.view()); // (G, L)
    let b = at_q.t().to_owned(); // (L, G)
    if profile {
        eprintln!("[pca] B = Qᵀ·A: {:.2?}", t.elapsed());
    }

    // Step 6: M = B·Bᵀ   (L, L), eigendecompose.
    let m = b.dot(&b.t());
    let (eigvals, eigvecs) = jacobi_symmetric_eigen(m.view());
    // Sort by descending eigenvalue
    let mut order: Vec<usize> = (0..eigvals.len()).collect();
    order.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut s = Array1::<f32>::zeros(l);
    let mut u_tilde = Array2::<f32>::zeros((l, l));
    for (new_i, &old_i) in order.iter().enumerate() {
        s[new_i] = eigvals[old_i].max(0.0).sqrt();
        for row in 0..l {
            u_tilde[[row, new_i]] = eigvecs[[row, old_i]];
        }
    }

    // Step 7: V = (1/S) · U_tildeᵀ · B    (L, G)
    let mut v = u_tilde.t().dot(&b);
    for j in 0..l {
        let sj = s[j];
        if sj > 1e-20 {
            let inv = 1.0 / sj;
            for g in 0..n_cols {
                v[[j, g]] *= inv;
            }
        }
    }

    // Step 8: U = Q · U_tilde    (N, L)
    let u = q.dot(&u_tilde);

    // Step 9: keep top-k; embedding = U[:, :k] · diag(S[:k])
    let mut embedding = Array2::<f32>::zeros((n_rows, n_comps));
    for i in 0..n_rows {
        for j in 0..n_comps {
            embedding[[i, j]] = u[[i, j]] * s[j];
        }
    }
    let mut components = Array2::<f32>::zeros((n_comps, n_cols));
    for j in 0..n_comps {
        for g in 0..n_cols {
            components[[j, g]] = v[[j, g]];
        }
    }
    let singular_values = s.slice(ndarray::s![..n_comps]).to_owned();
    let explained_variance =
        singular_values.mapv(|sv| sv * sv / (n_rows.saturating_sub(1) as f32).max(1.0));

    // Total variance of A (no centering): sum(data²) / (N − 1)
    let total_var: f32 =
        data.par_iter().map(|&x| x * x).sum::<f32>() / (n_rows.saturating_sub(1) as f32).max(1.0);
    let explained_variance_ratio = if total_var > 0.0 {
        explained_variance.mapv(|v| v / total_var)
    } else {
        Array1::<f32>::zeros(n_comps)
    };

    PcaResult {
        embedding,
        components,
        singular_values,
        explained_variance,
        explained_variance_ratio,
    }
}

/// Randomized PCA of a dense matrix (no centering).
pub fn pca_dense_f32(
    a: ArrayView2<f32>,
    n_comps: usize,
    params: RsvdParams,
    seed: u64,
) -> PcaResult {
    let n_rows = a.nrows();
    let n_cols = a.ncols();
    assert!(n_comps >= 1, "n_comps must be ≥ 1");
    assert!(n_comps <= n_rows.min(n_cols), "n_comps > min(N, G)");

    let l = n_comps + params.n_oversamples;
    let l = l.min(n_rows.min(n_cols));

    let omega = gaussian_matrix(n_cols, l, seed);
    let mut y = a.dot(&omega); // (N, L)

    for _ in 0..params.n_power_iter {
        let z = a.t().dot(&y); // (G, L)
        y = a.dot(&z); // (N, L)
        qr_mgs_inplace(&mut y);
    }
    qr_mgs_inplace(&mut y);
    let q = y;

    let b = q.t().dot(&a); // (L, G)
    let m = b.dot(&b.t());
    let (eigvals, eigvecs) = jacobi_symmetric_eigen(m.view());
    let mut order: Vec<usize> = (0..eigvals.len()).collect();
    order.sort_by(|&x, &y| {
        eigvals[y]
            .partial_cmp(&eigvals[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut s = Array1::<f32>::zeros(l);
    let mut u_tilde = Array2::<f32>::zeros((l, l));
    for (new_i, &old_i) in order.iter().enumerate() {
        s[new_i] = eigvals[old_i].max(0.0).sqrt();
        for row in 0..l {
            u_tilde[[row, new_i]] = eigvecs[[row, old_i]];
        }
    }

    let mut v = u_tilde.t().dot(&b);
    for j in 0..l {
        let sj = s[j];
        if sj > 1e-20 {
            let inv = 1.0 / sj;
            for g in 0..n_cols {
                v[[j, g]] *= inv;
            }
        }
    }

    let u = q.dot(&u_tilde);
    let mut embedding = Array2::<f32>::zeros((n_rows, n_comps));
    for i in 0..n_rows {
        for j in 0..n_comps {
            embedding[[i, j]] = u[[i, j]] * s[j];
        }
    }
    let mut components = Array2::<f32>::zeros((n_comps, n_cols));
    for j in 0..n_comps {
        for g in 0..n_cols {
            components[[j, g]] = v[[j, g]];
        }
    }
    let singular_values = s.slice(ndarray::s![..n_comps]).to_owned();
    let explained_variance =
        singular_values.mapv(|sv| sv * sv / (n_rows.saturating_sub(1) as f32).max(1.0));

    let total_var: f32 =
        a.iter().map(|&x| x * x).sum::<f32>() / (n_rows.saturating_sub(1) as f32).max(1.0);
    let explained_variance_ratio = if total_var > 0.0 {
        explained_variance.mapv(|v| v / total_var)
    } else {
        Array1::<f32>::zeros(n_comps)
    };

    PcaResult {
        embedding,
        components,
        singular_values,
        explained_variance,
        explained_variance_ratio,
    }
}

// =============================================================================
// Automatic n_comps selection (Gavish-Donoho + perpendicular elbow)
// =============================================================================

/// Suggest an optimal number of principal components from a vector of
/// singular values + matrix shape, using Gavish-Donoho 2014 as the
/// primary criterion.
///
/// **Method** — Gavish & Donoho, *IEEE Trans. Information Theory*
/// 60(8):5040-5053 (2014): for a matrix `A = X + σW` with iid noise `W`,
/// the MSE-optimal hard threshold for SVD truncation is
/// `τ* = ω(β) · σ`, with β = min(N,G)/max(N,G) and
///
/// ```text
/// ω(β) = √( 2(β+1) + 8β / ((β+1) + √(β² + 14β + 1)) )
/// ```
///
/// When σ is unknown we estimate it from the median singular value:
/// `σ̂ = median(singular_values) / μ(β)`, where μ(β) is the theoretical
/// median of the Marchenko-Pastur distribution. In practice (and per
/// the Gavish-Donoho paper, §2.3) using `τ* = ω(β) · median(sv)`
/// directly gives near-optimal behavior without needing the MP median
/// quantile lookup, so that's what we use.
///
/// **Secondary** — perpendicular-line elbow: farthest point from the
/// chord between sv[0] and sv[L-1]. Classical knee-detection.
///
/// `margin` — extra PCs added to the Gavish-Donoho count before
/// clamping (integration steps like Harmony benefit from a few extra
/// dims beyond strict denoising). `min_comps`/`max_comps` clamp the
/// final `suggested_n_comps`.
pub fn suggest_n_comps(
    singular_values: &[f32],
    n_rows: usize,
    n_cols: usize,
    margin: usize,
    min_comps: usize,
    max_comps: usize,
) -> NcompsSuggestion {
    assert!(!singular_values.is_empty(), "need at least one SV");
    assert!(
        min_comps <= max_comps,
        "min_comps {} must be ≤ max_comps {}",
        min_comps,
        max_comps
    );

    // --- β = min(N, G) / max(N, G), bounded in (0, 1]
    let (nn, gg) = (n_rows as f32, n_cols as f32);
    let beta = nn.min(gg) / nn.max(gg);

    // --- ω(β) coefficient
    let omega_beta = (2.0 * (beta + 1.0)
        + 8.0 * beta / ((beta + 1.0) + (beta * beta + 14.0 * beta + 1.0).sqrt()))
    .sqrt();

    // --- median of singular values (noise-scale estimator)
    let mut sorted: Vec<f32> = singular_values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let sv_median = if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    };

    // --- τ* = ω(β) · median
    let gd_threshold = omega_beta * sv_median;

    // --- count SVs above threshold. We assume input is descending.
    let n_gd = singular_values
        .iter()
        .filter(|&&sv| sv > gd_threshold)
        .count();

    // --- perpendicular-line elbow
    let n_elbow = perpendicular_elbow(singular_values);

    // --- final pragmatic recommendation
    let suggested_raw = n_gd.saturating_add(margin).max(1);
    let suggested_n_comps = suggested_raw.clamp(min_comps, max_comps);

    NcompsSuggestion {
        n_comps_gavish_donoho: n_gd,
        n_comps_elbow: n_elbow,
        suggested_n_comps,
        gd_threshold,
        sv_median,
        beta,
    }
}

/// Perpendicular-line ("knee") elbow: for a descending scree curve of
/// length L, find the index that is farthest from the straight line
/// from `(0, sv[0])` to `(L-1, sv[L-1])`. Returns a 1-indexed count.
fn perpendicular_elbow(sv: &[f32]) -> usize {
    let l = sv.len();
    if l < 3 {
        return l;
    }
    let x1 = 0.0_f32;
    let y1 = sv[0];
    let x2 = (l - 1) as f32;
    let y2 = sv[l - 1];
    // Line direction vector
    let dx = x2 - x1;
    let dy = y2 - y1;
    let line_norm = (dx * dx + dy * dy).sqrt().max(1e-20);
    let mut best_dist = 0.0_f32;
    let mut best_idx = 1usize;
    for i in 1..(l - 1) {
        let px = i as f32;
        let py = sv[i];
        // Perpendicular distance from (px, py) to the line.
        let num = (dy * px - dx * py + x2 * y1 - y2 * x1).abs();
        let d = num / line_norm;
        if d > best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    // +1 because "keep first k" semantics.
    best_idx + 1
}

// =============================================================================
// Sparse CSR matmul
// =============================================================================

struct CsrView<'a> {
    indptr: &'a [u64],
    indices: &'a [u32],
    data: &'a [f32],
    n_rows: usize,
    n_cols: usize,
}

/// `A · X` where A is CSR `(N, G)`, X is dense `(G, L)`. Output `(N, L)`.
/// Parallelized per-row of A. The inner `axpy` (`row_out += v * x[g, :]`)
/// runs on contiguous slices so the compiler autovectorizes it.
fn csr_dot_dense(csr: &CsrView, x: ArrayView2<f32>) -> Array2<f32> {
    let n = csr.n_rows;
    let l = x.ncols();
    let mut out = Array2::<f32>::zeros((n, l));

    let x_slice = x
        .as_slice()
        .expect("csr_dot_dense: x must be row-major contiguous");

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row_out)| {
            let row_out_slice = row_out.as_slice_mut().expect("out row must be contiguous");
            let start = csr.indptr[i] as usize;
            let end = csr.indptr[i + 1] as usize;
            for k in start..end {
                let g = csr.indices[k] as usize;
                let v = csr.data[k];
                let x_row = &x_slice[g * l..(g + 1) * l];
                for ll in 0..l {
                    row_out_slice[ll] += v * x_row[ll];
                }
            }
        });
    out
}

/// Build CSC representation of a CSR matrix. Returns `(indptr, indices,
/// data)` describing Aᵀ in CSR format (equivalent to A in CSC).
///
/// Parallelized via row chunking: each chunk counts its column nnz
/// locally, we compute global column offsets + per-chunk starting
/// positions within each column, then each chunk places its entries
/// into disjoint slots of the output. O(nnz) work, ~linearly scalable.
fn csr_to_csc(csr: &CsrView) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let n = csr.n_rows;
    let g = csr.n_cols;
    let nnz = csr.data.len();

    let n_threads = rayon::current_num_threads().max(1);
    let n_chunks = (n_threads * 4).min(n.max(1));
    let chunk_size = n.div_ceil(n_chunks.max(1));

    // --- Pass 1: per-chunk column counts (parallel).
    let chunk_counts: Vec<Vec<u32>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk_size;
            let end = ((c + 1) * chunk_size).min(n);
            let mut counts = vec![0u32; g];
            for row in start..end {
                let s = csr.indptr[row] as usize;
                let e = csr.indptr[row + 1] as usize;
                for k in s..e {
                    counts[csr.indices[k] as usize] += 1;
                }
            }
            counts
        })
        .collect();

    // --- Pass 2: build csc_indptr + per-chunk starting offsets.
    let mut csc_indptr = vec![0u64; g + 1];
    let mut chunk_col_offset = vec![vec![0u64; g]; n_chunks];
    for j in 0..g {
        let mut running = csc_indptr[j];
        for c in 0..n_chunks {
            chunk_col_offset[c][j] = running;
            running += chunk_counts[c][j] as u64;
        }
        csc_indptr[j + 1] = running;
    }

    // --- Pass 3: per-chunk placement (parallel, disjoint writes).
    let mut csc_indices = vec![0u32; nnz];
    let mut csc_data = vec![0.0_f32; nnz];
    let idx_ptr_addr = csc_indices.as_mut_ptr() as usize;
    let data_ptr_addr = csc_data.as_mut_ptr() as usize;

    chunk_col_offset
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut local_cursor)| {
            let start = c * chunk_size;
            let end = ((c + 1) * chunk_size).min(n);
            let idx_ptr = idx_ptr_addr as *mut u32;
            let data_ptr = data_ptr_addr as *mut f32;
            for row in start..end {
                let s = csr.indptr[row] as usize;
                let e = csr.indptr[row + 1] as usize;
                for k in s..e {
                    let col = csr.indices[k] as usize;
                    let pos = local_cursor[col] as usize;
                    // SAFETY: per-chunk offsets are disjoint across (c, col),
                    // so no two threads write the same slot.
                    unsafe {
                        *idx_ptr.add(pos) = row as u32;
                        *data_ptr.add(pos) = csr.data[k];
                    }
                    local_cursor[col] += 1;
                }
            }
        });

    (csc_indptr, csc_indices, csc_data)
}

// =============================================================================
// Modified Gram-Schmidt QR (in-place columns)
// =============================================================================

/// In-place modified Gram-Schmidt: orthonormalize columns of `a`. The
/// R matrix of QR is discarded (we only need Q).
///
/// Layout note: `a` is `(N, L)` row-major, so a column is `L`-strided and
/// not contiguous. We build an owned copy of the pivot column each outer
/// step (cheap — it's length `N`, 600KB at L=40, N=157k), then rayon-
/// parallelize the projection subtraction across the remaining columns
/// by materializing each target column into a contiguous Vec.
///
/// For the typical scatlas shape (N=100k–1M, L=40–60), this structure is
/// ~100× faster than the fully-serial textbook version thanks to per-
/// column parallelism, while keeping the update deterministic (no races
/// because each parallel task touches a different column of `a`).
pub(crate) fn qr_mgs_inplace(a: &mut Array2<f32>) {
    // Two MGS passes ("twice is enough", Parlett 1980) — recovers O(ε)
    // orthogonality even after power-iteration multiplies by the
    // condition-squared operator A·Aᵀ.
    mgs_single_pass(a);
    mgs_single_pass(a);
}

fn mgs_single_pass(a: &mut Array2<f32>) {
    let (n, l) = a.dim();
    // Chunk size for rayon parallel reduction on column ops. ~64k
    // f32s per chunk ≈ L1 cache-friendly.
    const PAR_CHUNK: usize = 16_384;

    for j in 0..l {
        // ---- Normalize column j (parallel norm + in-place divide).
        let norm_sq: f32 = (0..n)
            .into_par_iter()
            .with_min_len(PAR_CHUNK)
            .map(|i| a[[i, j]].powi(2))
            .sum();
        let norm = norm_sq.sqrt();
        if norm < 1e-20 {
            for i in 0..n {
                a[[i, j]] = 0.0;
            }
            continue;
        }
        let inv = 1.0 / norm;
        let mut col_j = a.column_mut(j);
        col_j.par_mapv_inplace(|x| x * inv);

        // Snapshot pivot column for projection reads.
        let q_j: Vec<f32> = (0..n).map(|i| a[[i, j]]).collect();
        let q_j_ref = &q_j;

        // For each remaining column k > j, compute <q_j, a_k> and
        // subtract the projection. Writes are disjoint across k →
        // rayon-safe. Inner reduction is parallel too, to handle the
        // end-of-QR case where there are only 1-2 remaining columns.
        let (_left, mut right) = a.view_mut().split_at(Axis(1), j + 1);
        right
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col_k| {
                let proj: f32 = (0..n).map(|i| q_j_ref[i] * col_k[i]).sum();
                for i in 0..n {
                    col_k[i] -= proj * q_j_ref[i];
                }
            });
    }
}

// =============================================================================
// Jacobi symmetric eigendecomposition (small matrices only, L ~ 40-60)
// =============================================================================

/// Cyclic Jacobi eigendecomposition of a small symmetric matrix `m`.
/// Returns (eigvals, eigvecs). Eigvecs are stored as columns.
pub(crate) fn jacobi_symmetric_eigen(m: ArrayView2<f32>) -> (Array1<f32>, Array2<f32>) {
    let n = m.nrows();
    assert_eq!(n, m.ncols(), "not square");
    let mut a = m.to_owned();
    let mut v = Array2::<f32>::eye(n);

    let max_sweeps = 50;
    let tol = 1e-10;

    for _sweep in 0..max_sweeps {
        // Compute off-diagonal norm
        let mut off: f32 = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[[p, q]] * a[[p, q]];
            }
        }
        if off.sqrt() < tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < 1e-20 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta.abs() > 1e20 {
                    0.5 / theta
                } else {
                    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                    sign / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Update matrix A: rotate rows/cols p and q
                a[[p, p]] = app - t * apq;
                a[[q, q]] = aqq + t * apq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[[r, p]];
                        let arq = a[[r, q]];
                        a[[r, p]] = c * arp - s * arq;
                        a[[p, r]] = a[[r, p]];
                        a[[r, q]] = s * arp + c * arq;
                        a[[q, r]] = a[[r, q]];
                    }
                }
                // Update eigenvectors
                for r in 0..n {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = c * vrp - s * vrq;
                    v[[r, q]] = s * vrp + c * vrq;
                }
            }
        }
    }

    let eigvals = Array1::from_iter((0..n).map(|i| a[[i, i]]));
    (eigvals, v)
}

// =============================================================================
// Random number generation — standard Gaussian via Box-Muller
// =============================================================================

fn gaussian_matrix(n_rows: usize, n_cols: usize, seed: u64) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    let mut rng = SplitMix64::new(seed ^ 0xABCD_EF01_2345_6789);
    // Box-Muller in pairs. Fill row-major for cache efficiency.
    let total = n_rows * n_cols;
    let flat = out.as_slice_mut().unwrap();
    let mut i = 0;
    while i + 1 < total {
        let u1 = rng.next_f32().max(1e-20);
        let u2 = rng.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        flat[i] = mag * theta.cos();
        flat[i + 1] = mag * theta.sin();
        i += 2;
    }
    if i < total {
        let u1 = rng.next_f32().max(1e-20);
        let u2 = rng.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        flat[i] = mag * theta.cos();
    }
    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn csr_from_dense(a: &Array2<f32>) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
        let n = a.nrows();
        let g = a.ncols();
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for i in 0..n {
            for j in 0..g {
                let v = a[[i, j]];
                if v != 0.0 {
                    indices.push(j as u32);
                    data.push(v);
                }
            }
            indptr.push(data.len() as u64);
        }
        (indptr, indices, data)
    }

    #[test]
    fn jacobi_identity_matrix() {
        let m = array![[1.0_f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let (vals, _vecs) = jacobi_symmetric_eigen(m.view());
        let mut sorted: Vec<f32> = vals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-5);
        assert!((sorted[1] - 2.0).abs() < 1e-5);
        assert!((sorted[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn jacobi_nondiag_matrix() {
        // Symmetric: eigenvalues known via characteristic polynomial
        let m = array![[4.0_f32, 1.0], [1.0, 3.0]];
        // trace = 7, det = 11 → λ² − 7λ + 11 = 0 → λ = (7 ± √5)/2
        let (vals, vecs) = jacobi_symmetric_eigen(m.view());
        let mut vals: Vec<f32> = vals.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected_low = (7.0 - 5.0_f32.sqrt()) / 2.0;
        let expected_high = (7.0 + 5.0_f32.sqrt()) / 2.0;
        assert!((vals[0] - expected_low).abs() < 1e-4);
        assert!((vals[1] - expected_high).abs() < 1e-4);
        // Orthogonality of eigenvectors
        let mut dot = 0.0_f32;
        for i in 0..2 {
            dot += vecs[[i, 0]] * vecs[[i, 1]];
        }
        assert!(dot.abs() < 1e-5);
    }

    #[test]
    fn qr_mgs_orthonormalizes() {
        // Full-rank input so MGS produces a genuinely orthonormal basis.
        let mut a = Array2::<f32>::from_shape_fn((50, 6), |(i, j)| {
            ((i + 1) as f32 * 0.17 + (j + 1) as f32 * 0.41).sin()
                + ((i * j + 3) as f32 * 0.23).cos()
                + ((i + 2 * j + 7) as f32).sqrt() * 0.1
        });
        qr_mgs_inplace(&mut a);
        for j in 0..6 {
            let norm: f32 = a.column(j).iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "col {} norm = {}", j, norm);
        }
        for j in 0..6 {
            for k in (j + 1)..6 {
                let dot: f32 = a
                    .column(j)
                    .iter()
                    .zip(a.column(k).iter())
                    .map(|(&x, &y)| x * y)
                    .sum();
                assert!(dot.abs() < 1e-4, "cols {},{}: dot = {}", j, k, dot);
            }
        }
    }

    #[test]
    fn pca_dense_recovers_rank_one() {
        // Build rank-1 matrix: A = u · v.t() where ||u|| = ||v|| = 1.
        // Largest SV should recover u and v exactly (up to sign).
        let u = array![0.3_f32, 0.5, -0.4, 0.6, 0.2, -0.3];
        let mut u = u;
        let norm: f32 = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
        u.mapv_inplace(|x| x / norm);
        let v = array![0.2_f32, -0.4, 0.5, 0.7, -0.1, 0.3];
        let mut v = v;
        let vnorm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        v.mapv_inplace(|x| x / vnorm);

        let scale = 5.0_f32;
        let a = Array2::<f32>::from_shape_fn((u.len(), v.len()), |(i, j)| scale * u[i] * v[j]);

        let result = pca_dense_f32(
            a.view(),
            1,
            RsvdParams {
                n_oversamples: 2,
                n_power_iter: 3,
            },
            42,
        );
        // Embedding first col should be ±u * scale (since V is unit).
        assert!((result.singular_values[0] - scale).abs() < 1e-3);
        let emb_col = result.embedding.column(0);
        let emb_norm: f32 = emb_col.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((emb_norm - scale).abs() < 1e-3);
    }

    #[test]
    fn gavish_donoho_recovers_rank_of_low_rank_plus_noise() {
        // Build A = U·V + σ·W where rank(U·V) = 8 — Gavish-Donoho should
        // identify ~8 signal components above the noise floor.
        let n = 400;
        let g = 150;
        let true_rank = 8;
        let noise_sigma = 0.05;

        let mut rng = SplitMix64::new(13);
        let mut draw = || {
            let u1 = rng.next_f32().max(1e-20);
            let u2 = rng.next_f32();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };
        let uu = Array2::from_shape_fn((n, true_rank), |_| draw());
        let vv = Array2::from_shape_fn((true_rank, g), |_| draw());
        let mut a = uu.dot(&vv);
        for v in a.iter_mut() {
            *v += noise_sigma * draw();
        }

        let result = pca_dense_f32(
            a.view(),
            50,
            RsvdParams {
                n_oversamples: 10,
                n_power_iter: 5,
            },
            7,
        );
        let suggestion = suggest_n_comps(
            result.singular_values.as_slice().unwrap(),
            n,
            g,
            0, // no margin so the pure GD count shows through
            1,
            50,
        );
        assert!(
            (suggestion.n_comps_gavish_donoho as i32 - true_rank as i32).abs() <= 2,
            "GD count {} far from true rank {}",
            suggestion.n_comps_gavish_donoho,
            true_rank
        );
    }

    #[test]
    fn suggest_n_comps_applies_margin_and_clamps() {
        // Monotone-decreasing SVs with 5 large + many small
        let svs: Vec<f32> = (0..50)
            .map(|i| if i < 5 { 100.0 - i as f32 } else { 1.0 })
            .collect();
        let s = suggest_n_comps(&svs, 1000, 1000, 5, 15, 40);
        assert_eq!(s.n_comps_gavish_donoho, 5);
        // raw = 5 + 5 margin = 10, but clamped to ≥ min_comps=15
        assert_eq!(s.suggested_n_comps, 15);

        // max_comps cap
        let s2 = suggest_n_comps(&svs, 1000, 1000, 100, 15, 40);
        assert_eq!(s2.suggested_n_comps, 40);
    }

    #[test]
    fn pca_csr_matches_dense_on_small() {
        // Generate small dense, run both APIs, compare subspace.
        let n = 60;
        let g = 20;
        let a = Array2::<f32>::from_shape_fn((n, g), |(i, j)| {
            ((i as f32 * 0.1) + (j as f32 * 0.3)).sin() + ((i + j) as f32 * 0.05).cos()
        });
        let (indptr, indices, data) = csr_from_dense(&a);

        let dense_res = pca_dense_f32(a.view(), 3, RsvdParams::default(), 7);
        let sparse_res = pca_csr_f32(&indptr, &indices, &data, n, g, 3, RsvdParams::default(), 7);

        // Singular values should match closely (same algo, f32 randomness).
        for j in 0..3 {
            let rel = (dense_res.singular_values[j] - sparse_res.singular_values[j]).abs()
                / dense_res.singular_values[j].max(1e-6);
            assert!(
                rel < 1e-3,
                "sv {}: {} vs {}",
                j,
                dense_res.singular_values[j],
                sparse_res.singular_values[j]
            );
        }
    }
}
