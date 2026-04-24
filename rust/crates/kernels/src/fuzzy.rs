//! Fuzzy simplicial set construction — port of umap-learn's
//! `fuzzy_simplicial_set` in `umap/umap_.py`. Takes a kNN graph
//! `(indices, distances)` and produces a symmetric sparse CSR of
//! membership-strength edge weights.
//!
//! **Algorithm** (`fuzzy_simplicial_set` in umap-learn):
//!
//! 1. Per cell `i`, compute:
//!    - `rho_i` = distance to nearest non-self neighbor
//!    - `sigma_i` = bisection-search bandwidth s.t. `Σ_j exp(-(d_ij - rho_i)/sigma_i) = log2(k)`
//! 2. Per edge `(i, j)`: weight `w_ij = exp(-(d_ij - rho_i)/sigma_i)` if
//!    `d_ij > rho_i` else 1.
//! 3. Build sparse `P: (N, N)` from these weights.
//! 4. Symmetrize via probabilistic union:
//!    `P_sym = set_op_mix · (P + Pᵀ − P ∘ Pᵀ) + (1 − set_op_mix) · (P ∘ Pᵀ)`
//!    Default `set_op_mix_ratio = 1.0` — just `P + Pᵀ − P ∘ Pᵀ`.
//!
//! All four steps are rayon-parallelized per-cell. Construction of CSR
//! is linear-scan (per-row edges are pre-sorted by column order).
#![allow(clippy::needless_range_loop)]

use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Hyperparameters for fuzzy_simplicial_set.
#[derive(Debug, Clone, Copy)]
pub struct FuzzyParams {
    /// Effective number of neighbors target (uses `log2(k)` in umap).
    pub k: usize,
    /// Bisection iterations for bandwidth. umap default = 64.
    pub n_iter: usize,
    /// Interpolation between union (1.0) and intersection (0.0) for the
    /// symmetrization step. umap default = 1.0.
    pub set_op_mix_ratio: f32,
    /// Minimum rho when all distances are zero (degenerate rows).
    pub local_connectivity: f32,
}

impl Default for FuzzyParams {
    fn default() -> Self {
        Self {
            k: 15,
            n_iter: 64,
            set_op_mix_ratio: 1.0,
            local_connectivity: 1.0,
        }
    }
}

/// Output format — CSR triplets with u64 indptr for large-nnz safety.
#[derive(Debug)]
pub struct FuzzyCsr {
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub data: Vec<f32>,
    pub n_cells: usize,
}

/// Build symmetric fuzzy simplicial set from a `(N, k)` kNN graph.
///
/// `indices` may contain `u32::MAX` sentinels (BBKNN uses these when a
/// batch has fewer than `k` cells). Those slots are skipped.
pub fn fuzzy_simplicial_set(
    knn_indices: ArrayView2<u32>,
    knn_dists: ArrayView2<f32>,
    params: &FuzzyParams,
) -> FuzzyCsr {
    let (n, _k_cols) = knn_indices.dim();
    assert_eq!(knn_dists.dim(), (n, _k_cols));

    // Step 1: smooth_knn_dist → per-row rho, sigma
    let (rhos, sigmas) = smooth_knn_dist(knn_dists, params);

    // Step 2: compute membership strengths per edge → COO-per-row
    let per_row = compute_membership_strengths(knn_indices, knn_dists, &rhos, &sigmas);

    // Step 3: build CSR from per-row edges (already ordered by row, and
    // within each row ordered by original k-index — we sort by column
    // index to make transpose and union easier)
    let csr = csr_from_row_edges(per_row, n);

    // Step 4: symmetrize via probabilistic union.
    //   P_sym[i,j] = mix * (P[i,j] + P[j,i] - P[i,j]·P[j,i])
    //              + (1-mix) * P[i,j]·P[j,i]
    // Default mix=1 → just union.
    symmetrize(&csr, params.set_op_mix_ratio)
}

// =============================================================================
// Step 1: smooth_knn_dist — per-row bisection for (rho, sigma)
// =============================================================================

/// Return `(rhos, sigmas)` for every row of `knn_dists`. Parallel per-row.
fn smooth_knn_dist(
    knn_dists: ArrayView2<f32>,
    params: &FuzzyParams,
) -> (Vec<f32>, Vec<f32>) {
    let n = knn_dists.nrows();
    let target = (params.k as f32).log2();

    let mut out: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    out.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let row = knn_dists.row(i);
        *slot = smooth_knn_dist_row(row, target, params.n_iter, params.local_connectivity);
    });

    let mut rhos = Vec::with_capacity(n);
    let mut sigmas = Vec::with_capacity(n);
    for (r, s) in out {
        rhos.push(r);
        sigmas.push(s);
    }
    (rhos, sigmas)
}

fn smooth_knn_dist_row(
    row: ArrayView1<f32>,
    target: f32,
    n_iter: usize,
    local_connectivity: f32,
) -> (f32, f32) {
    // rho = distance to `floor(local_connectivity)`-th nearest non-zero.
    // Here we use the classical simplification: rho = min positive distance.
    let mut rho = f32::INFINITY;
    let mut any_nonzero = false;
    for &d in row.iter() {
        if d.is_finite() && d > 0.0 {
            any_nonzero = true;
            if d < rho {
                rho = d;
            }
        }
    }
    if !any_nonzero {
        rho = 0.0;
    }
    let _ = local_connectivity; // reserved for full umap parity; simplified form

    // Bisection on sigma such that Σ max(0, exp(-(d - rho)/sigma)) ≈ target.
    let mut lo = 0.0_f32;
    let mut hi = f32::INFINITY;
    let mut mid = 1.0_f32;
    for _ in 0..n_iter {
        let mut psum = 0.0_f32;
        for &d in row.iter() {
            if !d.is_finite() {
                continue;
            }
            let delta = d - rho;
            if delta > 0.0 {
                psum += (-delta / mid).exp();
            } else {
                psum += 1.0;
            }
        }
        if (psum - target).abs() < 1e-5 {
            break;
        }
        if psum > target {
            hi = mid;
            mid = 0.5 * (lo + hi);
        } else {
            lo = mid;
            if hi == f32::INFINITY {
                mid *= 2.0;
            } else {
                mid = 0.5 * (lo + hi);
            }
        }
    }
    (rho, mid)
}

// =============================================================================
// Step 2: compute_membership_strengths
// =============================================================================

/// Per-row COO triplets `(col, weight)`, sorted by col.
struct RowEdges {
    cols: Vec<u32>,
    weights: Vec<f32>,
}

fn compute_membership_strengths(
    knn_indices: ArrayView2<u32>,
    knn_dists: ArrayView2<f32>,
    rhos: &[f32],
    sigmas: &[f32],
) -> Vec<RowEdges> {
    let n = knn_indices.nrows();
    let k = knn_indices.ncols();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let rho = rhos[i];
            let sigma = sigmas[i];
            let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(k);
            for c in 0..k {
                let j = knn_indices[[i, c]];
                // Skip sentinel / self
                if j == u32::MAX || j as usize == i {
                    continue;
                }
                let d = knn_dists[[i, c]];
                if !d.is_finite() {
                    continue;
                }
                let w = if d <= rho || sigma <= 0.0 {
                    1.0_f32
                } else {
                    (-(d - rho) / sigma).exp()
                };
                if w > 1e-8 {
                    pairs.push((j, w));
                }
            }
            // Sort by column index so CSR is well-formed and transpose
            // operations are simple.
            pairs.sort_by_key(|&(j, _)| j);
            // Deduplicate (keep max weight if same cell appears twice)
            let mut cols = Vec::with_capacity(pairs.len());
            let mut ws = Vec::with_capacity(pairs.len());
            let mut last_col: Option<u32> = None;
            for (j, w) in pairs {
                if Some(j) == last_col {
                    let last = ws.last_mut().unwrap();
                    if w > *last {
                        *last = w;
                    }
                } else {
                    cols.push(j);
                    ws.push(w);
                    last_col = Some(j);
                }
            }
            RowEdges { cols, weights: ws }
        })
        .collect()
}

// =============================================================================
// Step 3: CSR construction from per-row edges
// =============================================================================

fn csr_from_row_edges(rows: Vec<RowEdges>, n_cells: usize) -> FuzzyCsr {
    let nnz: usize = rows.iter().map(|r| r.cols.len()).sum();
    let mut indptr = Vec::with_capacity(n_cells + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0u64);
    for r in rows {
        indices.extend_from_slice(&r.cols);
        data.extend_from_slice(&r.weights);
        indptr.push(indices.len() as u64);
    }
    FuzzyCsr { indptr, indices, data, n_cells }
}

// =============================================================================
// Step 4: symmetrize via `P + Pᵀ − P ∘ Pᵀ`
// =============================================================================

fn symmetrize(p: &FuzzyCsr, mix_ratio: f32) -> FuzzyCsr {
    let n = p.n_cells;
    // Build CSC view of P (i.e., Pᵀ in CSR form).
    let (csc_indptr, csc_indices, csc_data) = csr_to_csc(p);

    // For each row i, we want the union of column sets from P.row(i) and
    // Pᵀ.row(i) = P.col(i). Merge sorted lists (both pre-sorted by col).
    let rows: Vec<RowEdges> = (0..n)
        .into_par_iter()
        .map(|i| merge_row_with_col(p, &csc_indptr, &csc_indices, &csc_data, i, mix_ratio))
        .collect();

    csr_from_row_edges(rows, n)
}

/// For one cell i: merge P's row-i (from CSR) and P's col-i (from CSC).
/// Compute `union = mix * (p + p.t - p * p.t) + (1 - mix) * (p * p.t)`.
fn merge_row_with_col(
    p: &FuzzyCsr,
    csc_indptr: &[u64],
    csc_indices: &[u32],
    csc_data: &[f32],
    i: usize,
    mix: f32,
) -> RowEdges {
    let r_s = p.indptr[i] as usize;
    let r_e = p.indptr[i + 1] as usize;
    let c_s = csc_indptr[i] as usize;
    let c_e = csc_indptr[i + 1] as usize;

    let r_cols = &p.indices[r_s..r_e];
    let r_vals = &p.data[r_s..r_e];
    let c_cols = &csc_indices[c_s..c_e];
    let c_vals = &csc_data[c_s..c_e];

    let mut out_cols = Vec::with_capacity(r_cols.len() + c_cols.len());
    let mut out_vals = Vec::with_capacity(r_cols.len() + c_cols.len());

    let (mut ri, mut ci) = (0, 0);
    let one_minus_mix = 1.0 - mix;
    while ri < r_cols.len() || ci < c_cols.len() {
        let r_col = if ri < r_cols.len() { r_cols[ri] } else { u32::MAX };
        let c_col = if ci < c_cols.len() { c_cols[ci] } else { u32::MAX };
        let (j, p_val, pt_val) = match r_col.cmp(&c_col) {
            std::cmp::Ordering::Equal => {
                let v = (r_vals[ri], c_vals[ci]);
                ri += 1;
                ci += 1;
                (r_col, v.0, v.1)
            }
            std::cmp::Ordering::Less => {
                let v = r_vals[ri];
                ri += 1;
                (r_col, v, 0.0)
            }
            std::cmp::Ordering::Greater => {
                let v = c_vals[ci];
                ci += 1;
                (c_col, 0.0, v)
            }
        };
        let prod = p_val * pt_val;
        let union = p_val + pt_val - prod;
        let w = mix * union + one_minus_mix * prod;
        if w > 1e-8 {
            out_cols.push(j);
            out_vals.push(w);
        }
    }
    RowEdges { cols: out_cols, weights: out_vals }
}

/// CSR → CSC conversion (parallel chunked). Same algo as `pca::csr_to_csc`
/// but specialized for our `FuzzyCsr` layout.
fn csr_to_csc(csr: &FuzzyCsr) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let n = csr.n_cells;
    let g = n; // square matrix
    let nnz = csr.data.len();

    let n_threads = rayon::current_num_threads().max(1);
    let n_chunks = (n_threads * 4).min(n.max(1));
    let chunk_size = n.div_ceil(n_chunks.max(1));

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
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn fuzzy_simple_symmetric() {
        // 4-cell ring: 0-1-2-3-0. k=2, symmetric structure → union should
        // preserve the ring.
        let idx: Array2<u32> = array![[1u32, 3], [0, 2], [1, 3], [2, 0]];
        let dists: Array2<f32> = array![
            [1.0_f32, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ];
        let params = FuzzyParams {
            k: 2,
            n_iter: 64,
            set_op_mix_ratio: 1.0,
            local_connectivity: 1.0,
        };
        let result = fuzzy_simplicial_set(idx.view(), dists.view(), &params);
        assert_eq!(result.n_cells, 4);
        // Should have 8 edges (ring ×2 directions)
        assert_eq!(result.data.len(), 8);
        // Each cell has degree 2
        for i in 0..4 {
            let s = result.indptr[i];
            let e = result.indptr[i + 1];
            assert_eq!(e - s, 2, "cell {} should have degree 2", i);
        }
        // Ring weights should all be > 0
        for &w in &result.data {
            assert!(w > 0.0 && w <= 1.0, "weight out of range: {}", w);
        }
    }

    #[test]
    fn fuzzy_handles_sentinels() {
        // Cell 0 has only 1 valid neighbor (the other is sentinel)
        let idx: Array2<u32> = array![[1u32, u32::MAX], [0, 2], [1, 0]];
        let dists: Array2<f32> = array![[0.5_f32, 0.0], [0.5, 0.3], [0.3, 0.7]];
        let params = FuzzyParams {
            k: 2,
            n_iter: 32,
            set_op_mix_ratio: 1.0,
            local_connectivity: 1.0,
        };
        let result = fuzzy_simplicial_set(idx.view(), dists.view(), &params);
        // No NaN / Inf
        for &w in &result.data {
            assert!(w.is_finite(), "non-finite weight: {}", w);
        }
    }

    #[test]
    fn fuzzy_symmetry_after_union() {
        // Asymmetric input: cell 0 has 1 as neighbor with weight ~1, but
        // cell 1 doesn't have 0 as neighbor. Union should still make
        // the (0, 1) edge appear in both rows.
        let idx: Array2<u32> = array![[1u32, 2], [2, 3], [0, 3], [1, 2]];
        let dists: Array2<f32> = array![
            [0.1_f32, 0.2],
            [0.1, 0.2],
            [0.1, 0.2],
            [0.1, 0.2]
        ];
        let params = FuzzyParams::default();
        let result = fuzzy_simplicial_set(idx.view(), dists.view(), &params);
        // Build the full (4, 4) matrix and check symmetry
        let n = 4;
        let mut mat = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            let s = result.indptr[i] as usize;
            let e = result.indptr[i + 1] as usize;
            for k in s..e {
                let j = result.indices[k] as usize;
                mat[i][j] = result.data[k];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (mat[i][j] - mat[j][i]).abs();
                assert!(
                    diff < 1e-5,
                    "not symmetric at ({}, {}): {} vs {}",
                    i, j, mat[i][j], mat[j][i]
                );
            }
        }
    }
}
