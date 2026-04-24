//! Rayon-parallelized f32 matmul helpers for Harmony's two hot dots.
//!
//! ndarray's built-in `.dot()` delegates to `matrixmultiply`, which is
//! a well-tuned single-threaded kernel. On a 16-core WSL box that leaves
//! ~15× of multi-thread performance on the floor. These helpers split
//! the matmul into chunks over a chosen axis, delegate each chunk to
//! `ndarray`'s single-thread kernel, and run the chunks in parallel via
//! rayon.
//!
//! Two shape patterns matter for Harmony:
//!
//! 1. **Fat output, contracted small dim** — `(K, d) · (d, N) = (K, N)`
//!    where `d` is small (PCA dims, ~10-50). Split `N` across threads;
//!    each thread computes a column slab `C[:, n0..n1]`.
//!    (used in `compute_dist_mat`)
//!
//! 2. **Small output, long contracted dim** — `(d, N) · (N, K) = (d, K)`
//!    where `N` is large (number of cells). Split `N` across threads;
//!    each thread computes a partial sum and we reduce-add the results.
//!    (used at the start of `cluster::cluster` to update Y)
#![allow(clippy::needless_range_loop)]

use ndarray::{s, Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// How many output columns per rayon task in `matmul_kdn_par`.
/// Large enough to amortize matrixmultiply's blocking overhead; small
/// enough that 16 threads can work in parallel on 157k cells.
const COL_CHUNK: usize = 2048;

/// How many contracted indices per rayon task in `matmul_dnk_par`.
/// Larger chunks help cache, smaller gives more parallelism. 8192
/// empirically tracks well for N ~ 100k to 1M.
const CONTRACT_CHUNK: usize = 8192;

/// Pattern 1: `c = a · b` where `a: (K, d)`, `b: (d, N)`, output
/// `c: (K, N)` is fat-wide. Parallelize by splitting `N` into column
/// chunks.
pub fn matmul_kdn_par(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let k_rows = a.nrows();
    let d = a.ncols();
    assert_eq!(b.nrows(), d, "inner dim mismatch");
    let n = b.ncols();

    let mut c = Array2::<f32>::zeros((k_rows, n));
    // axis_chunks_iter_mut gives us non-overlapping column slabs.
    c.axis_chunks_iter_mut(Axis(1), COL_CHUNK)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_id, mut c_chunk)| {
            let n_start = chunk_id * COL_CHUNK;
            let n_end = (n_start + c_chunk.ncols()).min(n);
            let b_slice = b.slice(s![.., n_start..n_end]); // (d, chunk)
                                                           // matrixmultiply inside .dot() is single-threaded, cache-blocked.
            let c_local = a.dot(&b_slice); // (K, chunk)
            c_chunk.assign(&c_local);
        });
    c
}

/// Pattern 2: `c = a · b` where `a: (d, N)`, `b: (N, K)`, output
/// `c: (d, K)` is small and `N` is large. Parallelize by splitting
/// the contracted dimension `N` into chunks, compute partial products
/// in parallel, reduce-sum.
pub fn matmul_dnk_par(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let d = a.nrows();
    let n = a.ncols();
    assert_eq!(b.nrows(), n, "inner dim mismatch");
    let k_cols = b.ncols();

    // Build the list of [n_start, n_end) chunks up front so rayon knows
    // the iterator length.
    let n_chunks = n.div_ceil(CONTRACT_CHUNK);

    (0..n_chunks)
        .into_par_iter()
        .map(|chunk_id| {
            let n_start = chunk_id * CONTRACT_CHUNK;
            let n_end = (n_start + CONTRACT_CHUNK).min(n);
            let a_slice = a.slice(s![.., n_start..n_end]); // (d, chunk)
            let b_slice = b.slice(s![n_start..n_end, ..]); // (chunk, K)
            a_slice.dot(&b_slice) // (d, K)
        })
        .reduce(
            || Array2::<f32>::zeros((d, k_cols)),
            |mut acc, x| {
                acc += &x;
                acc
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn matmul_kdn_par_matches_ndarray_dot() {
        let k = 17;
        let d = 9;
        let n = 1013;
        let a = Array2::<f32>::from_shape_fn((k, d), |(i, j)| (i * 3 + j + 1) as f32 * 0.1);
        let b = Array2::<f32>::from_shape_fn((d, n), |(i, j)| ((i + j) as f32).sin());
        let c_par = matmul_kdn_par(a.view(), b.view());
        let c_ref = a.dot(&b);
        for i in 0..k {
            for j in 0..n {
                let delta = (c_par[[i, j]] - c_ref[[i, j]]).abs();
                assert!(delta < 1e-4, "delta {} at ({}, {})", delta, i, j);
            }
        }
    }

    #[test]
    fn matmul_dnk_par_matches_ndarray_dot() {
        let d = 12;
        let n = 20_000;
        let k = 31;
        let a = Array2::<f32>::from_shape_fn((d, n), |(i, j)| ((i + j) as f32 * 0.01).cos());
        let b = Array2::<f32>::from_shape_fn((n, k), |(i, j)| ((i * 2 + j) as f32 * 0.001).sin());
        let c_par = matmul_dnk_par(a.view(), b.view());
        let c_ref = a.dot(&b);
        // Reduction in f32 introduces ~1e-3 ULP error over 20k terms.
        for i in 0..d {
            for j in 0..k {
                let delta = (c_par[[i, j]] - c_ref[[i, j]]).abs();
                let rel = delta / c_ref[[i, j]].abs().max(1e-6);
                assert!(
                    delta < 1e-2 || rel < 1e-4,
                    "delta {} at ({}, {}) (ref={})",
                    delta,
                    i,
                    j,
                    c_ref[[i, j]]
                );
            }
        }
    }
}
