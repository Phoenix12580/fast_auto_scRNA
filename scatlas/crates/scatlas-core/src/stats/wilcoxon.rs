//! Wilcoxon rank-sum (Mann-Whitney U) p-values for every row of a matrix.
//!
//! Ported from `scvalidate_rust/src/wilcoxon.rs` (MIT). The PyO3 layer has
//! been lifted to `scatlas-py`; this module is pure-Rust, so the GIL-release
//! wrapper (`Python::allow_threads`) is the caller's responsibility.
//!
//! Semantics match `scvalidate.recall_py.core._wilcoxon_per_gene`:
//!   - Per row, compute tie-averaged ranks across the union of two groups
//!     (`mask1 | mask2`), using `scipy.stats.rankdata(method="average")`
//!     semantics.
//!   - `U = R1 - n1 * (n1 + 1) / 2`, normal-approx two-sided p-value
//!     `p = 2 * (1 - Φ(|z|))` with **no ties correction** (matches the Python
//!     comment "ignore ties correction — good enough for FDR calibration").
//!   - Clip `p ∈ [1e-300, 1.0]`.
//!
//! Parallelizes over matrix rows with rayon.

use std::fmt;

use ndarray::{s, ArrayView1, ArrayView2};
use rayon::prelude::*;

const P_FLOOR: f64 = 1e-300;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WilcoxonError {
    pub n_genes: usize,
    pub n_cells: usize,
    pub mask1_len: usize,
    pub mask2_len: usize,
}

impl fmt::Display for WilcoxonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mask shapes must match number of cells: log_counts is ({}, {}), \
             mask1 has {}, mask2 has {}",
            self.n_genes, self.n_cells, self.mask1_len, self.mask2_len
        )
    }
}

impl std::error::Error for WilcoxonError {}

#[inline]
fn two_sided_p(abs_z: f64) -> f64 {
    libm::erfc(abs_z / SQRT_2).clamp(P_FLOOR, 1.0)
}

/// Tie-averaged ranks of the selected cells of one row. Generic over `T:
/// PartialOrd + Copy` so f32 and f64 inputs share one implementation.
///
/// Returns `(n_group1, n_group2, rank_sum_group1)`.
#[inline]
fn row_rank_sum_group1<T: PartialOrd + Copy>(
    values: &[T],
    group1_mask: &[bool],
) -> (usize, usize, f64) {
    let n = values.len();
    debug_assert_eq!(group1_mask.len(), n);

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        values[a as usize]
            .partial_cmp(&values[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks: Vec<f64> = vec![0.0; n];
    let mut i = 0usize;
    while i < n {
        let vi = values[order[i] as usize];
        let mut j = i + 1;
        while j < n && values[order[j] as usize] == vi {
            j += 1;
        }
        let avg = (i + j + 1) as f64 * 0.5;
        for k in i..j {
            ranks[order[k] as usize] = avg;
        }
        i = j;
    }

    let mut n1: usize = 0;
    let mut r1: f64 = 0.0;
    for k in 0..n {
        if group1_mask[k] {
            n1 += 1;
            r1 += ranks[k];
        }
    }
    (n1, n - n1, r1)
}

/// Per-row Wilcoxon rank-sum p-values. Rows = genes, columns = cells.
///
/// `mask1 | mask2` picks the cells that participate in ranking; cells outside
/// both masks are ignored. Returns `vec![1.0; n_genes]` when either group is
/// empty.
pub fn wilcoxon_ranksum_matrix<T>(
    x: ArrayView2<T>,
    m1: ArrayView1<bool>,
    m2: ArrayView1<bool>,
) -> Result<Vec<f64>, WilcoxonError>
where
    T: PartialOrd + Copy + Sync + Send,
{
    let (n_genes, n_cells) = (x.nrows(), x.ncols());
    if m1.len() != n_cells || m2.len() != n_cells {
        return Err(WilcoxonError {
            n_genes,
            n_cells,
            mask1_len: m1.len(),
            mask2_len: m2.len(),
        });
    }

    let mut keep_idx: Vec<usize> = Vec::with_capacity(n_cells);
    let mut is_group1: Vec<bool> = Vec::with_capacity(n_cells);
    for k in 0..n_cells {
        let in1 = unsafe { *m1.uget(k) };
        let in2 = unsafe { *m2.uget(k) };
        if in1 || in2 {
            keep_idx.push(k);
            is_group1.push(in1);
        }
    }

    let n_total = keep_idx.len();
    let n1_global: usize = is_group1.iter().filter(|&&b| b).count();
    let n2_global: usize = n_total - n1_global;

    if n1_global == 0 || n2_global == 0 {
        return Ok(vec![1.0_f64; n_genes]);
    }

    let mu_u = (n1_global as f64) * (n2_global as f64) * 0.5;
    let sigma_u = ((n1_global as f64) * (n2_global as f64) * (n_total as f64 + 1.0) / 12.0).sqrt();

    let mut out = vec![0.0_f64; n_genes];
    out.par_iter_mut().enumerate().for_each(|(g, p)| {
        let row = x.slice(s![g, ..]);
        let mut vals: Vec<T> = Vec::with_capacity(n_total);
        for &k in keep_idx.iter() {
            vals.push(unsafe { *row.uget(k) });
        }
        let (n1, _n2, r1) = row_rank_sum_group1(&vals, &is_group1);
        debug_assert_eq!(n1, n1_global);
        let u1 = r1 - (n1 as f64) * (n1 as f64 + 1.0) * 0.5;
        let z = (u1 - mu_u) / sigma_u;
        *p = two_sided_p(z.abs());
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn rank_no_ties() {
        let vals = [10.0_f64, 30.0, 20.0];
        let mask = [true, true, true];
        let (n1, n2, r1) = row_rank_sum_group1(&vals, &mask);
        assert_eq!(n1, 3);
        assert_eq!(n2, 0);
        assert_eq!(r1, 6.0);
    }

    #[test]
    fn rank_with_ties() {
        let vals = [10.0_f64, 10.0, 20.0];
        let mask = [true, false, true];
        let (n1, n2, r1) = row_rank_sum_group1(&vals, &mask);
        assert_eq!(n1, 2);
        assert_eq!(n2, 1);
        assert!((r1 - 4.5).abs() < 1e-12, "r1 = {}", r1);
    }

    #[test]
    fn two_sided_p_known() {
        let p = two_sided_p(1.959964);
        assert!((p - 0.05).abs() < 1e-4, "p = {}", p);
        let p0 = two_sided_p(0.0);
        assert!((p0 - 1.0).abs() < 1e-12, "p0 = {}", p0);
    }

    #[test]
    fn matrix_empty_group_returns_ones() {
        let x: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let m1 = array![true, true, true];
        let m2 = array![false, false, false];
        let pvals = wilcoxon_ranksum_matrix(x.view(), m1.view(), m2.view()).unwrap();
        assert_eq!(pvals, vec![1.0, 1.0]);
    }

    #[test]
    fn matrix_shape_mismatch_errors() {
        let x: Array2<f64> = array![[1.0, 2.0, 3.0]];
        let m1 = array![true, false];
        let m2 = array![false, true];
        let err = wilcoxon_ranksum_matrix(x.view(), m1.view(), m2.view()).unwrap_err();
        assert_eq!(err.n_cells, 3);
        assert_eq!(err.mask1_len, 2);
    }

    #[test]
    fn matrix_separation_small() {
        // Two genes, 6 cells, 3 per group. Gene 0 is perfectly separating;
        // gene 1 is constant. Expect low p for gene 0, p=1 for gene 1.
        let x: Array2<f64> = array![[1.0, 2.0, 3.0, 10.0, 11.0, 12.0], [1.0; 6]];
        let m1 = array![true, true, true, false, false, false];
        let m2 = array![false, false, false, true, true, true];
        let pvals = wilcoxon_ranksum_matrix(x.view(), m1.view(), m2.view()).unwrap();
        assert!(pvals[0] < 0.1, "gene 0 should separate, got p={}", pvals[0]);
        assert!(
            (pvals[1] - 1.0).abs() < 1e-10,
            "constant gene should give p=1, got {}",
            pvals[1]
        );
    }
}
