//! kBET — k-nearest-neighbor Batch Effect Test (Büttner 2019).
//!
//! For each cell, perform a χ² goodness-of-fit test between the batch
//! composition of its k neighbors and the global batch distribution:
//!
//! ```text
//! χ²_i = Σ_b (observed_ib − expected_b)² / expected_b
//! expected_b = k_eff · (n_global_b / N)
//! ```
//!
//! With `df = n_batches − 1`. The p-value (via `scipy.stats.chi2.sf` in
//! the Python wrapper) gauges per-cell batch mixing; dataset-level kBET
//! acceptance rate = fraction of cells with p ≥ α (default 0.05).
//!
//! Padded neighbor slots (label `i32::MIN`) are skipped and `k_eff` is
//! reduced accordingly — important for BBKNN output when a batch is
//! smaller than `k_per_batch`.

use ndarray::ArrayView2;
use rayon::prelude::*;

/// Per-cell χ² statistic vs the global batch distribution.
///
/// * `knn_labels` shape (n_cells, k) — `i32` batch codes, `i32::MIN`
///   marks padding.
/// * `global_counts` length n_batches — total count of each batch in the
///   whole dataset (indexed by the encoded batch code).
///
/// Returns a length-n_cells vector of χ² values. `chi²_i = 0` when the
/// cell has no valid neighbors (all padded).
pub fn kbet_chi2_per_cell(knn_labels: ArrayView2<i32>, global_counts: &[u64]) -> Vec<f64> {
    let n_cells = knn_labels.nrows();
    let n_batches = global_counts.len();
    let n_total: u64 = global_counts.iter().sum();
    let n_total_f = n_total as f64;

    let mut out = vec![0.0_f64; n_cells];
    if n_batches == 0 || n_total == 0 {
        return out;
    }

    out.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let row = knn_labels.row(i);
        let mut observed = vec![0.0_f64; n_batches];
        let mut k_eff = 0usize;
        for j in 0..row.len() {
            let b = row[j];
            if b == i32::MIN {
                continue;
            }
            let bu = b as usize;
            if bu < n_batches {
                observed[bu] += 1.0;
                k_eff += 1;
            }
        }
        if k_eff == 0 {
            return;
        }
        let k = k_eff as f64;
        let mut chi2 = 0.0_f64;
        for b in 0..n_batches {
            let expected = k * (global_counts[b] as f64) / n_total_f;
            if expected > 0.0 {
                let diff = observed[b] - expected;
                chi2 += diff * diff / expected;
            }
        }
        *slot = chi2;
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn perfectly_balanced_cell_gives_zero_chi2() {
        // Two equally-sized batches in the dataset; a cell with 2/2 neighbors
        // exactly matches the expected 50/50 → χ² = 0.
        let labels: Array2<i32> = array![[0, 0, 1, 1]];
        let counts = [50u64, 50];
        let out = kbet_chi2_per_cell(labels.view(), &counts);
        assert!(out[0].abs() < 1e-10, "χ² = {}", out[0]);
    }

    #[test]
    fn completely_imbalanced_cell_gives_large_chi2() {
        // Dataset is 50/50 but cell's 4 neighbors are all batch 0.
        // observed = [4, 0]; expected = [2, 2]; χ² = 4/2 + 4/2 = 4.
        let labels: Array2<i32> = array![[0, 0, 0, 0]];
        let counts = [50u64, 50];
        let out = kbet_chi2_per_cell(labels.view(), &counts);
        assert!((out[0] - 4.0).abs() < 1e-10, "χ² = {}", out[0]);
    }

    #[test]
    fn padding_reduces_effective_k() {
        // Only 2 valid neighbors (both batch 0); 50/50 dataset.
        // observed = [2, 0]; expected = [1, 1]; χ² = 1 + 1 = 2.
        let labels: Array2<i32> = array![[0, 0, i32::MIN, i32::MIN]];
        let counts = [50u64, 50];
        let out = kbet_chi2_per_cell(labels.view(), &counts);
        assert!((out[0] - 2.0).abs() < 1e-10, "χ² = {}", out[0]);
    }

    #[test]
    fn three_batches_with_skew() {
        // Dataset proportions: 1/2, 1/4, 1/4.
        // Cell has 8 neighbors: 4 from batch 0, 2 from batch 1, 2 from batch 2.
        // expected = 8 * [.5, .25, .25] = [4, 2, 2]. observed = [4, 2, 2]. χ² = 0.
        let labels: Array2<i32> = array![[0, 0, 0, 0, 1, 1, 2, 2]];
        let counts = [4u64, 2, 2];
        let out = kbet_chi2_per_cell(labels.view(), &counts);
        assert!(out[0].abs() < 1e-10, "χ² = {}", out[0]);
    }

    #[test]
    fn all_padding_returns_zero() {
        let labels: Array2<i32> = array![[i32::MIN, i32::MIN]];
        let counts = [10u64, 10];
        let out = kbet_chi2_per_cell(labels.view(), &counts);
        assert!(out[0].abs() < 1e-10);
    }

    #[test]
    fn many_cells_parallel_monotone() {
        // Dataset is 60/40 split; N cells with varying neighborhoods —
        // just check shape/finiteness.
        let n = 1000;
        let k = 6;
        let mut data = Vec::with_capacity(n * k);
        for i in 0..n {
            for j in 0..k {
                data.push(((i + j) % 2) as i32);
            }
        }
        let labels = Array2::from_shape_vec((n, k), data).unwrap();
        let counts = [600u64, 400];
        let out = kbet_chi2_per_cell(labels.view(), &counts);
        assert_eq!(out.len(), n);
        for v in &out {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }
}
