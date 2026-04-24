//! Silhouette score on a precomputed distance matrix.
//!
//! Matches `sklearn.metrics.silhouette_score(X, labels, metric="precomputed")`
//! semantics exactly:
//!
//! For each sample `i`:
//!   * `a(i) = mean(d(i, j))` over `j` in i's cluster, `j != i`
//!   * `b(i) = min_c' mean(d(i, j))` over `j` in cluster `c'`, for `c' != c_i`
//!   * `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
//!   * If i's cluster has only one member, `s(i) = 0`
//!
//! Overall silhouette = mean over all `s(i)`.
//!
//! Bridges the GS-3 milestone: the Python fallback in
//! `cluster/resolution.py::_silhouette_impl` dispatches here via
//! `fast_auto_scrna._native.silhouette.silhouette_precomputed` when the
//! compiled kernel is available, dropping the 890 s sklearn sweep on 222 k
//! atlas smokes to ~20 s.

use std::collections::HashMap;

use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Silhouette score on a precomputed distance matrix.
///
/// - `distances`: square `(n, n)` f32; symmetric, non-negative, diagonal
///   conventionally 0 (diagonal is skipped anyway)
/// - `labels`: `(n,)` i32 cluster ids (need not be contiguous — this
///   function remaps internally)
///
/// Returns 0.0 when fewer than 2 distinct clusters are present (matches
/// the Python fallback's convention — sklearn raises instead).
///
/// # Panics
/// Panics if `labels.len() != distances.shape()[0]` or the matrix isn't
/// square. These are invariants the PyO3 wrapper checks on the Python side.
pub fn silhouette_precomputed(
    distances: ArrayView2<'_, f32>,
    labels: ArrayView1<'_, i32>,
) -> f32 {
    let n = labels.len();
    assert_eq!(
        distances.shape(),
        [n, n],
        "distance matrix shape {:?} != (n={}, n)", distances.shape(), n,
    );

    // Remap labels to contiguous 0..k-1 indices. Small-k assumption — a
    // HashMap is fine; the typical use case is k = 3..30.
    let mut remap: HashMap<i32, usize> = HashMap::new();
    let mut labels_idx: Vec<usize> = Vec::with_capacity(n);
    for &l in labels.iter() {
        let len_before = remap.len();
        let idx = *remap.entry(l).or_insert(len_before);
        labels_idx.push(idx);
    }
    let k = remap.len();
    if k < 2 {
        return 0.0;
    }

    // Per-cluster sizes
    let mut sizes = vec![0usize; k];
    for &c in &labels_idx {
        sizes[c] += 1;
    }

    // Per-point silhouette; embarrassingly parallel over i.
    let scores: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let ci = labels_idx[i];
            let size_ci = sizes[ci];
            if size_ci < 2 {
                // Singleton cluster — sklearn convention: s = 0
                return 0.0;
            }

            let row = distances.row(i);

            // Sum distances into each cluster bucket (excluding self for ci)
            let mut cluster_sums = vec![0.0f32; k];
            for j in 0..n {
                if j == i {
                    continue;
                }
                cluster_sums[labels_idx[j]] += row[j];
            }

            // a(i): mean distance within own cluster (excluding self)
            let a = cluster_sums[ci] / ((size_ci - 1) as f32);

            // b(i): min over other clusters of mean distance to that cluster
            let mut b = f32::INFINITY;
            for c in 0..k {
                if c == ci {
                    continue;
                }
                let size_c = sizes[c];
                if size_c == 0 {
                    continue;
                }
                let mean = cluster_sums[c] / (size_c as f32);
                if mean < b {
                    b = mean;
                }
            }

            // Edge case: identical points → both a and b could be 0
            let denom = a.max(b);
            if denom <= 0.0 {
                return 0.0;
            }
            (b - a) / denom
        })
        .collect();

    let sum: f32 = scores.iter().sum();
    sum / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    /// Two perfectly separated clusters → silhouette ≈ 1.0
    #[test]
    fn perfectly_separated() {
        // 4 points: pair 0/1 close, pair 2/3 close, pairs far
        let d = arr2(&[
            [0.0, 0.1, 10.0, 10.1],
            [0.1, 0.0, 10.1, 10.0],
            [10.0, 10.1, 0.0, 0.1],
            [10.1, 10.0, 0.1, 0.0],
        ]);
        let l = arr1(&[0, 0, 1, 1]);
        let s = silhouette_precomputed(d.view(), l.view());
        assert!(s > 0.99, "expected ~1.0 silhouette for separated clusters, got {}", s);
    }

    /// Single cluster → 0 (by our convention; sklearn would raise)
    #[test]
    fn single_cluster_returns_zero() {
        let d = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
        let l = arr1(&[5, 5]);
        let s = silhouette_precomputed(d.view(), l.view());
        assert_eq!(s, 0.0);
    }

    /// Singleton cluster member contributes s=0
    #[test]
    fn singleton_cluster_member_contributes_zero() {
        // 3 points: two in cluster 0 (close), one in cluster 1 (alone)
        // For i=2 (singleton), s=0. For i=0,1: a ≈ 0.1, b = distance to i=2, ratio ≈ 1.
        let d = arr2(&[
            [0.0, 0.1, 5.0],
            [0.1, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ]);
        let l = arr1(&[0, 0, 1]);
        let s = silhouette_precomputed(d.view(), l.view());
        // i=0: a = 0.1, b = 5.0, s = (5.0 - 0.1) / 5.0 = 0.98
        // i=1: same, s = 0.98
        // i=2: singleton → 0
        // mean = (0.98 + 0.98 + 0) / 3 ≈ 0.653
        assert!((s - 0.653).abs() < 0.01, "expected ~0.653, got {}", s);
    }

    /// Non-contiguous label ids should still work (internal remap)
    #[test]
    fn non_contiguous_labels() {
        let d = arr2(&[
            [0.0, 0.1, 10.0, 10.1],
            [0.1, 0.0, 10.1, 10.0],
            [10.0, 10.1, 0.0, 0.1],
            [10.1, 10.0, 0.1, 0.0],
        ]);
        let l = arr1(&[42, 42, -7, -7]);
        let s = silhouette_precomputed(d.view(), l.view());
        assert!(s > 0.99, "expected ~1.0 regardless of label id choice, got {}", s);
    }
}
