//! Barber-Candès knockoff threshold with offset=1.
//!
//! Python reference: `scvalidate.recall_py.knockoff.knockoff_threshold_offset1`.
//!
//! Formula:
//! ```text
//! T = min { t > 0 : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q }
//! ```
//! Returns `+inf` if no such `t` exists.
//!
//! Algorithmic note: Python scans via `np.sum(w <= -t)` and `np.sum(w >= t)`
//! inside a loop over candidate thresholds (~30k per call on 32k genes →
//! ~30k × 32k = 10⁹ ops per call × 371 calls at 10k cells). This Rust version
//! sorts `w` once and uses `partition_point` binary searches, reducing each
//! call to `O(N log N)` up front and `O(log N)` per threshold.

use std::cmp::Ordering;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
pub fn knockoff_threshold_offset1(w: PyReadonlyArray1<f64>, fdr: f64) -> f64 {
    let w = w.as_array();

    // Candidate thresholds: sorted unique |w| over non-zero entries.
    let mut ts: Vec<f64> = w
        .iter()
        .filter(|&&x| x != 0.0)
        .map(|&x| x.abs())
        .collect();
    if ts.is_empty() {
        return f64::INFINITY;
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    ts.dedup_by(|a, b| *a == *b);

    // Sorted copy of the full vector for fast counting via binary search.
    let mut w_sorted: Vec<f64> = w.iter().copied().collect();
    w_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = w_sorted.len();

    for t in ts {
        // #{w <= -t}: largest index i such that w_sorted[i] <= -t.
        let num_neg = w_sorted.partition_point(|&x| x <= -t);
        // #{w >= t}: n - smallest index i such that w_sorted[i] >= t.
        let num_pos = n - w_sorted.partition_point(|&x| x < t);

        let num = 1.0_f64 + num_neg as f64;
        let denom = num_pos.max(1) as f64;
        if num / denom <= fdr {
            return t;
        }
    }
    f64::INFINITY
}
