//! Barber-Candès knockoff threshold with offset=1.
//!
//! Ported from `scvalidate_rust/src/knockoff.rs` (MIT). Takes a slice so
//! callers can pass an ndarray `.as_slice()`, a `Vec`, or any contiguous
//! buffer — no PyO3 dependency at this layer.
//!
//! Python reference: `scvalidate.recall_py.knockoff.knockoff_threshold_offset1`.
//!
//! Formula:
//! ```text
//! T = min { t > 0 : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q }
//! ```
//! Returns `+inf` if no such `t` exists.

use std::cmp::Ordering;

pub fn knockoff_threshold_offset1(w: &[f64], fdr: f64) -> f64 {
    let mut ts: Vec<f64> = w.iter().filter(|&&x| x != 0.0).map(|&x| x.abs()).collect();
    if ts.is_empty() {
        return f64::INFINITY;
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    ts.dedup_by(|a, b| *a == *b);

    let mut w_sorted: Vec<f64> = w.to_vec();
    w_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = w_sorted.len();

    for t in ts {
        let num_neg = w_sorted.partition_point(|&x| x <= -t);
        let num_pos = n - w_sorted.partition_point(|&x| x < t);

        let num = 1.0_f64 + num_neg as f64;
        let denom = num_pos.max(1) as f64;
        if num / denom <= fdr {
            return t;
        }
    }
    f64::INFINITY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_all_zero_returns_inf() {
        assert_eq!(knockoff_threshold_offset1(&[], 0.1), f64::INFINITY);
        assert_eq!(
            knockoff_threshold_offset1(&[0.0, 0.0, 0.0], 0.1),
            f64::INFINITY
        );
    }

    #[test]
    fn single_large_positive_accepted() {
        // One big positive W, no negatives: (1 + 0) / 1 = 1.0, only satisfies
        // fdr >= 1. At fdr=0.1 no threshold passes → inf.
        let w = vec![5.0];
        assert_eq!(knockoff_threshold_offset1(&w, 0.1), f64::INFINITY);
        assert_eq!(knockoff_threshold_offset1(&w, 1.0), 5.0);
    }

    #[test]
    fn mixed_signs_threshold() {
        // Ten positive W=1..=10, two negatives W=-0.5, -1.5.
        // At t=2.0: num_neg = 0, num_pos = 9 (2..=10), ratio = 1/9 ≈ 0.111.
        // At t=1.0: num_neg = 1 (-1.5), num_pos = 10 (1..=10), ratio = 2/10 = 0.2.
        let mut w: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        w.extend_from_slice(&[-0.5, -1.5]);
        let t = knockoff_threshold_offset1(&w, 0.15);
        assert!((t - 2.0).abs() < 1e-10, "threshold = {}", t);
    }
}
