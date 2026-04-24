//! Local Inverse Simpson's Index (LISI).
//!
//! Given per-cell k-nearest-neighbor distances + neighbor labels, compute
//! for each cell the Gaussian-weighted inverse Simpson's index over the
//! label distribution:
//!
//! ```text
//! p_ℓ = Σ_{j : label(j) = ℓ} w_ij            (w_ij = exp(-d_ij² · β_i), Σ w = 1)
//! LISI_i = 1 / Σ_ℓ p_ℓ²
//! ```
//!
//! `β_i` (= 1 / (2 σ_i²) in Gaussian terms) is calibrated per cell by
//! bisection so that the *perplexity* of `{w_ij}` matches the requested
//! target, mirroring the t-SNE / harmonypy / scib_metrics formulation:
//!
//! ```text
//! perplexity({w}) = 2^H({w}),   H({w}) = -Σ w_ij log₂ w_ij
//! ```
//!
//! Range: 1 (all neighbors in one label) ≤ LISI ≤ #distinct_labels.
//! iLISI uses batch labels — larger is better.
//! cLISI uses cell-type labels — smaller is better.

use std::fmt;

use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LisiError {
    pub message: String,
}

impl fmt::Display for LisiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for LisiError {}

/// Max perplexity-calibration iterations. 50 is overkill for bisection on
/// a monotone function; matches harmonypy's default.
const MAX_ITER: usize = 50;

/// Bisection tolerance on log-perplexity difference.
const TOL: f32 = 1e-5;

/// LISI per cell.
///
/// * `knn_distances` shape (n_cells, k) — Euclidean (or any non-negative)
///   distances to the k nearest neighbors of each cell.
/// * `knn_labels` shape (n_cells, k) — label code of each neighbor,
///   `i32`. `i32::MIN` is treated as "padded / no neighbor" and skipped.
/// * `perplexity` — effective neighborhood size for the Gaussian kernel.
///   scib_metrics / harmonypy default is 30. A typical choice is `k/3`.
pub fn lisi(
    knn_distances: ArrayView2<f32>,
    knn_labels: ArrayView2<i32>,
    perplexity: f32,
) -> Result<Vec<f32>, LisiError> {
    if knn_distances.shape() != knn_labels.shape() {
        return Err(LisiError {
            message: format!(
                "knn_distances shape {:?} != knn_labels shape {:?}",
                knn_distances.shape(),
                knn_labels.shape()
            ),
        });
    }
    if perplexity <= 1.0 {
        return Err(LisiError {
            message: format!("perplexity must be > 1.0, got {}", perplexity),
        });
    }

    let (n_cells, k) = (knn_distances.nrows(), knn_distances.ncols());
    if k == 0 {
        return Ok(vec![1.0; n_cells]);
    }
    let log_perp_target = perplexity.log2();

    let mut out = vec![1.0_f32; n_cells];
    out.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let dists = knn_distances.row(i);
        let labels_i = knn_labels.row(i);
        *slot = lisi_one_cell(dists, labels_i, log_perp_target);
    });

    Ok(out)
}

#[inline]
fn lisi_one_cell(dists: ArrayView1<f32>, labels: ArrayView1<i32>, log_perp_target: f32) -> f32 {
    // Collect non-padded neighbors.
    let mut valid_d: Vec<f32> = Vec::with_capacity(dists.len());
    let mut valid_l: Vec<i32> = Vec::with_capacity(dists.len());
    for j in 0..dists.len() {
        let l = labels[j];
        if l == i32::MIN {
            continue;
        }
        valid_d.push(dists[j]);
        valid_l.push(l);
    }
    let k = valid_d.len();
    if k == 0 {
        return 1.0;
    }
    if k == 1 {
        // Single neighbor — LISI is 1 by definition.
        return 1.0;
    }

    // Scale distances by their minimum so that exp overflow is avoided
    // even when all distances are large (matches harmonypy convention).
    let d2: Vec<f32> = valid_d.iter().map(|&d| d * d).collect();

    // Bisection on beta = 1 / (2σ²). Start with beta=1, widen bounds as
    // needed. Same structure as scipy.stats.binom_tail_bisect,
    // harmonypy _get_sigma.
    let mut beta = 1.0_f32;
    let mut beta_lo = f32::NEG_INFINITY;
    let mut beta_hi = f32::INFINITY;

    for _ in 0..MAX_ITER {
        let (weights, h) = gauss_weights_entropy(&d2, beta);
        let diff = h - log_perp_target;
        if diff.abs() < TOL {
            return inv_simpson(&weights, &valid_l);
        }
        if diff > 0.0 {
            // Entropy too high (distribution too uniform) → beta too low
            // → need to increase beta.
            beta_lo = beta;
            beta = if beta_hi.is_infinite() {
                beta * 2.0
            } else {
                (beta + beta_hi) * 0.5
            };
        } else {
            // Entropy too low → beta too high → decrease beta.
            beta_hi = beta;
            beta = if beta_lo.is_infinite() {
                beta * 0.5
            } else {
                (beta + beta_lo) * 0.5
            };
        }
        if !beta.is_finite() || beta <= 0.0 {
            break;
        }
    }

    let (weights, _) = gauss_weights_entropy(&d2, beta);
    inv_simpson(&weights, &valid_l)
}

#[inline]
fn gauss_weights_entropy(d2: &[f32], beta: f32) -> (Vec<f32>, f32) {
    // Subtract min(d²·β) for numerical stability.
    let min_logit = d2.iter().map(|&x| x * beta).fold(f32::INFINITY, f32::min);
    let mut weights: Vec<f32> = d2.iter().map(|&x| (-x * beta + min_logit).exp()).collect();
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    // Shannon entropy in log2 (to match perplexity = 2^H convention).
    let h: f32 = weights
        .iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| -w * w.log2())
        .sum();
    (weights, h)
}

#[inline]
fn inv_simpson(weights: &[f32], labels: &[i32]) -> f32 {
    // Sum weights per label into a small linear-scan dict. For ≤ ~30
    // neighbors the hash overhead beats a BTreeMap here.
    let mut ls: Vec<i32> = Vec::with_capacity(4);
    let mut ws: Vec<f32> = Vec::with_capacity(4);
    for j in 0..weights.len() {
        let l = labels[j];
        let w = weights[j];
        match ls.iter().position(|&x| x == l) {
            Some(pos) => ws[pos] += w,
            None => {
                ls.push(l);
                ws.push(w);
            }
        }
    }
    let simpson: f32 = ws.iter().map(|&p| p * p).sum();
    if simpson <= 0.0 {
        1.0
    } else {
        1.0 / simpson
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn uniform_two_labels_gives_near_two() {
        // Cell with 6 neighbors, 3 in label 0 and 3 in label 1, all
        // distances equal → uniform weights → LISI = 1 / (0.5² + 0.5²) = 2.
        let d: Array2<f32> = array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
        let l: Array2<i32> = array![[0, 0, 0, 1, 1, 1]];
        let out = lisi(d.view(), l.view(), 4.0).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-4, "LISI = {}", out[0]);
    }

    #[test]
    fn single_label_gives_one() {
        let d: Array2<f32> = array![[1.0, 1.0, 1.0, 1.0]];
        let l: Array2<i32> = array![[7, 7, 7, 7]];
        let out = lisi(d.view(), l.view(), 3.0).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn perplexity_calibration_recovers_expected() {
        // With 4 neighbors and perplexity = 4 (max), weights are uniform,
        // so LISI over 4 distinct labels equals 4.
        let d: Array2<f32> = array![[0.1, 0.5, 0.9, 1.2]];
        let l: Array2<i32> = array![[0, 1, 2, 3]];
        let out = lisi(d.view(), l.view(), 4.0).unwrap();
        // Perplexity 4 on 4 points can only be reached in the limit
        // beta → 0; bisection will push there and LISI ≈ 4.
        assert!(out[0] > 3.8, "LISI = {}", out[0]);
    }

    #[test]
    fn asymmetric_distances_bias_toward_near_label() {
        // Near neighbors are label 0, far ones are label 1. LISI should
        // tilt toward 1 (dominant label).
        let d: Array2<f32> = array![[0.1, 0.2, 0.3, 5.0, 5.0, 5.0]];
        let l: Array2<i32> = array![[0, 0, 0, 1, 1, 1]];
        let out = lisi(d.view(), l.view(), 3.0).unwrap();
        assert!(out[0] < 1.5, "LISI should be near 1, got {}", out[0]);
    }

    #[test]
    fn padded_neighbors_ignored() {
        let d: Array2<f32> = array![[1.0, 1.0, 1.0, 1.0]];
        let l: Array2<i32> = array![[0, 0, i32::MIN, i32::MIN]];
        let out = lisi(d.view(), l.view(), 1.5).unwrap();
        // Only 2 valid neighbors, both label 0 → LISI = 1.
        assert!((out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn shape_mismatch_errors() {
        let d: Array2<f32> = array![[1.0, 1.0, 1.0]];
        let l: Array2<i32> = array![[0, 1]];
        assert!(lisi(d.view(), l.view(), 2.0).is_err());
    }

    #[test]
    fn many_cells_parallel() {
        // Just check that the parallel path doesn't crash and gives
        // monotone results.
        let n = 500;
        let k = 10;
        let mut d = Vec::with_capacity(n * k);
        let mut l = Vec::with_capacity(n * k);
        for i in 0..n {
            for j in 0..k {
                d.push((j + 1) as f32);
                l.push(((i + j) % 3) as i32);
            }
        }
        let d = Array2::from_shape_vec((n, k), d).unwrap();
        let l = Array2::from_shape_vec((n, k), l).unwrap();
        let out = lisi(d.view(), l.view(), 5.0).unwrap();
        assert_eq!(out.len(), n);
        for v in &out {
            assert!(*v >= 1.0 && *v <= 3.0, "LISI out of range: {}", v);
        }
    }

    #[test]
    fn two_batch_moderate_perplexity_ilisi_range() {
        // Batch-balanced neighborhood with 30 neighbors, equal split:
        // iLISI at perplexity=10 should be close to 2 (max for 2 batches).
        let k = 30;
        let d = Array1::from_iter((0..k).map(|i| 1.0 + i as f32 * 0.01));
        let l = Array1::from_iter((0..k).map(|i| (i % 2) as i32));
        let d2 = Array2::from_shape_vec((1, k), d.to_vec()).unwrap();
        let l2 = Array2::from_shape_vec((1, k), l.to_vec()).unwrap();
        let out = lisi(d2.view(), l2.view(), 10.0).unwrap();
        assert!(out[0] > 1.8, "iLISI = {}", out[0]);
    }
}
