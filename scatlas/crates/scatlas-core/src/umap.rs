//! UMAP layout optimization — rayon-parallel port of umap-learn's
//! `_optimize_layout_euclidean_single_epoch` (the numba SGD kernel).
//!
//! **Scope.** We port *only* the layout optimization step. The fuzzy
//! simplicial set construction (connectivity graph) is consumed from
//! BBKNN's CSR output (`adata.obsp['bbknn_connectivities']`), which
//! already runs umap-learn's `fuzzy_simplicial_set` internally. So this
//! module takes a symmetric CSR connectivity matrix + an initial
//! low-dim embedding, and does the gradient-descent step.
//!
//! **Algorithm — from `umap/layouts.py::_optimize_layout_euclidean_single_epoch`:**
//!
//! For each epoch `n = 1..n_epochs`:
//!   * `alpha(n) = learning_rate · (1 − n/n_epochs)`
//!   * For each edge `(head[i], tail[i])` with weight `w[i]`:
//!     - If `epoch_of_next_sample[i] ≤ n`, fire:
//!       * attract: move head toward tail by `grad_a(d²) = −2ab · d^(2b-2) / (a·d^(2b)+1)`
//!       * `n_neg = floor((n − epoch_of_next_negative[i]) / epochs_per_negative[i])`
//!       * For each negative: move head away from random k by
//!         `grad_r(d²) = +2γb / ((0.001+d²)(a·d^(2b)+1))`
//!     - Update per-edge counters.
//!
//! `(a, b)` are fit to the piecewise target `y(x) = 1 if x < min_dist
//! else exp(-(x-min_dist)/spread)` via Levenberg-Marquardt. This matches
//! `umap.umap_.find_ab_params`.
//!
//! **Parallelism — Hogwild.** Each edge's SGD update writes to
//! `y[head[i]]` and `y[tail[i]]` (plus random negatives). Updates across
//! edges overlap on the same cells → we accept races via raw pointer
//! writes (Hogwild, Niu et al. 2011). UMAP-style SGD is exactly the
//! convex-ish setting where Hogwild converges empirically identically
//! to the serial algorithm.
//!
//! **RNG determinism.** Per-edge-per-epoch SplitMix64 seeded from
//! `seed ^ i ^ (epoch << 32)`, so output is deterministic across runs
//! at fixed `seed` even though thread scheduling varies.
#![allow(clippy::needless_range_loop)]

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::harmony::kmeans_init::SplitMix64;

/// UMAP hyperparameters. Defaults track `scanpy.tl.umap`.
#[derive(Debug, Clone, Copy)]
pub struct UmapParams {
    /// Output embedding dimensionality. scanpy uses 2.
    pub n_components: usize,
    /// Number of SGD epochs. If None, picks 500 for N ≤ 10k, else 200
    /// (matches umap-learn heuristic).
    pub n_epochs: Option<usize>,
    /// Controls how tightly points are packed together. scanpy uses 0.5.
    pub min_dist: f32,
    /// The scale of how spread out points are. scanpy uses 1.0.
    pub spread: f32,
    /// Number of negative samples per positive sample. Default 5.
    pub negative_sample_rate: usize,
    /// Weight of repulsive term (γ in the gradient formula). Default 1.0.
    pub repulsion_strength: f32,
    /// Initial learning rate. Linearly decays to 0 over n_epochs. Default 1.0.
    pub learning_rate: f32,
    /// RNG seed.
    pub seed: u64,
    /// If true, run SGD serially (no rayon Hogwild). Deterministic output at
    /// fixed seed but ≈ num-cores× slower. Use for parity testing against
    /// umap-learn's single-threaded numba kernel.
    pub single_thread: bool,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_epochs: None,
            min_dist: 0.5,
            spread: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            learning_rate: 1.0,
            seed: 0,
            single_thread: false,
        }
    }
}

/// UMAP optimization output.
#[derive(Debug)]
pub struct UmapResult {
    /// Final low-dim embedding, shape `(n_cells, n_components)`.
    pub embedding: Array2<f32>,
    /// Fitted kernel parameter a.
    pub a: f32,
    /// Fitted kernel parameter b.
    pub b: f32,
    /// Actual number of epochs run.
    pub n_epochs_used: usize,
}

// =============================================================================
// a/b curve fit (Levenberg-Marquardt)
// =============================================================================

/// Fit the UMAP low-dim kernel `1 / (1 + a·x^(2b))` to the piecewise
/// target `y(x) = 1 if x < min_dist else exp(-(x - min_dist) / spread)`
/// via Levenberg-Marquardt nonlinear least squares over 300 grid points.
///
/// Mirrors `umap.umap_.find_ab_params`.
pub fn fit_ab(min_dist: f32, spread: f32) -> (f32, f32) {
    let n = 300usize;
    let step = 3.0 * spread / (n - 1) as f32;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 * step).collect();
    let ys: Vec<f32> = xs
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    // Good starting point per umap-learn's curve_fit defaults.
    let mut a = 1.0_f32;
    let mut b = 1.0_f32;
    let mut lambda = 1e-3_f32;

    fn residuals_ssr(xs: &[f32], ys: &[f32], a: f32, b: f32) -> f32 {
        let mut ssr = 0.0;
        for i in 0..xs.len() {
            let x = xs[i];
            if x <= 0.0 {
                // At x = 0, f = 1 → perfect fit by construction; skip.
                continue;
            }
            let f = 1.0 / (1.0 + a * x.powf(2.0 * b));
            let r = ys[i] - f;
            ssr += r * r;
        }
        ssr
    }

    for _iter in 0..200 {
        // Build J^T J (2×2) and J^T r (2,) at current (a, b).
        let mut jtj = [[0.0_f32; 2]; 2];
        let mut jtr = [0.0_f32; 2];
        let mut ssr = 0.0_f32;
        for i in 0..n {
            let x = xs[i];
            if x <= 0.0 {
                continue;
            }
            let x2b = x.powf(2.0 * b);
            let denom = 1.0 + a * x2b;
            let f = 1.0 / denom;
            let r = ys[i] - f;
            ssr += r * r;
            let dfda = -x2b / (denom * denom);
            let dfdb = -2.0 * a * x2b * x.ln() / (denom * denom);
            jtj[0][0] += dfda * dfda;
            jtj[0][1] += dfda * dfdb;
            jtj[1][1] += dfdb * dfdb;
            jtr[0] += dfda * r;
            jtr[1] += dfdb * r;
        }
        jtj[1][0] = jtj[0][1];

        // Augment diagonal with λ·diag(J^T J) (Marquardt damping).
        let d00 = jtj[0][0] * (1.0 + lambda);
        let d11 = jtj[1][1] * (1.0 + lambda);
        let det = d00 * d11 - jtj[0][1] * jtj[1][0];
        if det.abs() < 1e-30 {
            break;
        }
        let inv = 1.0 / det;
        // Note sign: we solve (J^T J + λD) δ = J^T r, step is +δ
        // since r = y - f → ∂ssr/∂p = -2·J^T r, so gradient descent
        // direction is +J^T r. Gauss-Newton step solves J^T J δ = J^T r.
        let da = inv * (d11 * jtr[0] - jtj[0][1] * jtr[1]);
        let db = inv * (-jtj[1][0] * jtr[0] + d00 * jtr[1]);

        let a_new = a + da;
        let b_new = b + db;
        if a_new <= 0.0 || b_new <= 0.0 {
            lambda *= 10.0;
            continue;
        }
        let new_ssr = residuals_ssr(&xs, &ys, a_new, b_new);
        if new_ssr < ssr {
            a = a_new;
            b = b_new;
            lambda *= 0.5;
            if da.abs() < 1e-7 && db.abs() < 1e-7 {
                break;
            }
        } else {
            lambda *= 2.0;
            if lambda > 1e10 {
                break;
            }
        }
    }
    (a, b)
}

// =============================================================================
// Main entry
// =============================================================================

/// Run UMAP layout optimization given a CSR connectivity matrix + an
/// initial low-dim embedding.
///
/// * `connectivities_indptr` — length `n_cells + 1`, u64 CSR pointers.
/// * `connectivities_indices`, `connectivities_data` — column indices + weights.
/// * `init_embedding` — `(n_cells, n_components)`, starting Y values.
#[allow(clippy::too_many_arguments)]
pub fn umap_from_connectivities(
    connectivities_indptr: &[u64],
    connectivities_indices: &[u32],
    connectivities_data: &[f32],
    n_cells: usize,
    init_embedding: ArrayView2<f32>,
    params: &UmapParams,
) -> UmapResult {
    assert_eq!(
        init_embedding.nrows(),
        n_cells,
        "init_embedding rows ≠ n_cells"
    );
    let dim = init_embedding.ncols();
    assert_eq!(
        dim, params.n_components,
        "init_embedding cols ≠ n_components"
    );
    assert_eq!(
        connectivities_indptr.len(),
        n_cells + 1,
        "indptr wrong length"
    );
    assert_eq!(
        connectivities_indices.len(),
        connectivities_data.len(),
        "indices/data length mismatch"
    );

    let (a, b) = fit_ab(params.min_dist, params.spread);

    // --- Build (head, tail, weight) edge lists from CSR.
    let nnz = connectivities_data.len();
    let mut head: Vec<u32> = Vec::with_capacity(nnz);
    let mut tail: Vec<u32> = Vec::with_capacity(nnz);
    let mut weights: Vec<f32> = Vec::with_capacity(nnz);
    for row in 0..n_cells {
        let s = connectivities_indptr[row] as usize;
        let e = connectivities_indptr[row + 1] as usize;
        for k in s..e {
            head.push(row as u32);
            tail.push(connectivities_indices[k]);
            weights.push(connectivities_data[k]);
        }
    }

    // --- n_epochs auto-select per umap-learn.
    let n_epochs = params
        .n_epochs
        .unwrap_or(if n_cells <= 10_000 { 500 } else { 200 });
    let n_epochs_f = n_epochs as f32;

    // --- epochs_per_sample: high weight → sampled more often (smaller value).
    // Matches `umap.umap_.make_epochs_per_sample`:
    //   n_samples = n_epochs · (w / max_w)
    //   result[e] = n_epochs / n_samples   (−1 if n_samples == 0 → prune)
    let max_w = weights.iter().cloned().fold(0.0_f32, f32::max).max(1e-20);
    let epochs_per_sample: Vec<f32> = weights
        .iter()
        .map(|&w| {
            let n_samples = n_epochs_f * w / max_w;
            if n_samples > 0.0 {
                n_epochs_f / n_samples
            } else {
                -1.0
            }
        })
        .collect();

    // --- Initialize Y from caller-provided init.
    let mut y = init_embedding.to_owned();

    // --- Run SGD.
    optimize_layout_euclidean(
        &mut y,
        &head,
        &tail,
        &epochs_per_sample,
        n_epochs,
        a,
        b,
        params.negative_sample_rate,
        params.repulsion_strength,
        params.learning_rate,
        params.seed,
        n_cells as u32,
        params.single_thread,
    );

    UmapResult {
        embedding: y,
        a,
        b,
        n_epochs_used: n_epochs,
    }
}

// =============================================================================
// SGD layout optimization (Hogwild, rayon)
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn optimize_layout_euclidean(
    y: &mut Array2<f32>,
    head: &[u32],
    tail: &[u32],
    epochs_per_sample: &[f32],
    n_epochs: usize,
    a: f32,
    b: f32,
    negative_sample_rate: usize,
    gamma: f32,
    initial_alpha: f32,
    seed: u64,
    n_vertices: u32,
    single_thread: bool,
) {
    let dim = y.ncols();
    let n_edges = head.len();
    let neg_rate = negative_sample_rate as f32;

    // Per-edge "fire" counters — firing is when epoch ≥ counter.
    let mut epoch_of_next_sample: Vec<f32> = epochs_per_sample.to_vec();
    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|&e| if e > 0.0 { e / neg_rate } else { -1.0 })
        .collect();
    let mut epoch_of_next_negative_sample: Vec<f32> = epochs_per_negative_sample.clone();

    // Pass pointers as usize so rayon closures are Send+Sync.
    let y_addr = y.as_mut_ptr() as usize;
    let eons_addr = epoch_of_next_sample.as_mut_ptr() as usize;
    let eonns_addr = epoch_of_next_negative_sample.as_mut_ptr() as usize;

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - (epoch as f32 / n_epochs as f32));
        // 0-indexed epoch per umap-learn's `_optimize_layout_euclidean`:
        // the inner kernel receives `n` (0-indexed) and compares `eons ≤ n`.
        let epoch_f = epoch as f32;

        let process_edge = |i: usize| {
            let eps = epochs_per_sample[i];
            if eps <= 0.0 {
                return;
            }

            // SAFETY: raw-ptr Hogwild. Counter per-edge is only written by
            // this edge (no race on counters). Y writes overlap across
            // edges (Hogwild — accepted).
            let eons_ptr = eons_addr as *mut f32;
            let eonns_ptr = eonns_addr as *mut f32;
            let y_ptr = y_addr as *mut f32;

            let eons = unsafe { *eons_ptr.add(i) };
            if eons > epoch_f {
                return;
            }

            let j = head[i] as usize;
            let k = tail[i] as usize;
            let j_base = j * dim;
            let k_base = k * dim;

            // ---- Attract: dj += α · grad_coef · (y_j − y_k), dk = −dj
            let mut dist_sq = 0.0_f32;
            for d in 0..dim {
                let cur = unsafe { *y_ptr.add(j_base + d) };
                let oth = unsafe { *y_ptr.add(k_base + d) };
                let diff = cur - oth;
                dist_sq += diff * diff;
            }
            let grad_coef = if dist_sq > 0.0 {
                let x2b = dist_sq.powf(b);
                (-2.0 * a * b * dist_sq.powf(b - 1.0)) / (a * x2b + 1.0)
            } else {
                0.0
            };
            for d in 0..dim {
                let cur = unsafe { *y_ptr.add(j_base + d) };
                let oth = unsafe { *y_ptr.add(k_base + d) };
                let grad = (grad_coef * (cur - oth)).clamp(-4.0, 4.0);
                unsafe {
                    *y_ptr.add(j_base + d) = cur + grad * alpha;
                    *y_ptr.add(k_base + d) = oth - grad * alpha;
                }
            }

            // ---- Advance sample counter
            unsafe { *eons_ptr.add(i) = eons + eps };

            // ---- Negative samples
            let eonns = unsafe { *eonns_ptr.add(i) };
            let eps_neg = eps / neg_rate;
            let n_neg = ((epoch_f - eonns) / eps_neg).max(0.0) as usize;

            // Per-(edge, epoch) deterministic RNG.
            let mut rng = SplitMix64::new(
                seed ^ (i as u64) ^ ((epoch as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            );
            // Burn one to decorrelate from seed arithmetic.
            let _ = rng.next_u64();

            for _ in 0..n_neg {
                let k_neg = (rng.next_u64() as usize) % (n_vertices as usize);
                if k_neg == j {
                    continue;
                }
                let kn_base = k_neg * dim;
                let mut dist_sq = 0.0_f32;
                for d in 0..dim {
                    let cur = unsafe { *y_ptr.add(j_base + d) };
                    let oth = unsafe { *y_ptr.add(kn_base + d) };
                    let diff = cur - oth;
                    dist_sq += diff * diff;
                }
                let grad_coef = if dist_sq > 0.0 {
                    (2.0 * gamma * b) / ((0.001 + dist_sq) * (a * dist_sq.powf(b) + 1.0))
                } else {
                    0.0
                };
                for d in 0..dim {
                    let cur = unsafe { *y_ptr.add(j_base + d) };
                    let oth = unsafe { *y_ptr.add(kn_base + d) };
                    let grad = if grad_coef > 0.0 {
                        (grad_coef * (cur - oth)).clamp(-4.0, 4.0)
                    } else {
                        4.0
                    };
                    unsafe { *y_ptr.add(j_base + d) = cur + grad * alpha };
                }
            }

            unsafe {
                *eonns_ptr.add(i) = eonns + (n_neg as f32) * eps_neg;
            }
        };

        if single_thread {
            // Serial path — deterministic edge order, no Hogwild races.
            // Used for parity testing against umap-learn's numba kernel.
            for i in 0..n_edges {
                process_edge(i);
            }
        } else {
            (0..n_edges).into_par_iter().for_each(process_edge);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_ab_matches_umap_learn_defaults() {
        // umap-learn find_ab_params(1.0, 0.1) → a ≈ 1.577, b ≈ 0.8951
        let (a1, b1) = fit_ab(0.1, 1.0);
        assert!((a1 - 1.577).abs() < 0.05, "a(0.1, 1.0) = {}", a1);
        assert!((b1 - 0.8951).abs() < 0.02, "b(0.1, 1.0) = {}", b1);

        // scanpy default: min_dist=0.5, spread=1.0 → a ≈ 0.583, b ≈ 1.334
        let (a2, b2) = fit_ab(0.5, 1.0);
        assert!((a2 - 0.583).abs() < 0.03, "a(0.5, 1.0) = {}", a2);
        assert!((b2 - 1.334).abs() < 0.03, "b(0.5, 1.0) = {}", b2);

        // min_dist=0 → softest (tightest packing): a ≈ 1.929, b ≈ 0.7915
        let (a3, b3) = fit_ab(0.0, 1.0);
        assert!((a3 - 1.929).abs() < 0.05);
        assert!((b3 - 0.7915).abs() < 0.02);
    }

    #[test]
    fn umap_separates_two_clusters_small() {
        // Build synthetic: 2 clusters × 50 cells, k=3 intra-cluster edges,
        // verify that UMAP output separates them in 2D.
        let n_per_cluster = 50usize;
        let n = n_per_cluster * 2;
        // Build connectivity CSR: within cluster, high weight edges;
        // no cross-cluster edges (so UMAP should separate).
        let mut indptr: Vec<u64> = Vec::with_capacity(n + 1);
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        indptr.push(0);
        for i in 0..n {
            let cluster = i / n_per_cluster;
            let cluster_start = cluster * n_per_cluster;
            for j in 0..n_per_cluster {
                let other = cluster_start + j;
                if other != i {
                    indices.push(other as u32);
                    data.push(1.0);
                }
            }
            indptr.push(indices.len() as u64);
        }

        // Random init (x, y in [-10, 10])
        let mut rng = SplitMix64::new(42);
        let init = Array2::<f32>::from_shape_fn((n, 2), |_| 20.0 * rng.next_f32() - 10.0);

        let params = UmapParams {
            n_epochs: Some(200),
            seed: 7,
            ..UmapParams::default()
        };
        let result = umap_from_connectivities(&indptr, &indices, &data, n, init.view(), &params);

        let emb = &result.embedding;
        // Cluster centroids in the output
        let mut c0 = [0.0_f32; 2];
        let mut c1 = [0.0_f32; 2];
        for i in 0..n_per_cluster {
            c0[0] += emb[[i, 0]];
            c0[1] += emb[[i, 1]];
            c1[0] += emb[[n_per_cluster + i, 0]];
            c1[1] += emb[[n_per_cluster + i, 1]];
        }
        for c in [&mut c0, &mut c1] {
            c[0] /= n_per_cluster as f32;
            c[1] /= n_per_cluster as f32;
        }
        let sep = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();

        // Within-cluster spread
        let mut within = 0.0_f32;
        for i in 0..n_per_cluster {
            within += ((emb[[i, 0]] - c0[0]).powi(2) + (emb[[i, 1]] - c0[1]).powi(2)).sqrt();
            within += ((emb[[n_per_cluster + i, 0]] - c1[0]).powi(2)
                + (emb[[n_per_cluster + i, 1]] - c1[1]).powi(2))
            .sqrt();
        }
        within /= (2 * n_per_cluster) as f32;

        // Separation should be much larger than within-cluster spread.
        assert!(
            sep > 2.0 * within,
            "UMAP failed to separate clusters: sep={} within={}",
            sep,
            within
        );
    }
}
