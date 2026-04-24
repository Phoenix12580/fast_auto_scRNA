//! Harmony state struct + setup (mirrors `harmony::setup` in harmony.cpp).
#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// Ridge regularization strategy for MoE correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LambdaMode {
    /// Scalar `lambda` broadcast to every batch (every cluster, every
    /// outer iteration — matches the Korsunsky 2019 paper default).
    Fixed,
    /// Per-cluster `lambda_kb = alpha · E[k, b]`, recomputed each
    /// correction step from the live expected cluster-batch counts.
    /// Matches `harmony.cpp` `find_lambda_cpp(alpha, E.row(k).t())`.
    /// Harmony R default when user passes `lambda=NULL` (alpha=0.2).
    Dynamic,
}

/// User-facing hyperparameters. Defaults match `RunHarmony()` R wrapper
/// (see `R/ui.R` and `R/harmony_option.R` in the immunogenomics source).
#[derive(Debug, Clone)]
pub struct HarmonyParams {
    /// Number of soft k-means clusters. Harmony's R wrapper picks
    /// `min(round(N/30), 100)` when the user leaves it None.
    pub n_clusters: usize,
    /// Diversity clustering penalty (per batch). One scalar broadcast to
    /// all batches — Harmony's R wrapper supports one θ per covariate,
    /// but for scatlas' single-covariate MVP a scalar is enough.
    pub theta: f32,
    /// Soft k-means bandwidth. Smaller = harder clustering.
    pub sigma: f32,
    /// Ridge regularization — see [`LambdaMode`].
    pub lambda_mode: LambdaMode,
    /// Used when `lambda_mode == Fixed`. Intercept row always gets 0;
    /// batch rows get this scalar.
    pub lambda: f32,
    /// Used when `lambda_mode == Dynamic`. Harmony R default = 0.2.
    pub alpha: f32,
    /// Max outer Harmony iterations.
    pub max_iter_harmony: usize,
    /// Max soft-k-means iterations per outer step.
    pub max_iter_cluster: usize,
    /// Cluster-step convergence tolerance (on windowed objective).
    pub epsilon_cluster: f32,
    /// Harmony convergence tolerance.
    pub epsilon_harmony: f32,
    /// Proportion of cells updated per `update_R` block (0, 1].
    pub block_size: f32,
    /// Averaging window size (rounds) used in `check_convergence`.
    pub window_size: usize,
    /// Reproducibility seed for k-means init + randomized block order.
    pub seed: u64,
}

impl Default for HarmonyParams {
    fn default() -> Self {
        Self {
            n_clusters: 100,
            theta: 2.0,
            sigma: 0.1,
            lambda_mode: LambdaMode::Fixed,
            lambda: 1.0,
            alpha: 0.2,
            max_iter_harmony: 10,
            max_iter_cluster: 20,
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
            block_size: 0.05,
            window_size: 3,
            seed: 0,
        }
    }
}

/// Output of a full Harmony run.
#[derive(Debug)]
pub struct HarmonyResult {
    /// Corrected embedding, shape `(N, d)`.
    pub z_corrected: Array2<f32>,
    /// Soft cluster assignments, shape `(n_clusters, N)`.
    pub r: Array2<f32>,
    /// Cluster centroids, shape `(d, n_clusters)`.
    pub y: Array2<f32>,
    /// `objective_harmony` — one entry per outer iteration.
    pub objective_harmony: Vec<f32>,
    /// `objective_kmeans` — one entry per inner cluster step (across all
    /// outer iterations).
    pub objective_kmeans: Vec<f32>,
    /// Outer iteration at which convergence was declared (1-indexed);
    /// `None` if we hit `max_iter_harmony`.
    pub converged_at_iter: Option<usize>,
}

/// Mutable state held across the cluster / correct loop. Fields mirror
/// the C++ `class harmony` one-for-one. Matrices follow Armadillo's
/// column-major orientation in shape comments — ndarray is row-major so
/// when a field's "shape" reads `(K, N)` we store it as `Array2<f32>` of
/// shape `[K, N]` and access column-major equivalents via `.t()` /
/// `.column()` as needed.
pub struct HarmonyState {
    // Dimensions
    pub n_cells: usize,
    pub n_dims: usize,
    pub n_clusters: usize,
    pub n_batches: usize,

    // Hyperparameters (resolved)
    pub sigma: Array1<f32>,  // (K,) per-cluster bandwidth
    pub theta: Array1<f32>,  // (B,) per-batch diversity
    pub lambda: Array1<f32>, // (B+1,) ridge for Fixed mode; unused in Dynamic
    pub lambda_mode: LambdaMode,
    pub alpha: f32, // ridge coefficient for Dynamic mode
    pub max_iter_cluster: usize,
    pub epsilon_cluster: f32,
    pub epsilon_harmony: f32,
    pub block_size: f32,
    pub window_size: usize,
    pub seed: u64,
    pub max_iter_harmony: usize,

    // Batch data (built from batch_codes)
    pub batch_codes: Array1<i32>,  // (N,) per-cell batch index in 0..B-1
    pub batch_counts: Array1<u32>, // (B,) — number of cells per batch
    pub pr_b: Array1<f32>,         // (B,) — batch proportions

    // Embeddings — stored as (d, N) like Armadillo's default for matmul
    pub z_orig: Array2<f32>, // (d, N) original PCA, transposed from input
    pub z_corr: Array2<f32>, // (d, N) running corrected
    pub z_cos: Array2<f32>,  // (d, N) L2-normalized per column

    // Cluster state
    pub y: Array2<f32>,        // (d, K) centroids
    pub r: Array2<f32>,        // (K, N) soft assignments
    pub dist_mat: Array2<f32>, // (K, N) cosine distance 2*(1 - Y'Z_cos)
    pub o: Array2<f32>,        // (K, B) observed cluster-batch soft counts
    pub e: Array2<f32>,        // (K, B) expected cluster-batch soft counts

    // Diagnostics
    pub objective_kmeans: Vec<f32>,
    pub objective_kmeans_dist: Vec<f32>,
    pub objective_kmeans_entropy: Vec<f32>,
    pub objective_kmeans_cross: Vec<f32>,
    pub objective_harmony: Vec<f32>,
    pub kmeans_rounds: Vec<usize>,
    pub converged_at_iter: Option<usize>,
}

impl HarmonyState {
    pub fn new(pca: ArrayView2<f32>, batch_codes: ArrayView1<i32>, params: &HarmonyParams) -> Self {
        let n_cells = pca.nrows();
        let n_dims = pca.ncols();
        assert_eq!(
            batch_codes.len(),
            n_cells,
            "batch_codes length must equal pca rows"
        );
        assert!(n_cells >= 6, "refusing to run with fewer than 6 cells");

        let n_batches = (*batch_codes.iter().max().unwrap_or(&-1) + 1).max(1) as usize;
        let mut batch_counts = Array1::<u32>::zeros(n_batches);
        for &b in batch_codes.iter() {
            assert!(b >= 0, "batch code {} is negative", b);
            batch_counts[b as usize] += 1;
        }

        // Pr_b = counts / N
        let n_f = n_cells as f32;
        let pr_b = batch_counts.mapv(|c| c as f32 / n_f);

        // Transpose PCA (N, d) -> (d, N) for column-major ops
        let z_orig = pca.t().to_owned();

        // Z_cos = L2-normalize each column (each cell)
        let mut z_cos = z_orig.clone();
        normalize_columns_l2(&mut z_cos);

        // Block size safety: tiny N → shrink to 0.2 like harmony.cpp
        let block_size = if n_cells < 40 { 0.2 } else { params.block_size };

        // sigma broadcast to (K,). theta broadcast to (B,).
        let sigma = Array1::from_elem(params.n_clusters, params.sigma);
        let theta = Array1::from_elem(n_batches, params.theta);

        // lambda: intercept (row 0) gets 0, each batch gets params.lambda.
        let mut lambda = Array1::zeros(n_batches + 1);
        for i in 1..=n_batches {
            lambda[i] = params.lambda;
        }

        Self {
            n_cells,
            n_dims,
            n_clusters: params.n_clusters,
            n_batches,
            sigma,
            theta,
            lambda,
            lambda_mode: params.lambda_mode,
            alpha: params.alpha,
            max_iter_cluster: params.max_iter_cluster,
            epsilon_cluster: params.epsilon_cluster,
            epsilon_harmony: params.epsilon_harmony,
            block_size,
            window_size: params.window_size,
            seed: params.seed,
            max_iter_harmony: params.max_iter_harmony,
            batch_codes: batch_codes.to_owned(),
            batch_counts,
            pr_b,
            z_corr: Array2::zeros((n_dims, n_cells)),
            z_orig,
            z_cos,
            y: Array2::zeros((n_dims, params.n_clusters)),
            r: Array2::zeros((params.n_clusters, n_cells)),
            dist_mat: Array2::zeros((params.n_clusters, n_cells)),
            o: Array2::zeros((params.n_clusters, n_batches)),
            e: Array2::zeros((params.n_clusters, n_batches)),
            objective_kmeans: Vec::new(),
            objective_kmeans_dist: Vec::new(),
            objective_kmeans_entropy: Vec::new(),
            objective_kmeans_cross: Vec::new(),
            objective_harmony: Vec::new(),
            kmeans_rounds: Vec::new(),
            converged_at_iter: None,
        }
    }

    /// Windowed convergence check for the outer Harmony loop. Mirrors
    /// `check_convergence(type=1)` in harmony.cpp.
    pub fn check_convergence_harmony(&self) -> bool {
        let n = self.objective_harmony.len();
        if n < 2 {
            return false;
        }
        let obj_old = self.objective_harmony[n - 2];
        let obj_new = self.objective_harmony[n - 1];
        (obj_old - obj_new) / obj_old.abs() < self.epsilon_harmony
    }

    /// Convert back to `(N, d)` layout and return final result.
    pub fn finish(self) -> HarmonyResult {
        HarmonyResult {
            z_corrected: self.z_corr.t().to_owned(),
            r: self.r,
            y: self.y,
            objective_harmony: self.objective_harmony,
            objective_kmeans: self.objective_kmeans,
            converged_at_iter: self.converged_at_iter,
        }
    }
}

/// In-place L2-normalize each column of a `(d, N)` matrix.
/// Parallelized over columns — each column's norm + divide is independent.
pub fn normalize_columns_l2(m: &mut Array2<f32>) {
    m.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let n: f32 = col.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if n > 1e-12 {
                col.mapv_inplace(|x| x / n);
            }
        });
}

/// `x * log(x)` with `0 * log(0) := 0`. Rowwise safe variant of
/// Harmony's `safe_entropy` utility (takes and returns (K, N)).
pub fn safe_entropy(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| if v > 0.0 { v * v.ln() } else { 0.0 })
}
