//! Harmony 2.0 (Korsunsky 2019, refined in immunogenomics/harmony 1.2.x).
//!
//! Rust port of `references/harmony/src/*.cpp`. We follow the C++ method
//! decomposition 1:1 so readers can cross-reference:
//!
//! | Harmony C++ method                | Rust module              |
//! |-----------------------------------|--------------------------|
//! | `harmony::setup`, `allocate_…`    | `state::HarmonyState::new` |
//! | `harmony::init_cluster_cpp`       | `cluster::init_cluster`  |
//! | `harmony::cluster_cpp`            | `cluster::cluster`       |
//! | `harmony::update_R`               | `cluster::update_r`      |
//! | `harmony::compute_objective`      | `cluster::compute_objective` |
//! | `harmony::check_convergence`      | `cluster::check_convergence` |
//! | `harmony::moe_correct_ridge_cpp`  | `correct::moe_correct_ridge` |
//! | `utils::kmeans_centers`           | `kmeans_init::kmeans_centers` |
//! | `utils::safe_entropy`             | `state::safe_entropy`     |
//!
//! Convention differences vs the C++ reference:
//! - Data layout is `(N, d)` row-major at every public API boundary
//!   (matches numpy / scanpy `obsm['X_pca']`). Internally the algorithm
//!   works on `(d, N)` column-major views for matmul-heavy ops.
//! - Batch labels come in as `i32` codes (0..B-1) instead of a sparse
//!   one-hot matrix — we materialize `Phi` lazily from counts.
//! - Dense linear algebra uses `ndarray` + `rayon`; the only matrix
//!   inverse we need is `(B+1) × (B+1)`, solved in-crate
//!   (`linalg::invert_small`).
#![allow(
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_overindented_list_items
)]

pub mod cluster;
pub mod correct;
pub mod kmeans_init;
pub mod linalg;
pub mod matmul;
pub mod state;

pub use state::{HarmonyParams, HarmonyResult, HarmonyState, LambdaMode};

use ndarray::{ArrayView1, ArrayView2};

/// Top-level entry point: take a PCA embedding `(N, d)` and batch codes
/// `(N,)`, return a batch-corrected embedding of the same shape.
///
/// This mirrors `RunHarmony.default()`'s call sequence:
///
/// 1. `HarmonyState::new(...)` — setup + buffers
/// 2. `cluster::init_cluster(...)` — k-means init + initial R/E/O
/// 3. Loop `max_iter_harmony` times:
///    a. `cluster::cluster(...)` — soft k-means with diversity, until
///       `epsilon_cluster` or `max_iter_cluster`
///    b. `correct::moe_correct_ridge(...)` — MoE ridge correction of Z
///    c. Convergence check on `objective_harmony`
/// 4. Return `Z_corr` (transposed back to (N, d)) + stats.
pub fn harmony_integrate(
    pca: ArrayView2<f32>,
    batch_codes: ArrayView1<i32>,
    params: &HarmonyParams,
) -> HarmonyResult {
    let profile = std::env::var("SCATLAS_HARMONY_PROFILE").is_ok();
    let mk_inst = || std::time::Instant::now();

    let t0 = mk_inst();
    let mut state = HarmonyState::new(pca, batch_codes, params);
    if profile {
        eprintln!("[harmony] setup: {:.2?}", t0.elapsed());
    }

    let t0 = mk_inst();
    cluster::init_cluster(&mut state);
    if profile {
        eprintln!("[harmony] init_cluster: {:.2?}", t0.elapsed());
    }

    let mut total_cluster = std::time::Duration::ZERO;
    let mut total_correct = std::time::Duration::ZERO;

    for outer_iter in 0..params.max_iter_harmony {
        let t = mk_inst();
        cluster::cluster(&mut state);
        total_cluster += t.elapsed();
        let c_elapsed = t.elapsed();

        let t = mk_inst();
        correct::moe_correct_ridge(&mut state);
        total_correct += t.elapsed();
        let r_elapsed = t.elapsed();

        if profile {
            eprintln!(
                "[harmony] outer {}: cluster {:.2?}, correct {:.2?}",
                outer_iter + 1,
                c_elapsed,
                r_elapsed
            );
        }

        if state.check_convergence_harmony() {
            state.converged_at_iter = Some(outer_iter + 1);
            break;
        }
    }

    if profile {
        eprintln!(
            "[harmony] total cluster: {:.2?}, total correct: {:.2?}",
            total_cluster, total_correct
        );
    }

    state.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Reduce a (n, d) embedding to per-cluster per-batch mean distance
    /// so we can verify batch offset shrinks after Harmony.
    fn per_cluster_batch_mean_gap(z: &Array2<f32>, cluster: &[usize], batch: &[i32]) -> f32 {
        // For each (cluster, pair-of-batches) compute ||centroid_0 - centroid_1||,
        // then average.
        let n = z.nrows();
        let d = z.ncols();
        let n_clusters = *cluster.iter().max().unwrap() as usize + 1;
        let n_batches = *batch.iter().max().unwrap() as usize + 1;
        assert_eq!(n_batches, 2, "this helper only handles 2 batches");
        let mut centroid = vec![vec![vec![0.0_f32; d]; n_batches]; n_clusters];
        let mut count = vec![vec![0usize; n_batches]; n_clusters];
        for i in 0..n {
            let c = cluster[i];
            let b = batch[i] as usize;
            for di in 0..d {
                centroid[c][b][di] += z[[i, di]];
            }
            count[c][b] += 1;
        }
        for c in 0..n_clusters {
            for b in 0..n_batches {
                if count[c][b] > 0 {
                    for di in 0..d {
                        centroid[c][b][di] /= count[c][b] as f32;
                    }
                }
            }
        }
        let mut total = 0.0_f32;
        let mut valid = 0usize;
        for c in 0..n_clusters {
            if count[c][0] == 0 || count[c][1] == 0 {
                continue;
            }
            let mut s = 0.0_f32;
            for di in 0..d {
                let diff = centroid[c][0][di] - centroid[c][1][di];
                s += diff * diff;
            }
            total += s.sqrt();
            valid += 1;
        }
        if valid == 0 {
            0.0
        } else {
            total / valid as f32
        }
    }

    /// Build a synthetic 2-cluster, 2-batch embedding with a known batch
    /// offset applied to batch 1 only. Cells in cluster c cluster around
    /// a common point; batch 1 shifts them all by the same offset.
    fn synth_2_clusters_2_batches(
        n_per_cell_type: usize,
        d: usize,
        offset: f32,
        seed: u64,
    ) -> (Array2<f32>, Array1<i32>, Vec<usize>, Vec<i32>) {
        let n = n_per_cell_type * 2 * 2; // 2 clusters × 2 batches
        let mut z = Array2::<f32>::zeros((n, d));
        let mut batch_codes = Array1::<i32>::zeros(n);
        let mut cluster_gt = vec![0usize; n];
        let mut batch_gt = vec![0i32; n];

        let mut rng = kmeans_init::SplitMix64::new(seed);

        // Two well-separated cluster centers in d-dim space.
        let mut c0 = vec![0.0_f32; d];
        let mut c1 = vec![0.0_f32; d];
        c0[0] = -3.0;
        c1[0] = 3.0;

        // Batch offset direction (arbitrary, small compared to cluster gap).
        let mut off = vec![0.0_f32; d];
        off[1] = offset;

        let mut idx = 0usize;
        for cluster in 0..2 {
            for batch in 0..2 {
                for _ in 0..n_per_cell_type {
                    let centroid = if cluster == 0 { &c0 } else { &c1 };
                    for di in 0..d {
                        // Gaussian-ish noise via two uniforms (Box-Muller-lite).
                        let u1 = (rng.next_f32() as f32).max(1e-6);
                        let u2 = rng.next_f32() as f32;
                        let n01 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        let noise = n01 * 0.3;
                        let b_off = if batch == 1 { off[di] } else { 0.0 };
                        z[[idx, di]] = centroid[di] + b_off + noise;
                    }
                    batch_codes[idx] = batch as i32;
                    cluster_gt[idx] = cluster;
                    batch_gt[idx] = batch as i32;
                    idx += 1;
                }
            }
        }
        (z, batch_codes, cluster_gt, batch_gt)
    }

    #[test]
    fn harmony_reduces_known_batch_offset_on_synthetic() {
        // Build a 2 cluster × 2 batch synthetic with offset=2.5 (less
        // than the 6.0 cluster gap so identity is recoverable).
        let (z, batch_codes, cluster_gt, batch_gt) = synth_2_clusters_2_batches(200, 10, 2.5, 42);

        let gap_before = per_cluster_batch_mean_gap(&z, &cluster_gt, &batch_gt);

        let params = HarmonyParams {
            n_clusters: 2,
            seed: 7,
            ..HarmonyParams::default()
        };

        let result = harmony_integrate(z.view(), batch_codes.view(), &params);
        assert_eq!(result.z_corrected.shape(), &[800, 10]);
        let gap_after = per_cluster_batch_mean_gap(&result.z_corrected, &cluster_gt, &batch_gt);

        // Harmony should shrink the per-cluster batch gap substantially.
        // Tolerate imperfect recovery (≤ 30% of original gap is a solid pass).
        assert!(
            gap_after < 0.3 * gap_before,
            "gap_before = {}, gap_after = {} (Harmony failed to remove ≥70% of \
             known batch offset)",
            gap_before,
            gap_after,
        );
    }

    #[test]
    fn harmony_preserves_cluster_separation() {
        // Same synthetic; verify the two biological clusters remain far
        // apart after correction (Harmony shouldn't collapse signal).
        let (z, batch_codes, cluster_gt, _) = synth_2_clusters_2_batches(150, 8, 1.5, 3);

        let params = HarmonyParams {
            n_clusters: 2,
            ..HarmonyParams::default()
        };
        let result = harmony_integrate(z.view(), batch_codes.view(), &params);

        // Cluster 0 mean vs cluster 1 mean in corrected space.
        let d = result.z_corrected.ncols();
        let mut m0 = vec![0.0_f32; d];
        let mut m1 = vec![0.0_f32; d];
        let mut n0 = 0usize;
        let mut n1 = 0usize;
        for i in 0..result.z_corrected.nrows() {
            for di in 0..d {
                if cluster_gt[i] == 0 {
                    m0[di] += result.z_corrected[[i, di]];
                } else {
                    m1[di] += result.z_corrected[[i, di]];
                }
            }
            if cluster_gt[i] == 0 {
                n0 += 1;
            } else {
                n1 += 1;
            }
        }
        for di in 0..d {
            m0[di] /= n0 as f32;
            m1[di] /= n1 as f32;
        }
        let sep: f32 = (0..d)
            .map(|di| (m0[di] - m1[di]).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            sep > 2.0,
            "cluster separation collapsed to {} (Harmony over-corrected)",
            sep
        );
    }
}
