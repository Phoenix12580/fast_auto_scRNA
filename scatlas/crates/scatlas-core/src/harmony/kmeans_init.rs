//! Cosine-space k-means++ initialization.
//!
//! Harmony uses R's `stats::kmeans` with 10 random starts and 25 Lloyd
//! iterations on cosine-normalized cells. We replicate the behavior with
//! a pure-Rust k-means++ seeding + Lloyd refinement. The result won't be
//! bit-identical to R's RNG, but Harmony's own clustering loop will
//! refine the centroids anyway, so final `Z_corr` differs only in iter
//! count, not quality.

#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use super::matmul::matmul_kdn_par;
use super::state::normalize_columns_l2;

/// Run k-means++ init + a few Lloyd iterations on `Z_cos` (shape
/// `(d, N)`). Returns centroid matrix shape `(d, K)` with each column
/// L2-normalized (matches `Y` convention in `init_cluster_cpp`).
///
/// * `n_starts` independent k-means++ seedings; best by inertia wins.
/// * `n_iter` Lloyd iterations per seed.
/// * `seed` reproducibility seed.
pub fn kmeans_centers(
    z_cos: ArrayView2<f32>,
    k: usize,
    n_starts: usize,
    n_iter: usize,
    seed: u64,
) -> Array2<f32> {
    let (_d, n) = (z_cos.nrows(), z_cos.ncols());
    assert!(k >= 1, "k must be >= 1");
    assert!(n >= k, "n={} < k={}", n, k);

    let mut best_y = Array2::<f32>::zeros((z_cos.nrows(), k));
    let mut best_inertia = f32::INFINITY;

    let mut rng = SplitMix64::new(seed);

    for start in 0..n_starts {
        let local_seed = rng.next_u64();
        let mut sub = SplitMix64::new(local_seed);
        let centroids = kmeans_pp_seed(z_cos, k, &mut sub);
        let (y, inertia) = lloyd_iterations(z_cos, centroids, n_iter);
        if inertia < best_inertia {
            best_inertia = inertia;
            best_y = y;
        }
        // Prevent unused_variable warning when n_starts == 0 in a weird config.
        let _ = start;
    }

    normalize_columns_l2(&mut best_y);
    best_y
}

fn kmeans_pp_seed(z_cos: ArrayView2<f32>, k: usize, rng: &mut SplitMix64) -> Array2<f32> {
    let (d, n) = (z_cos.nrows(), z_cos.ncols());
    let mut centroids = Array2::<f32>::zeros((d, k));

    // Pick first center uniformly at random.
    let first = (rng.next_u64() as usize) % n;
    centroids.column_mut(0).assign(&z_cos.column(first));

    // Pick subsequent centers weighted by squared cosine distance.
    let mut closest_dist2 = Array1::<f32>::from_elem(n, f32::INFINITY);
    update_closest_dist2(z_cos, &centroids.column(0), &mut closest_dist2);

    for c in 1..k {
        // Weighted sampling: pick index with probability proportional to
        // closest_dist2. Use a cumulative-sum scan for correctness.
        let total: f32 = closest_dist2.iter().sum();
        let pick: f32 = (rng.next_f32()) * total.max(f32::EPSILON);
        let mut running = 0.0_f32;
        let mut chosen = n - 1;
        for (i, &d2) in closest_dist2.iter().enumerate() {
            running += d2;
            if running >= pick {
                chosen = i;
                break;
            }
        }
        centroids.column_mut(c).assign(&z_cos.column(chosen));
        if c + 1 < k {
            update_closest_dist2(z_cos, &centroids.column(c), &mut closest_dist2);
        }
    }
    centroids
}

fn update_closest_dist2(
    z: ArrayView2<f32>,
    new_center: &ndarray::ArrayView1<f32>,
    closest_dist2: &mut Array1<f32>,
) {
    let d = z.nrows();
    // Parallelize per-cell — each cell's distance + min-update is independent.
    closest_dist2
        .as_slice_mut()
        .expect("closest_dist2 contiguous")
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            let col = z.column(i);
            let mut s = 0.0_f32;
            for di in 0..d {
                let diff = col[di] - new_center[di];
                s += diff * diff;
            }
            if s < *slot {
                *slot = s;
            }
        });
}

fn lloyd_iterations(
    z_cos: ArrayView2<f32>,
    mut centroids: Array2<f32>,
    n_iter: usize,
) -> (Array2<f32>, f32) {
    let (d, n) = (z_cos.nrows(), z_cos.ncols());
    let k = centroids.ncols();
    let mut assignments = vec![0usize; n];
    let mut inertia = f32::INFINITY;

    for _ in 0..n_iter {
        // --- Assignment step --------------------------------------------
        // For cosine-normalized inputs, Euclidean d² = 2 − 2·<u, v>, so
        // argmin_c d² = argmax_c <centroid_c, z_i>. Compute all N·K dot
        // products as a single matmul and argmax per column in parallel.
        //
        // centroids: (d, K) → centroids.t(): (K, d)
        // z_cos:     (d, N)
        // sim:       (K, N) = centroids.t() · z_cos
        let sim = matmul_kdn_par(centroids.t(), z_cos);

        // Per-column argmax + accumulate inertia in parallel.
        let results: Vec<(usize, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut best_c = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for c in 0..k {
                    let s = sim[[c, i]];
                    if s > best_sim {
                        best_sim = s;
                        best_c = c;
                    }
                }
                // Convert back to Euclidean d² for inertia.
                // d² = 2 - 2·sim; clamp ≥ 0 for numerical noise.
                let dist2 = (2.0 - 2.0 * best_sim).max(0.0);
                (best_c, dist2)
            })
            .collect();

        let mut new_inertia = 0.0_f32;
        for (i, &(c, d2)) in results.iter().enumerate() {
            assignments[i] = c;
            new_inertia += d2;
        }

        // --- Centroid update ---------------------------------------------
        // Per-thread local (new_centroids, counts) accumulators, then
        // reduce-add. Each cell contributes one scatter write, so the
        // serial version was memory-bound — with rayon fold we turn it
        // into fan-in aggregation.
        let (new_centroids_sum, counts) = z_cos
            .axis_iter(Axis(1))
            .into_par_iter()
            .enumerate()
            .fold(
                || (Array2::<f32>::zeros((d, k)), vec![0u32; k]),
                |(mut cent_acc, mut count_acc), (i, col)| {
                    let c = assignments[i];
                    for di in 0..d {
                        cent_acc[[di, c]] += col[di];
                    }
                    count_acc[c] += 1;
                    (cent_acc, count_acc)
                },
            )
            .reduce(
                || (Array2::<f32>::zeros((d, k)), vec![0u32; k]),
                |(mut a_cent, mut a_count), (b_cent, b_count)| {
                    a_cent += &b_cent;
                    for c in 0..k {
                        a_count[c] += b_count[c];
                    }
                    (a_cent, a_count)
                },
            );

        let mut new_centroids = new_centroids_sum;
        for c in 0..k {
            if counts[c] > 0 {
                let denom = counts[c] as f32;
                for di in 0..d {
                    new_centroids[[di, c]] /= denom;
                }
            } else {
                // Empty cluster — keep previous centroid to avoid NaN.
                new_centroids.column_mut(c).assign(&centroids.column(c));
            }
        }

        centroids = new_centroids;
        if (inertia - new_inertia).abs() / inertia.max(f32::EPSILON) < 1e-4 {
            inertia = new_inertia;
            break;
        }
        inertia = new_inertia;
    }
    (centroids, inertia)
}

/// Tiny splitmix64 — good enough for seeding our k-means without pulling
/// in `rand`. Harmony's R port uses R's Mersenne Twister, so we're
/// already non-bit-identical; determinism + uniformity is sufficient.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9E3779B97F4A7C15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64() >> 40;
        (u as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Shuffle a `[0, n)` index vector (Fisher-Yates).
    pub fn shuffle_range(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            v.swap(i, j);
        }
        v
    }
}
