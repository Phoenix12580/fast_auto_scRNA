//! Soft k-means clustering with Harmony's diversity penalty.
//!
//! Mirrors `cluster_cpp` / `update_R` / `compute_objective` /
//! `check_convergence` in `harmony.cpp`, with the shape conventions:
//!   R:      (K, N)
//!   Y:      (d, K)
//!   Z_cos:  (d, N)
//!   O, E:   (K, B)
//!   Phi:    (B, N) [implicit — we track batch codes as (N,) i32 and
//!           compute `R * Phi.t()` by accumulating into (K, B)]

#![allow(clippy::needless_range_loop)]

use ndarray::{Array2, Axis};
use rayon::prelude::*;

use super::kmeans_init::{kmeans_centers, SplitMix64};
use super::matmul::{matmul_dnk_par, matmul_kdn_par};
use super::state::{normalize_columns_l2, HarmonyState};

/// `init_cluster_cpp` analogue. After this call:
///   Y   — K centroids, L2-normalized
///   R   — softmax over K, column-normalized to 1
///   O, E — cluster-batch count matrices
pub fn init_cluster(s: &mut HarmonyState) {
    // Y = kmeans_centers(Z_cos, K); normalized to unit length already.
    s.y = kmeans_centers(s.z_cos.view(), s.n_clusters, 5, 25, s.seed);

    // dist_mat = 2 * (1 - Y.t() @ Z_cos)
    compute_dist_mat(s);

    // R = softmax over clusters: R[k, i] = exp(-dist_mat[k, i] / sigma[k])
    softmax_over_clusters(&mut s.r, &s.dist_mat, &s.sigma);

    // E = sum(R, 1) * Pr_b.t()   (K, B)
    // O = R * Phi.t()            (K, B)  — Phi is one-hot batches
    recompute_o_e_from_scratch(s);

    compute_objective(s);
    // Mirror C++: objective_harmony starts at objective_kmeans.back()
    s.objective_harmony
        .push(*s.objective_kmeans.last().unwrap());
}

/// `cluster_cpp` analogue — inner loop of soft k-means with blocked
/// randomized update_R. Runs until `epsilon_cluster` or
/// `max_iter_cluster`, whichever comes first.
pub fn cluster(state: &mut HarmonyState) {
    let profile = std::env::var("SCATLAS_HARMONY_PROFILE_INNER").is_ok();
    let mut t_y = std::time::Duration::ZERO;
    let mut t_dist = std::time::Duration::ZERO;
    let mut t_r = std::time::Duration::ZERO;
    let mut t_obj = std::time::Duration::ZERO;

    let mut iter: usize = 0;
    loop {
        let t = std::time::Instant::now();
        state.y = matmul_dnk_par(state.z_cos.view(), state.r.t());
        normalize_columns_l2(&mut state.y);
        t_y += t.elapsed();

        let t = std::time::Instant::now();
        compute_dist_mat(state);
        t_dist += t.elapsed();

        let t = std::time::Instant::now();
        update_r(state);
        t_r += t.elapsed();

        let t = std::time::Instant::now();
        compute_objective(state);
        t_obj += t.elapsed();

        if iter > state.window_size && check_convergence_cluster(state) {
            iter += 1;
            break;
        }
        iter += 1;
        if iter >= state.max_iter_cluster {
            break;
        }
    }
    state.kmeans_rounds.push(iter);
    state
        .objective_harmony
        .push(*state.objective_kmeans.last().unwrap());
    if profile {
        eprintln!(
            "  [cluster] iters={}, Y={:.2?}, dist={:.2?}, update_R={:.2?}, obj={:.2?}",
            iter, t_y, t_dist, t_r, t_obj
        );
    }
}

/// `compute_objective` analogue. Pushes four diagnostic values onto
/// `objective_kmeans*` — only the aggregate one is used by convergence.
pub fn compute_objective(s: &mut HarmonyState) {
    let norm_const = 2000.0_f32 / (s.n_cells as f32);

    // kmeans_error = sum(R % dist_mat) — elementwise product then sum.
    // Parallelize per-row (K threads) — reduce cost from 15.7M serial
    // multiplies to ~K chunks of N each.
    let kmeans_error: f32 =
        s.r.axis_iter(Axis(0))
            .into_par_iter()
            .zip(s.dist_mat.axis_iter(Axis(0)).into_par_iter())
            .map(|(r_row, d_row)| {
                r_row
                    .iter()
                    .zip(d_row.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
            })
            .sum();

    // entropy = sum_k sigma[k] * sum_i R[k,i] * log(R[k,i])
    // (fused — no separate ent_mat allocation; safe_entropy handles 0).
    let sigma_ref = &s.sigma;
    let _entropy: f32 =
        s.r.axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .map(|(k, row)| {
                let s_k = sigma_ref[k];
                let row_ent: f32 = row
                    .iter()
                    .map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 })
                    .sum();
                row_ent * s_k
            })
            .sum();

    // cross_entropy = sum( (R.each_col() * sigma) *
    //                     ((repmat(theta.t(), K, 1) * log((O + E)/E)) * Phi) )
    // Building (K, N) piece by piece: for each (k, b, i-in-batch-b), the
    // contribution is sigma[k] * theta[b] * log((O[k,b]+E[k,b])/E[k,b]) *
    // R[k, i]. Sum over (k, i) of these pieces.
    let mut _cross = 0.0_f32;
    for b in 0..s.n_batches {
        let th_b = s.theta[b];
        for k in 0..s.n_clusters {
            let e_kb = s.e[[k, b]];
            let o_kb = s.o[[k, b]];
            if e_kb <= 0.0 {
                continue;
            }
            let log_ratio = ((o_kb + e_kb) / e_kb).ln();
            let coef = s.sigma[k] * th_b * log_ratio;
            // Sum of R[k, i] over cells in batch b — we have it as O[k, b]!
            // Because O = R * Phi.t() → O[k, b] = sum_i R[k, i] * 1[batch_i = b].
            _cross += coef * o_kb;
        }
    }

    let total = (kmeans_error + _entropy + _cross) * norm_const;
    s.objective_kmeans.push(total);
    s.objective_kmeans_dist.push(kmeans_error * norm_const);
    s.objective_kmeans_entropy.push(_entropy * norm_const);
    s.objective_kmeans_cross.push(_cross * norm_const);
}

/// `check_convergence(type=0)` — windowed average of the last
/// `window_size` objective values vs the prior window.
pub fn check_convergence_cluster(s: &HarmonyState) -> bool {
    let n = s.objective_kmeans.len();
    if n < 2 * s.window_size {
        return false;
    }
    let mut obj_old = 0.0_f32;
    let mut obj_new = 0.0_f32;
    for i in 0..s.window_size {
        obj_old += s.objective_kmeans[n - 2 - i];
        obj_new += s.objective_kmeans[n - 1 - i];
    }
    (obj_old - obj_new) / obj_old.abs() < s.epsilon_cluster
}

/// `update_R` — blocked randomized update with Harmony's diversity
/// penalty. This is the core 2.0 innovation.
///
/// **M4.3 fused-pass rewrite**: the original 3-phase per-block flow
/// (remove old R from O/E → recompute R → add new R to O/E) read each
/// cell's R-column three times. We fuse them into a single parallel
/// sweep that reads old R[:, i], computes new R[:, i], writes it back,
/// and accumulates the (new − old) delta into per-thread local O and E
/// scratch, then reduces once per block.
pub fn update_r(state: &mut HarmonyState) {
    let profile = std::env::var("SCATLAS_HARMONY_PROFILE_UPDATE_R").is_ok();
    let t_sm = std::time::Instant::now();
    // _scale_dist = softmax_over_clusters(-dist_mat / sigma)  (K, N)
    let mut scale_dist = Array2::<f32>::zeros(state.dist_mat.dim());
    softmax_over_clusters(&mut scale_dist, &state.dist_mat, &state.sigma);
    let t_softmax = t_sm.elapsed();

    // Shuffled cell order
    let mut rng = SplitMix64::new(state.seed ^ 0x5DEECE66Du64);
    let update_order = rng.shuffle_range(state.n_cells);

    let n_blocks = (1.0 / state.block_size).ceil() as usize;
    let cells_per_block = (state.n_cells as f32 * state.block_size) as usize;
    let cells_per_block = cells_per_block.max(1);

    let mut t_coef = std::time::Duration::ZERO;
    let mut t_fused = std::time::Duration::ZERO;
    let mut t_apply = std::time::Duration::ZERO;

    for b_idx in 0..n_blocks {
        let idx_min = b_idx * cells_per_block;
        let idx_max = if b_idx == n_blocks - 1 {
            state.n_cells
        } else {
            ((b_idx + 1) * cells_per_block).min(state.n_cells)
        };
        if idx_min >= state.n_cells {
            break;
        }
        let cell_ids: Vec<usize> = (idx_min..idx_max).map(|i| update_order[i]).collect();

        // ------ Step 1: recompute per-(k, batch) diversity coefficient.
        let t = std::time::Instant::now();
        let mut coef: Array2<f32> = Array2::zeros((state.n_clusters, state.n_batches));
        for k in 0..state.n_clusters {
            for b in 0..state.n_batches {
                let denom = state.o[[k, b]] + state.e[[k, b]];
                if denom <= 0.0 || state.e[[k, b]] <= 0.0 {
                    coef[[k, b]] = 1.0;
                } else {
                    let ratio = state.e[[k, b]] / denom;
                    coef[[k, b]] = ratio.powf(state.theta[b]);
                }
            }
        }
        t_coef += t.elapsed();

        // ------ Step 2: fused parallel sweep per cell.
        // For cell i (unique within the block), do:
        //    old_r[k] := R[k, i]  (strided read)
        //    new_r[k] := softmax(scale_dist[k, i] * coef[k, batch])
        //    R[k, i]  := new_r[k]  (strided write)
        //    accumulate per-thread: o_delta[k, batch] += new_r - old_r
        //                           rs_delta[k]       += new_r - old_r
        // One traversal of R's columns, one reduction per block.
        let coef_ref = &coef;
        let scale_dist_ref = &scale_dist;
        let batch_codes_ref = &state.batch_codes;
        let n_clusters = state.n_clusters;
        let n_batches = state.n_batches;
        let n_cells = state.n_cells;
        let r_ptr_addr: usize = state.r.as_mut_ptr() as usize;

        let t = std::time::Instant::now();
        let (o_delta, rs_delta) = cell_ids
            .par_iter()
            .fold(
                || {
                    (
                        Array2::<f32>::zeros((n_clusters, n_batches)),
                        vec![0.0_f32; n_clusters],
                    )
                },
                |(mut o_acc, mut rs_acc), &i| {
                    let batch = batch_codes_ref[i] as usize;
                    let r_ptr = r_ptr_addr as *mut f32;
                    // First pass: compute col_sum (softmax normalization).
                    let mut col_sum = 0.0_f32;
                    for k in 0..n_clusters {
                        col_sum += scale_dist_ref[[k, i]] * coef_ref[[k, batch]];
                    }
                    let inv = if col_sum > 0.0 { 1.0 / col_sum } else { 0.0 };
                    let fallback = 1.0 / n_clusters as f32;
                    for k in 0..n_clusters {
                        // SAFETY: distinct cells → distinct strided slots.
                        let slot = unsafe { r_ptr.add(k * n_cells + i) };
                        let old_r = unsafe { *slot };
                        let new_r = if inv > 0.0 {
                            scale_dist_ref[[k, i]] * coef_ref[[k, batch]] * inv
                        } else {
                            fallback
                        };
                        unsafe { *slot = new_r };
                        let delta = new_r - old_r;
                        o_acc[[k, batch]] += delta;
                        rs_acc[k] += delta;
                    }
                    (o_acc, rs_acc)
                },
            )
            .reduce(
                || {
                    (
                        Array2::<f32>::zeros((n_clusters, n_batches)),
                        vec![0.0_f32; n_clusters],
                    )
                },
                |(mut a_o, mut a_rs), (b_o, b_rs)| {
                    a_o += &b_o;
                    for k in 0..n_clusters {
                        a_rs[k] += b_rs[k];
                    }
                    (a_o, a_rs)
                },
            );
        t_fused += t.elapsed();

        // ------ Step 3: apply per-block deltas to O and E, clamp ≥ 0.
        let t = std::time::Instant::now();
        state.o += &o_delta;
        // E[k, b] += rs_delta[k] * pr_b[b]
        for k in 0..state.n_clusters {
            let rs = rs_delta[k];
            for b in 0..state.n_batches {
                state.e[[k, b]] += rs * state.pr_b[b];
            }
        }
        // Numerical noise safety clamp (only matters when a row-sum delta
        // net-negates everything in O or E, which shouldn't happen but
        // does show up in floats over many cluster iterations).
        state.o.mapv_inplace(|v| v.max(0.0));
        state.e.mapv_inplace(|v| v.max(0.0));
        t_apply += t.elapsed();
    }
    if profile {
        eprintln!(
            "    [update_R] softmax={:.2?}, coef={:.2?}, fused_sweep={:.2?}, apply_delta={:.2?}",
            t_softmax, t_coef, t_fused, t_apply
        );
    }
}

// --- Helpers -----------------------------------------------------------------

fn compute_dist_mat(s: &mut HarmonyState) {
    // dist_mat = 2 * (1 - Y.t() · Z_cos)
    // Y.t() is (K, d), Z_cos is (d, N). Output (K, N) is fat —
    // split N into column slabs across rayon threads.
    let mut dot = matmul_kdn_par(s.y.t(), s.z_cos.view());
    dot.par_mapv_inplace(|v| 2.0 - 2.0 * v);
    s.dist_mat = dot;
}

fn softmax_over_clusters(
    out: &mut Array2<f32>,
    dist_mat: &Array2<f32>,
    sigma: &ndarray::Array1<f32>,
) {
    // Iterate cells (axis=1) in parallel — each cell's softmax is a
    // K-element read+write with no cross-cell dependency.
    let k_rows = dist_mat.nrows();
    out.axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(dist_mat.axis_iter(Axis(1)).into_par_iter())
        .for_each(|(mut out_col, dist_col)| {
            let mut max_logit = f32::NEG_INFINITY;
            for k in 0..k_rows {
                let logit = -dist_col[k] / sigma[k];
                if logit > max_logit {
                    max_logit = logit;
                }
            }
            let mut sum = 0.0_f32;
            for k in 0..k_rows {
                let logit = -dist_col[k] / sigma[k];
                let v = (logit - max_logit).exp();
                out_col[k] = v;
                sum += v;
            }
            if sum > 0.0 {
                for k in 0..k_rows {
                    out_col[k] /= sum;
                }
            }
        });
}

fn recompute_o_e_from_scratch(state: &mut HarmonyState) {
    state.o.fill(0.0);
    state.e.fill(0.0);
    // E[k, b] = sum_i R[k, i] * Pr_b[b]  →  sum_i R[k, i]  times Pr_b[b]
    let row_sums = state.r.sum_axis(Axis(1)); // (K,)
    for k in 0..state.n_clusters {
        for b in 0..state.n_batches {
            state.e[[k, b]] = row_sums[k] * state.pr_b[b];
        }
    }
    // O[k, b] = sum_{i : batch_i = b} R[k, i]
    for i in 0..state.n_cells {
        let b = state.batch_codes[i] as usize;
        for k in 0..state.n_clusters {
            state.o[[k, b]] += state.r[[k, i]];
        }
    }
}

// Note: `remove_cells_from_o_e` / `add_cells_to_o_e` were the original
// 3-phase per-block O/E updaters. The fused sweep in `update_r` replaces
// both with a single delta-accumulation pass.
