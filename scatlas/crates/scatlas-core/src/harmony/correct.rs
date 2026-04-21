//! MoE ridge correction (`moe_correct_ridge_cpp`).
//!
//! Lint notes — several loops deliberately index multiple parallel arrays
//! by (row, col), so `needless_range_loop` is silenced at module level.
#![allow(clippy::needless_range_loop)]
//!
//! Per cluster k, the C++ code builds a (B+1, N) design matrix
//! `Phi_moe = [1; Phi]` (intercept + one-hot batches) and solves
//!
//! ```text
//! W_k = inv(Phi_moe · diag(R[k]) · Phi_moe.t  +  diag(lambda_k)) ·
//!       Phi_moe · diag(R[k]) · Z_orig.t
//! ```
//!
//! `W_k` is (B+1, d). The intercept row is zeroed out (we never subtract
//! the cluster-mean itself — only per-batch offsets). Because Phi_moe is
//! structured (row 0 all-ones, rows 1..=B are mutually exclusive
//! one-hots), `A_k = Phi_moe·diag(R[k])·Phi_moe.t` is a thin cross:
//!
//! ```text
//! A_k[0, 0]        = sum_i R[k, i]
//! A_k[0, b+1]      = sum_{i∈batch b} R[k, i]   (= O[k, b])
//! A_k[b+1, 0]      = O[k, b]                   (symmetric)
//! A_k[b+1, b'+1]   = O[k, b]  if b == b' else 0
//! ```
//!
//! Same structure exploited for RHS — row 0 sums over ALL cells,
//! rows 1..=B sum per-batch. These decompose cleanly into `B+1`
//! contiguous weighted aggregations of `Z_orig`.
//!
//! **M4.2 parallelization**: per-cluster W solves are embarrassingly
//! parallel (each only reads shared state). We compute all `K` W-slabs
//! via `par_iter`, then apply corrections with rayon over cells.

use ndarray::{Array1, Array2, Array3, Axis};
use rayon::prelude::*;

use super::linalg::invert_small;
use super::state::{normalize_columns_l2, HarmonyState, LambdaMode};

pub fn moe_correct_ridge(state: &mut HarmonyState) {
    state.z_corr.assign(&state.z_orig);

    let b_plus_1 = state.n_batches + 1;

    // Precompute per-batch cell index lists (used by all clusters).
    let mut batch_cells: Vec<Vec<usize>> = vec![Vec::new(); state.n_batches];
    for (i, &b) in state.batch_codes.iter().enumerate() {
        batch_cells[b as usize].push(i);
    }

    // ----------------------- Phase 1 (parallel-over-K) ----------------------
    // Compute W_k for every cluster in parallel. Each closure reads shared
    // state (R, O, E, Z_orig, lambda) read-only and produces a small
    // (B, d) slab of the ridge coefficients — the intercept row is
    // zeroed out anyway so we skip it.
    //
    // We borrow the big matrices up front (outside the par_iter) so the
    // rayon threads share references without re-borrowing state.
    let r = &state.r;
    let o = &state.o;
    let e = &state.e;
    let z_orig = &state.z_orig;
    let lambda_fixed = &state.lambda;
    let batch_codes = &state.batch_codes;
    let n_dims = state.n_dims;
    let n_batches = state.n_batches;
    let n_cells = state.n_cells;
    let lambda_mode = state.lambda_mode;
    let alpha = state.alpha;
    let batch_cells_ref = &batch_cells;

    let w_batch_slabs: Vec<Array2<f32>> = (0..state.n_clusters)
        .into_par_iter()
        .map(|k| {
            compute_w_batch_for_cluster(
                k,
                n_cells,
                n_batches,
                n_dims,
                b_plus_1,
                r,
                o,
                z_orig,
                lambda_fixed,
                lambda_mode,
                alpha,
                e,
                batch_codes,
                batch_cells_ref,
            )
        })
        .collect();

    // ---------------- Phase 2: stack → (K, B, d) ----------------
    // For fast per-cell access we want a (B, K, d) layout: per batch b,
    // a (K, d) slab that dots with R[:, i] for cells in batch b.
    let mut w_stack = Array3::<f32>::zeros((n_batches, state.n_clusters, n_dims));
    for (k, slab) in w_batch_slabs.iter().enumerate() {
        for b in 0..n_batches {
            for d in 0..n_dims {
                w_stack[[b, k, d]] = slab[[b, d]];
            }
        }
    }

    // ----------------- Phase 3: parallel apply over cells -----------------
    // For cell i in batch b: delta[d] = Σ_k R[k, i] · W_stack[b, k, d].
    // Cells write independent columns of Z_corr → safe to parallelize.
    state
        .z_corr
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut zcol)| {
            let b = batch_codes[i] as usize;
            let wmat = w_stack.index_axis(Axis(0), b); // (K, d)
            for d in 0..n_dims {
                let mut s = 0.0_f32;
                for k in 0..state.n_clusters {
                    s += r[[k, i]] * wmat[[k, d]];
                }
                zcol[d] -= s;
            }
        });

    // Re-normalize Z_cos for the next clustering pass.
    let mut z_cos_new = state.z_corr.clone();
    normalize_columns_l2(&mut z_cos_new);
    state.z_cos = z_cos_new;
}

#[allow(clippy::too_many_arguments)]
fn compute_w_batch_for_cluster(
    k: usize,
    _n_cells: usize,
    n_batches: usize,
    n_dims: usize,
    b_plus_1: usize,
    r: &Array2<f32>,
    o: &Array2<f32>,
    z_orig: &Array2<f32>,
    lambda_fixed: &Array1<f32>,
    lambda_mode: LambdaMode,
    alpha: f32,
    e: &Array2<f32>,
    _batch_codes: &Array1<i32>,
    batch_cells: &[Vec<usize>],
) -> Array2<f32> {
    let r_row = r.row(k); // (N,)
    let total_rk: f32 = r_row.iter().sum();

    // --- A_k = Phi_moe · diag(R[k]) · Phi_moe.t + diag(lambda)
    let mut a = Array2::<f32>::zeros((b_plus_1, b_plus_1));

    // Resolve lambda for this cluster.
    let lambda_k: Array1<f32> = match lambda_mode {
        LambdaMode::Fixed => lambda_fixed.clone(),
        LambdaMode::Dynamic => {
            let mut v = Array1::zeros(b_plus_1);
            for b in 0..n_batches {
                v[b + 1] = alpha * e[[k, b]];
            }
            v
        }
    };

    a[[0, 0]] = total_rk + lambda_k[0];
    for b in 0..n_batches {
        let o_kb = o[[k, b]];
        a[[0, b + 1]] = o_kb;
        a[[b + 1, 0]] = o_kb;
        a[[b + 1, b + 1]] = o_kb + lambda_k[b + 1];
    }

    // --- RHS (B+1, d)
    let mut rhs = Array2::<f32>::zeros((b_plus_1, n_dims));

    // Row 0: sum over all cells
    for i in 0..r_row.len() {
        let r_ki = r_row[i];
        if r_ki == 0.0 {
            continue;
        }
        for d in 0..n_dims {
            rhs[[0, d]] += r_ki * z_orig[[d, i]];
        }
    }

    // Rows 1..=B: sum per batch
    for b in 0..n_batches {
        let cells = &batch_cells[b];
        for &i in cells {
            let r_ki = r_row[i];
            if r_ki == 0.0 {
                continue;
            }
            for d in 0..n_dims {
                rhs[[b + 1, d]] += r_ki * z_orig[[d, i]];
            }
        }
    }

    // W_k = inv(A) · RHS
    let a_inv = invert_small(a.view());
    let w = a_inv.dot(&rhs);

    // Strip the intercept row, keep (B, d) slab of batch betas
    let mut w_batch = Array2::<f32>::zeros((n_batches, n_dims));
    for b in 0..n_batches {
        for d in 0..n_dims {
            w_batch[[b, d]] = w[[b + 1, d]];
        }
    }
    w_batch
}

/// Silence unused imports until full test coverage lands.
#[allow(dead_code)]
fn _unused_array1(_: &Array1<f32>) {}
