//! Small-matrix solves used by the MoE ridge correction.
#![allow(clippy::needless_range_loop)]
//!
//! The only linear system Harmony needs to solve is
//! `(Phi_Rk · Phi_moe.t + diag(lambda)) · x = b`, where the matrix is
//! `(B+1) × (B+1)`. For the typical `B ∈ [1, 10]` that appears in scRNA
//! integration this fits easily in a hand-rolled LU with partial
//! pivoting.

use ndarray::{Array2, ArrayView2};

/// Invert an `n × n` matrix via LU decomposition with partial pivoting.
///
/// Returns the inverse. Panics if the matrix is singular (no fallback —
/// the ridge `+ λI` added by the caller guarantees non-singularity at
/// positive `λ`).
pub fn invert_small(a: ArrayView2<f32>) -> Array2<f32> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n, "matrix must be square");

    // Build augmented [A | I] row-major, then Gauss-Jordan reduce.
    let mut aug = Array2::<f32>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for k in 0..n {
        // Partial pivot: find row with largest |aug[*, k]| at or below k.
        let mut max_row = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            let v = aug[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-12 {
            panic!(
                "invert_small: matrix is singular at column {} (pivot={})",
                k, max_val
            );
        }
        if max_row != k {
            // Swap rows.
            for j in 0..(2 * n) {
                aug.swap([k, j], [max_row, j]);
            }
        }
        // Normalize pivot row.
        let piv = aug[[k, k]];
        for j in 0..(2 * n) {
            aug[[k, j]] /= piv;
        }
        // Eliminate all other rows.
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = aug[[i, k]];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * n) {
                let v = aug[[i, j]] - factor * aug[[k, j]];
                aug[[i, j]] = v;
            }
        }
    }

    // Extract right half as the inverse.
    let mut inv = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn inverts_identity() {
        let i3 = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_small(i3.view());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn inverts_2x2() {
        // Matrix [[4, 7], [2, 6]] → det = 10, inv = [[0.6, -0.7], [-0.2, 0.4]]
        let m = array![[4.0_f32, 7.0], [2.0, 6.0]];
        let inv = invert_small(m.view());
        let expected = array![[0.6_f32, -0.7], [-0.2, 0.4]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (inv[[i, j]] - expected[[i, j]]).abs() < 1e-5,
                    "inv[{},{}] = {}",
                    i,
                    j,
                    inv[[i, j]]
                );
            }
        }
    }

    #[test]
    fn multiply_back_is_identity() {
        let m = array![[2.0_f32, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let inv = invert_small(m.view());
        // Multiply m × inv, check identity.
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0_f32;
                for k in 0..3 {
                    s += m[[i, k]] * inv[[k, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((s - expected).abs() < 1e-4, "product[{},{}] = {}", i, j, s);
            }
        }
    }
}
