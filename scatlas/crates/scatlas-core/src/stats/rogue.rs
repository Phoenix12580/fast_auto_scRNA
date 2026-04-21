//! ROGUE entropy kernels — hot inner loops for `CalculateRogue`.
//!
//! Ported from `scvalidate.rogue_py.core.entropy_table` (which itself is a
//! port of PaulingLiu/ROGUE's `Entropy` R function, rogue_ROGUE.R L11–L23).
//!
//! Scope is deliberately narrow: only the per-gene summary statistics that
//! feed `entropy_fit`. The loess-based fit, outlier pruning, and the
//! `rogue_per_cluster` orchestrator stay in Python — no pure-Rust loess in
//! the ecosystem matches R's C loess bit-for-bit, and the 3-pass trim-refit
//! is not a performance hotspot (scvalidate's 157k × 50 cluster benchmark
//! spends most of its 349 s in loess passes, not entropy computation, so
//! Rust-izing this one kernel already captures everything worth capturing
//! from scatlas-core's side).

use ndarray::ArrayView2;
use rayon::prelude::*;

/// Per-gene `(log(mean(counts) + r), mean(log(counts + 1)))`.
///
/// Returns a contiguous vector of length `2 * n_genes`, laid out as
/// `[mean_expr_0, entropy_0, mean_expr_1, entropy_1, ...]`. The flat layout
/// lets the PyO3 wrapper reshape into a `(n_genes, 2)` numpy array without
/// an extra copy.
///
/// `expr` must be genes × cells of raw counts (non-log). `r` is the pseudo-
/// count added before `log(mean + r)`; R's default is `1.0`.
pub fn entropy_table_dense<T>(expr: ArrayView2<T>, r: f64) -> Vec<f64>
where
    T: Into<f64> + Copy + Sync + Send,
{
    let (n_genes, n_cells) = (expr.nrows(), expr.ncols());
    let inv_n = if n_cells == 0 {
        0.0
    } else {
        1.0 / n_cells as f64
    };

    let mut out = vec![0.0_f64; 2 * n_genes];
    out.par_chunks_mut(2).enumerate().for_each(|(g, slot)| {
        let row = expr.row(g);
        let mut sum = 0.0_f64;
        let mut sum_log1p = 0.0_f64;
        for &v in row.iter() {
            let x: f64 = v.into();
            sum += x;
            sum_log1p += libm::log1p(x);
        }
        let mean = sum * inv_n;
        slot[0] = (mean + r).ln();
        slot[1] = sum_log1p * inv_n;
    });
    out
}

/// Per-gene entropy for a CSR sparse matrix of shape `n_genes × n_cells`.
///
/// `indptr.len() == n_genes + 1`, `indices.len() == data.len() == nnz`. The
/// CSR layout lets us skip the zero entries: `log1p(0) == 0` and `0` adds
/// nothing to the sum, so only `data` contributes. `mean` still divides by
/// full `n_cells`.
pub fn entropy_table_csr<T>(indptr: &[usize], data: &[T], n_cells: usize, r: f64) -> Vec<f64>
where
    T: Into<f64> + Copy + Sync + Send,
{
    assert!(!indptr.is_empty(), "indptr must have at least one element");
    let n_genes = indptr.len() - 1;
    let inv_n = if n_cells == 0 {
        0.0
    } else {
        1.0 / n_cells as f64
    };

    let mut out = vec![0.0_f64; 2 * n_genes];
    out.par_chunks_mut(2).enumerate().for_each(|(g, slot)| {
        let start = indptr[g];
        let end = indptr[g + 1];
        let mut sum = 0.0_f64;
        let mut sum_log1p = 0.0_f64;
        for &v in &data[start..end] {
            let x: f64 = v.into();
            sum += x;
            sum_log1p += libm::log1p(x);
        }
        let mean = sum * inv_n;
        slot[0] = (mean + r).ln();
        slot[1] = sum_log1p * inv_n;
    });
    out
}

/// `1 - sig_value / (sig_value + k)`, where
/// `sig_value = sum(|ds_i|)` over genes passing `p_adj < cutoff &&
/// p_value < cutoff`.
///
/// R source: `CalculateRogue`, rogue_ROGUE.R L245–L271. `k` is the
/// platform-dependent normalization constant (45 for UMI, 500 for
/// full-length); `scatlas.stats.rogue` supplies the default in the Python
/// wrapper.
pub fn calculate_rogue(ds: &[f64], p_adj: &[f64], p_value: &[f64], cutoff: f64, k: f64) -> f64 {
    assert_eq!(ds.len(), p_adj.len());
    assert_eq!(ds.len(), p_value.len());
    let mut sig = 0.0_f64;
    for i in 0..ds.len() {
        let pa = p_adj[i];
        let pv = p_value[i];
        if pa.is_finite() && pv.is_finite() && pa < cutoff && pv < cutoff {
            let d = ds[i];
            if d.is_finite() {
                sig += d.abs();
            }
        }
    }
    1.0 - sig / (sig + k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn dense_small_matches_python_formula() {
        // expr = [[1, 2, 3], [0, 0, 4]]  r = 1.0
        // row 0: mean=2.0, log(mean+1)=log(3), entropy = (log(2)+log(3)+log(4))/3
        // row 1: mean=4/3,  log(mean+1)=log(7/3), entropy = (0+0+log(5))/3
        let x: Array2<f64> = array![[1.0, 2.0, 3.0], [0.0, 0.0, 4.0]];
        let out = entropy_table_dense(x.view(), 1.0);
        assert_eq!(out.len(), 4);

        let expect_me0 = 3.0_f64.ln();
        let expect_ent0 = (2.0_f64.ln() + 3.0_f64.ln() + 4.0_f64.ln()) / 3.0;
        let expect_me1 = (7.0_f64 / 3.0).ln();
        let expect_ent1 = 5.0_f64.ln() / 3.0;

        assert!((out[0] - expect_me0).abs() < 1e-12, "mean0 = {}", out[0]);
        assert!((out[1] - expect_ent0).abs() < 1e-12, "ent0  = {}", out[1]);
        assert!((out[2] - expect_me1).abs() < 1e-12, "mean1 = {}", out[2]);
        assert!((out[3] - expect_ent1).abs() < 1e-12, "ent1  = {}", out[3]);
    }

    #[test]
    fn csr_matches_dense() {
        // Same matrix as above in CSR.
        // row 0: cols 0,1,2  vals 1,2,3
        // row 1: col 2       val 4
        let indptr = vec![0usize, 3, 4];
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let csr_out = entropy_table_csr(&indptr, &data, 3, 1.0);

        let x: Array2<f64> = array![[1.0, 2.0, 3.0], [0.0, 0.0, 4.0]];
        let dense_out = entropy_table_dense(x.view(), 1.0);

        for (a, b) in csr_out.iter().zip(dense_out.iter()) {
            assert!((a - b).abs() < 1e-12, "csr={} dense={}", a, b);
        }
    }

    #[test]
    fn calculate_rogue_matches_python_semantics() {
        // 3 genes. Gene 0: p_adj=0.01, p_value=0.01, ds=2.0  → kept, |ds|=2
        // Gene 1: p_adj=0.5, p_value=0.01, ds=5.0            → filtered out
        // Gene 2: p_adj=0.01, p_value=0.04, ds=-3.0          → kept, |ds|=3
        // sig = 5, k = 45 → 1 - 5/50 = 0.9
        let ds = [2.0, 5.0, -3.0];
        let p_adj = [0.01, 0.5, 0.01];
        let p_value = [0.01, 0.01, 0.04];
        let r = calculate_rogue(&ds, &p_adj, &p_value, 0.05, 45.0);
        assert!((r - 0.9).abs() < 1e-12, "rogue = {}", r);
    }

    #[test]
    fn calculate_rogue_all_filtered_returns_one() {
        let ds = [1.0, 2.0];
        let p_adj = [0.5, 0.5];
        let p_value = [0.5, 0.5];
        let r = calculate_rogue(&ds, &p_adj, &p_value, 0.05, 45.0);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dense_f32_and_f64_agree() {
        let x64: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x32: Array2<f32> = x64.mapv(|v| v as f32);
        let a = entropy_table_dense(x64.view(), 1.0);
        let b = entropy_table_dense(x32.view(), 1.0);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "f64={} f32={}", x, y);
        }
    }
}
