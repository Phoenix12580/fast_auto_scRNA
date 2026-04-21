//! Wilcoxon rank-sum (Mann-Whitney U) p-values for every row of a matrix.
//!
//! Mirrors `scvalidate.recall_py.core._wilcoxon_per_gene`:
//!   - Per row, compute tie-averaged ranks across the union of two groups
//!     (`mask1 | mask2`), using `scipy.stats.rankdata(method="average")`
//!     semantics.
//!   - U = R1 - n1*(n1+1)/2, normal-approx two-sided p-value
//!     `p = 2 * (1 - Φ(|z|))` with **no ties correction** (matches the Python
//!     comment "ignore ties correction — good enough for FDR calibration").
//!   - Clip `p ∈ [1e-300, 1.0]`.
//!
//! Two entry points: `_wilcoxon_ranksum_matrix_f64` and `_f32`. The Python
//! wrapper in `scvalidate.recall_py.core` dispatches on `log_counts.dtype` so
//! callers can save memory by storing the 2G × n_cells log-normalized matrix
//! in f32 at 157k scale. f64 remains the canonical reference for parity tests.
//!
//! Parallelizes over matrix rows with rayon.

use ndarray::{s, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const P_FLOOR: f64 = 1e-300;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

#[inline]
fn two_sided_p(abs_z: f64) -> f64 {
    libm::erfc(abs_z / SQRT_2).clamp(P_FLOOR, 1.0)
}

/// Tie-averaged ranks of the selected cells of one row.  Generic over `T:
/// PartialOrd + Copy` so the same algorithm handles f32 and f64 inputs.
#[inline]
fn row_rank_sum_group1<T: PartialOrd + Copy>(
    values: &[T],
    group1_mask: &[bool],
) -> (usize, usize, f64) {
    let n = values.len();
    debug_assert_eq!(group1_mask.len(), n);

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        values[a as usize]
            .partial_cmp(&values[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks: Vec<f64> = vec![0.0; n];
    let mut i = 0usize;
    while i < n {
        let vi = values[order[i] as usize];
        let mut j = i + 1;
        while j < n && values[order[j] as usize] == vi {
            j += 1;
        }
        let avg = (i + j + 1) as f64 * 0.5;
        for k in i..j {
            ranks[order[k] as usize] = avg;
        }
        i = j;
    }

    let mut n1: usize = 0;
    let mut r1: f64 = 0.0;
    for k in 0..n {
        if group1_mask[k] {
            n1 += 1;
            r1 += ranks[k];
        }
    }
    (n1, n - n1, r1)
}

/// Core dispatch, generic over element dtype.  Returns the per-row p-values.
fn wilcoxon_core<T>(
    py: Python<'_>,
    x: ArrayView2<T>,
    m1: ArrayView1<bool>,
    m2: ArrayView1<bool>,
) -> PyResult<Vec<f64>>
where
    T: PartialOrd + Copy + Sync + Send,
{
    let (n_genes, n_cells) = (x.nrows(), x.ncols());
    if m1.len() != n_cells || m2.len() != n_cells {
        return Err(PyValueError::new_err(format!(
            "mask shapes must match number of cells: log_counts is ({}, {}), \
             mask1 has {}, mask2 has {}",
            n_genes,
            n_cells,
            m1.len(),
            m2.len(),
        )));
    }

    let mut keep_idx: Vec<usize> = Vec::with_capacity(n_cells);
    let mut is_group1: Vec<bool> = Vec::with_capacity(n_cells);
    for k in 0..n_cells {
        let in1 = unsafe { *m1.uget(k) };
        let in2 = unsafe { *m2.uget(k) };
        if in1 || in2 {
            keep_idx.push(k);
            is_group1.push(in1);
        }
    }

    let n_total = keep_idx.len();
    let n1_global: usize = is_group1.iter().filter(|&&b| b).count();
    let n2_global: usize = n_total - n1_global;

    if n1_global == 0 || n2_global == 0 {
        return Ok(vec![1.0_f64; n_genes]);
    }

    let mu_u = (n1_global as f64) * (n2_global as f64) * 0.5;
    let sigma_u =
        ((n1_global as f64) * (n2_global as f64) * (n_total as f64 + 1.0) / 12.0).sqrt();

    let out = py.allow_threads(|| -> Vec<f64> {
        let mut out = vec![0.0_f64; n_genes];
        out.par_iter_mut().enumerate().for_each(|(g, p)| {
            let row = x.slice(s![g, ..]);
            let mut vals: Vec<T> = Vec::with_capacity(n_total);
            for &k in keep_idx.iter() {
                vals.push(unsafe { *row.uget(k) });
            }
            let (n1, _n2, r1) = row_rank_sum_group1(&vals, &is_group1);
            debug_assert_eq!(n1, n1_global);
            let u1 = r1 - (n1 as f64) * (n1 as f64 + 1.0) * 0.5;
            let z = (u1 - mu_u) / sigma_u;
            *p = two_sided_p(z.abs());
        });
        out
    });
    Ok(out)
}

/// Dtype-dispatching wrapper: accepts f32 or f64 numpy arrays. f32 halves
/// the log_counts footprint on 157k (40 GB → 20 GB); f64 stays the canonical
/// reference for parity tests.
#[pyfunction]
pub fn wilcoxon_ranksum_matrix<'py>(
    py: Python<'py>,
    log_counts: &Bound<'py, PyAny>,
    mask1: PyReadonlyArray1<'py, bool>,
    mask2: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if let Ok(arr) = log_counts.extract::<PyReadonlyArray2<f64>>() {
        let out = wilcoxon_core(py, arr.as_array(), mask1.as_array(), mask2.as_array())?;
        return Ok(out.into_pyarray(py));
    }
    if let Ok(arr) = log_counts.extract::<PyReadonlyArray2<f32>>() {
        let out = wilcoxon_core(py, arr.as_array(), mask1.as_array(), mask2.as_array())?;
        return Ok(out.into_pyarray(py));
    }
    Err(PyValueError::new_err(
        "log_counts must be float32 or float64 2D numpy array",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_no_ties() {
        let vals = [10.0_f64, 30.0, 20.0];
        let mask = [true, true, true];
        let (n1, n2, r1) = row_rank_sum_group1(&vals, &mask);
        assert_eq!(n1, 3);
        assert_eq!(n2, 0);
        assert_eq!(r1, 6.0);
    }

    #[test]
    fn rank_with_ties() {
        let vals = [10.0_f64, 10.0, 20.0];
        let mask = [true, false, true];
        let (n1, n2, r1) = row_rank_sum_group1(&vals, &mask);
        assert_eq!(n1, 2);
        assert_eq!(n2, 1);
        assert!((r1 - 4.5).abs() < 1e-12, "r1 = {}", r1);
    }

    #[test]
    fn two_sided_p_known() {
        let p = two_sided_p(1.959964);
        assert!((p - 0.05).abs() < 1e-4, "p = {}", p);
        let p0 = two_sided_p(0.0);
        assert!((p0 - 1.0).abs() < 1e-12, "p0 = {}", p0);
    }
}
