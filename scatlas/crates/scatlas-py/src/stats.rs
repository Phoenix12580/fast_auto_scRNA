//! PyO3 bindings for `scatlas_core::stats`. Thin adapter: dtype-dispatch,
//! GIL release, error mapping.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scatlas_core::stats::{knockoff, rogue, wilcoxon};

/// Per-row Wilcoxon rank-sum p-values; accepts float32 or float64 matrices.
#[pyfunction]
pub fn wilcoxon_ranksum_matrix<'py>(
    py: Python<'py>,
    log_counts: &Bound<'py, PyAny>,
    mask1: PyReadonlyArray1<'py, bool>,
    mask2: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if let Ok(arr) = log_counts.extract::<PyReadonlyArray2<f64>>() {
        let x = arr.as_array();
        let m1 = mask1.as_array();
        let m2 = mask2.as_array();
        let out = py
            .allow_threads(|| wilcoxon::wilcoxon_ranksum_matrix(x, m1, m2))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(out.into_pyarray(py));
    }
    if let Ok(arr) = log_counts.extract::<PyReadonlyArray2<f32>>() {
        let x = arr.as_array();
        let m1 = mask1.as_array();
        let m2 = mask2.as_array();
        let out = py
            .allow_threads(|| wilcoxon::wilcoxon_ranksum_matrix(x, m1, m2))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(out.into_pyarray(py));
    }
    Err(PyValueError::new_err(
        "log_counts must be float32 or float64 2D numpy array",
    ))
}

/// Barber-Candès knockoff threshold (offset=1).
#[pyfunction]
pub fn knockoff_threshold_offset1(w: PyReadonlyArray1<f64>, fdr: f64) -> f64 {
    match w.as_slice() {
        Ok(slice) => knockoff::knockoff_threshold_offset1(slice, fdr),
        Err(_) => {
            let owned: Vec<f64> = w.as_array().iter().copied().collect();
            knockoff::knockoff_threshold_offset1(&owned, fdr)
        }
    }
}

/// Per-gene ROGUE entropy table. Returns a flat `(n_genes * 2,)` f64 array
/// laid out as `[mean_expr_0, entropy_0, mean_expr_1, entropy_1, ...]`; the
/// Python shim reshapes to `(n_genes, 2)`.
#[pyfunction]
#[pyo3(signature = (expr, r=1.0))]
pub fn entropy_table<'py>(
    py: Python<'py>,
    expr: &Bound<'py, PyAny>,
    r: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if let Ok(arr) = expr.extract::<PyReadonlyArray2<f64>>() {
        let x = arr.as_array();
        let out = py.allow_threads(|| rogue::entropy_table_dense(x, r));
        return Ok(out.into_pyarray(py));
    }
    if let Ok(arr) = expr.extract::<PyReadonlyArray2<f32>>() {
        let x = arr.as_array();
        let out = py.allow_threads(|| rogue::entropy_table_dense(x, r));
        return Ok(out.into_pyarray(py));
    }
    Err(PyValueError::new_err(
        "expr must be float32 or float64 2D numpy array \
         (for CSR sparse, use entropy_table_csr)",
    ))
}

/// CSR sparse variant: pass `indptr` (len=n_genes+1) and `data` (len=nnz).
#[pyfunction]
#[pyo3(signature = (indptr, data, n_cells, r=1.0))]
pub fn entropy_table_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    data: &Bound<'py, PyAny>,
    n_cells: usize,
    r: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr_arr = indptr.as_array();
    let indptr_usize: Vec<usize> = indptr_arr
        .iter()
        .map(|&v| if v < 0 { 0 } else { v as usize })
        .collect();

    if let Ok(arr) = data.extract::<PyReadonlyArray1<f64>>() {
        let d = arr.as_array();
        let d_slice: &[f64] = d
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("data must be contiguous"))?;
        let out = py.allow_threads(|| rogue::entropy_table_csr(&indptr_usize, d_slice, n_cells, r));
        return Ok(out.into_pyarray(py));
    }
    if let Ok(arr) = data.extract::<PyReadonlyArray1<f32>>() {
        let d = arr.as_array();
        let d_slice: &[f32] = d
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("data must be contiguous"))?;
        let out = py.allow_threads(|| rogue::entropy_table_csr(&indptr_usize, d_slice, n_cells, r));
        return Ok(out.into_pyarray(py));
    }
    Err(PyValueError::new_err(
        "data must be float32 or float64 1D numpy array",
    ))
}

/// `1 - sig / (sig + k)` over genes passing p_adj<cutoff && p_value<cutoff.
#[pyfunction]
pub fn calculate_rogue(
    ds: PyReadonlyArray1<f64>,
    p_adj: PyReadonlyArray1<f64>,
    p_value: PyReadonlyArray1<f64>,
    cutoff: f64,
    k: f64,
) -> PyResult<f64> {
    let ds_arr = ds.as_array();
    let pa_arr = p_adj.as_array();
    let pv_arr = p_value.as_array();
    if ds_arr.len() != pa_arr.len() || ds_arr.len() != pv_arr.len() {
        return Err(PyValueError::new_err(
            "ds / p_adj / p_value must have the same length",
        ));
    }
    let ds_v: Vec<f64> = ds_arr.iter().copied().collect();
    let pa_v: Vec<f64> = pa_arr.iter().copied().collect();
    let pv_v: Vec<f64> = pv_arr.iter().copied().collect();
    Ok(rogue::calculate_rogue(&ds_v, &pa_v, &pv_v, cutoff, k))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wilcoxon_ranksum_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(knockoff_threshold_offset1, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_table, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_table_csr, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rogue, m)?)?;
    Ok(())
}
