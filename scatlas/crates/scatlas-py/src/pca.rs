//! PyO3 binding for `scatlas_core::pca`.
//!
//! Exposes randomized truncated SVD on CSR sparse + dense f32 inputs.
//! Matches scanpy's `pp.pca(adata, zero_center=False)` path for sparse.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scatlas_core::pca::{pca_csr_f32, pca_dense_f32, suggest_n_comps, RsvdParams};

/// Randomized PCA on a CSR-format sparse matrix (no mean-centering).
/// Matches `sklearn.decomposition.TruncatedSVD(algorithm="randomized")`.
///
/// Returns `(embedding, components, singular_values,
/// explained_variance, explained_variance_ratio)`.
#[pyfunction]
#[pyo3(signature = (
    indptr,
    indices,
    data,
    n_rows,
    n_cols,
    n_comps=30,
    n_oversamples=10,
    n_power_iter=7,
    seed=0,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn pca_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u64>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_rows: usize,
    n_cols: usize,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iter: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    if n_comps == 0 {
        return Err(PyValueError::new_err("n_comps must be ≥ 1"));
    }
    if n_comps > n_rows.min(n_cols) {
        return Err(PyValueError::new_err(format!(
            "n_comps ({}) > min(n_rows={}, n_cols={})",
            n_comps, n_rows, n_cols
        )));
    }

    let indptr_slice = indptr.as_slice().unwrap_or(&[]).to_vec();
    let indices_slice = indices.as_slice().unwrap_or(&[]).to_vec();
    let data_slice = data.as_slice().unwrap_or(&[]).to_vec();

    let params = RsvdParams {
        n_oversamples,
        n_power_iter,
    };

    let result = py.allow_threads(|| {
        pca_csr_f32(
            &indptr_slice,
            &indices_slice,
            &data_slice,
            n_rows,
            n_cols,
            n_comps,
            params,
            seed,
        )
    });

    Ok((
        result.embedding.into_pyarray(py),
        result.components.into_pyarray(py),
        result.singular_values.into_pyarray(py),
        result.explained_variance.into_pyarray(py),
        result.explained_variance_ratio.into_pyarray(py),
    ))
}

/// Randomized PCA on a dense f32 matrix (no mean-centering).
#[pyfunction]
#[pyo3(signature = (
    x,
    n_comps=30,
    n_oversamples=10,
    n_power_iter=7,
    seed=0,
))]
#[allow(clippy::type_complexity)]
pub fn pca_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iter: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let view = x.as_array();
    if n_comps == 0 {
        return Err(PyValueError::new_err("n_comps must be ≥ 1"));
    }
    if n_comps > view.nrows().min(view.ncols()) {
        return Err(PyValueError::new_err(format!(
            "n_comps ({}) > min(N={}, G={})",
            n_comps,
            view.nrows(),
            view.ncols()
        )));
    }
    let params = RsvdParams {
        n_oversamples,
        n_power_iter,
    };

    let result = py.allow_threads(|| pca_dense_f32(view, n_comps, params, seed));

    Ok((
        result.embedding.into_pyarray(py),
        result.components.into_pyarray(py),
        result.singular_values.into_pyarray(py),
        result.explained_variance.into_pyarray(py),
        result.explained_variance_ratio.into_pyarray(py),
    ))
}

/// Suggest an optimal number of PCs from a vector of singular values.
///
/// Primary criterion: Gavish-Donoho 2014 optimal hard threshold
/// (`τ* = ω(β) · median(sv)`, β = min(N,G)/max(N,G)). MSE-optimal
/// truncation under iid noise.
///
/// Returns a dict with keys: `n_comps_gavish_donoho`, `n_comps_elbow`,
/// `suggested_n_comps`, `gd_threshold`, `sv_median`, `beta`.
#[pyfunction]
#[pyo3(signature = (
    singular_values,
    n_rows,
    n_cols,
    margin=5,
    min_comps=15,
    max_comps=50,
))]
pub fn suggest_n_comps_py<'py>(
    py: Python<'py>,
    singular_values: PyReadonlyArray1<'py, f32>,
    n_rows: usize,
    n_cols: usize,
    margin: usize,
    min_comps: usize,
    max_comps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let sv_slice = singular_values
        .as_slice()
        .map_err(|_| PyValueError::new_err("singular_values must be contiguous"))?;
    if sv_slice.is_empty() {
        return Err(PyValueError::new_err("need at least 1 singular value"));
    }
    let s = suggest_n_comps(sv_slice, n_rows, n_cols, margin, min_comps, max_comps);

    let dict = PyDict::new(py);
    dict.set_item("n_comps_gavish_donoho", s.n_comps_gavish_donoho)?;
    dict.set_item("n_comps_elbow", s.n_comps_elbow)?;
    dict.set_item("suggested_n_comps", s.suggested_n_comps)?;
    dict.set_item("gd_threshold", s.gd_threshold)?;
    dict.set_item("sv_median", s.sv_median)?;
    dict.set_item("beta", s.beta)?;
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pca_csr, m)?)?;
    m.add_function(wrap_pyfunction!(pca_dense, m)?)?;
    m.add_function(wrap_pyfunction!(suggest_n_comps_py, m)?)?;
    Ok(())
}
