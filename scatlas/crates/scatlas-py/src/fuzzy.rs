//! PyO3 binding for `scatlas_core::fuzzy`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scatlas_core::fuzzy::{fuzzy_simplicial_set as fuzzy_core, FuzzyParams};

/// Build symmetric fuzzy simplicial set from a `(N, k)` kNN graph.
///
/// Replaces `umap-learn.umap_.fuzzy_simplicial_set` when the caller
/// already has (indices, distances) arrays (e.g., from BBKNN).
///
/// Returns `(indptr, indices, data)` CSR triplets. Caller wraps into
/// `scipy.sparse.csr_matrix` in Python.
#[pyfunction]
#[pyo3(signature = (
    knn_indices,
    knn_dists,
    k=15,
    n_iter=64,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
))]
#[allow(clippy::type_complexity)]
pub fn fuzzy_simplicial_set<'py>(
    py: Python<'py>,
    knn_indices: PyReadonlyArray2<'py, u32>,
    knn_dists: PyReadonlyArray2<'py, f32>,
    k: usize,
    n_iter: usize,
    set_op_mix_ratio: f32,
    local_connectivity: f32,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let idx = knn_indices.as_array();
    let dists = knn_dists.as_array();
    if idx.dim() != dists.dim() {
        return Err(PyValueError::new_err(format!(
            "knn_indices {:?} and knn_dists {:?} shape mismatch",
            idx.dim(),
            dists.dim()
        )));
    }
    let params = FuzzyParams {
        k,
        n_iter,
        set_op_mix_ratio,
        local_connectivity,
    };
    let result = py.allow_threads(|| fuzzy_core(idx, dists, &params));
    Ok((
        ndarray::Array1::from(result.indptr).into_pyarray(py),
        ndarray::Array1::from(result.indices).into_pyarray(py),
        ndarray::Array1::from(result.data).into_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fuzzy_simplicial_set, m)?)?;
    Ok(())
}
