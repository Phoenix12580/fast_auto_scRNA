//! PyO3 bindings for `scatlas_core::metrics` (scib-metrics parity kernels).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scatlas_core::metrics::{graph_connectivity, kbet_chi2_per_cell, lisi};

/// Per-cell Local Inverse Simpson's Index.
///
/// `knn_distances` and `knn_labels` must have the same `(n_cells, k)`
/// shape. `knn_labels[i, j]` is the label code of the j-th neighbor of
/// cell i; `i32::MIN` marks a padded / missing slot and is skipped.
/// `perplexity` must be > 1 (default 30, matches scib_metrics).
#[pyfunction]
#[pyo3(signature = (knn_distances, knn_labels, perplexity=30.0))]
pub fn lisi_per_cell<'py>(
    py: Python<'py>,
    knn_distances: PyReadonlyArray2<'py, f32>,
    knn_labels: PyReadonlyArray2<'py, i32>,
    perplexity: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let d = knn_distances.as_array();
    let l = knn_labels.as_array();
    let out = py
        .allow_threads(|| lisi(d, l, perplexity))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

/// Mean per-label largest-CC fraction on the provided k-NN graph.
#[pyfunction]
pub fn graph_connectivity_score(
    py: Python<'_>,
    knn_indices: PyReadonlyArray2<u32>,
    labels: PyReadonlyArray1<i32>,
) -> PyResult<f64> {
    let idx = knn_indices.as_array();
    let lbl = labels.as_array();
    if lbl.len() != idx.nrows() {
        return Err(PyValueError::new_err(format!(
            "labels length {} != knn_indices rows {}",
            lbl.len(),
            idx.nrows()
        )));
    }
    Ok(py.allow_threads(|| graph_connectivity(idx, lbl)))
}

/// Per-cell χ² statistic for kBET.
///
/// `knn_labels` shape (n_cells, k) is int32 with `i32::MIN` marking
/// padded slots. `global_counts` length n_batches is uint64 (total
/// dataset cells per batch, indexed by encoded batch code).
#[pyfunction]
pub fn kbet_chi2<'py>(
    py: Python<'py>,
    knn_labels: PyReadonlyArray2<'py, i32>,
    global_counts: PyReadonlyArray1<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let labels = knn_labels.as_array();
    let counts = global_counts.as_array();
    let counts_slice = counts
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("global_counts must be contiguous"))?;
    let out = py.allow_threads(|| kbet_chi2_per_cell(labels, counts_slice));
    Ok(out.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lisi_per_cell, m)?)?;
    m.add_function(wrap_pyfunction!(graph_connectivity_score, m)?)?;
    m.add_function(wrap_pyfunction!(kbet_chi2, m)?)?;
    Ok(())
}
