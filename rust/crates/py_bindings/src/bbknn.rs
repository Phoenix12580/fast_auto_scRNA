//! Integration / batch-correction kernels. Right now: BBKNN.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use kernels::bbknn;

/// Batch-balanced k-NN on a PCA embedding.
///
/// `pca` is float32 `(n_cells, n_dims)`. `batch_labels` is int32 length
/// n_cells. `backend` is one of `"brute"`, `"hnsw"`, `"auto"` (default;
/// switches to HNSW when the largest batch exceeds `auto_threshold`).
/// `ef_search` is the HNSW query breadth (ignored for brute); `None`
/// picks `max(32, 4*k_per_batch)`.
///
/// Returns `(indices, distances, batches)` with the first two arrays
/// shape `(n_cells, k_per_batch * n_batches)` ordered by ascending batch
/// label. Unused slots (for batches smaller than `k_per_batch`) hold
/// `u32::MAX` / `inf`.
#[pyfunction]
#[pyo3(signature = (pca, batch_labels, k_per_batch=3, backend="auto", ef_search=None, auto_threshold=5000))]
#[allow(clippy::type_complexity)]
pub fn bbknn_kneighbors<'py>(
    py: Python<'py>,
    pca: PyReadonlyArray2<'py, f32>,
    batch_labels: PyReadonlyArray1<'py, i32>,
    k_per_batch: usize,
    backend: &str,
    ef_search: Option<usize>,
    auto_threshold: usize,
) -> PyResult<(
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i32>>,
)> {
    if k_per_batch == 0 {
        return Err(PyValueError::new_err("k_per_batch must be > 0"));
    }
    let pca_view = pca.as_array();
    let batch_view = batch_labels.as_array();
    if batch_view.len() != pca_view.nrows() {
        return Err(PyValueError::new_err(format!(
            "batch_labels has length {} but pca has {} rows",
            batch_view.len(),
            pca_view.nrows()
        )));
    }

    let ef = ef_search.unwrap_or_else(|| 32.max(4 * k_per_batch));

    let use_hnsw = match backend {
        "brute" => false,
        "hnsw" => true,
        "auto" => {
            // Count largest batch — brute cost ∝ N * max_batch, hnsw build
            // amortizes well once any batch exceeds the threshold.
            let mut counts: std::collections::HashMap<i32, usize> =
                std::collections::HashMap::new();
            for &b in batch_view.iter() {
                *counts.entry(b).or_insert(0) += 1;
            }
            counts.values().copied().max().unwrap_or(0) >= auto_threshold
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "backend must be one of 'brute' / 'hnsw' / 'auto', got {:?}",
                other
            )))
        }
    };

    let res = py.allow_threads(|| {
        if use_hnsw {
            bbknn::bbknn_hnsw(pca_view, batch_view, k_per_batch, ef)
        } else {
            bbknn::bbknn_brute(pca_view, batch_view, k_per_batch)
        }
    });

    let n_cells = pca_view.nrows();
    let idx_arr = ndarray::Array2::from_shape_vec((n_cells, res.k_total), res.indices)
        .map_err(|e| PyValueError::new_err(format!("indices shape error: {}", e)))?;
    let dist_arr = ndarray::Array2::from_shape_vec((n_cells, res.k_total), res.distances)
        .map_err(|e| PyValueError::new_err(format!("distances shape error: {}", e)))?;
    let batches_arr = ndarray::Array1::from(res.batches);

    Ok((
        idx_arr.into_pyarray(py),
        dist_arr.into_pyarray(py),
        batches_arr.into_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bbknn_kneighbors, m)?)?;
    Ok(())
}
