//! PyO3 binding for `kernels::silhouette`.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use kernels::silhouette;

/// Silhouette score on a precomputed distance matrix.
///
/// Matches ``sklearn.metrics.silhouette_score(X, labels, metric="precomputed")``
/// semantics. Returns 0.0 when fewer than 2 distinct clusters are present
/// (sklearn would raise instead — this matches the Python fallback in
/// ``cluster/resolution.py`` which returns 0 for that case).
///
/// Parameters
/// ----------
/// distances
///     ``(n, n)`` float32 distance matrix. Typically symmetric with
///     zero diagonal. Caller must ensure contiguous layout.
/// labels
///     ``(n,)`` int32 cluster assignments. Ids need not be contiguous
///     0..k-1 — the kernel remaps internally.
#[pyfunction]
pub fn silhouette_precomputed<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray2<'py, f32>,
    labels: PyReadonlyArray1<'py, i32>,
) -> PyResult<f32> {
    let d = distances.as_array();
    let l = labels.as_array();

    if d.shape()[0] != l.len() || d.shape()[0] != d.shape()[1] {
        return Err(PyValueError::new_err(format!(
            "shape mismatch: distances {:?}, labels {}",
            d.shape(),
            l.len()
        )));
    }

    Ok(py.allow_threads(|| silhouette::silhouette_precomputed(d, l)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(silhouette_precomputed, m)?)?;
    Ok(())
}
