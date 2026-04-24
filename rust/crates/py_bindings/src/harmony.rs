//! PyO3 binding for `kernels::harmony`.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use kernels::harmony::{harmony_integrate as harmony_run, HarmonyParams, LambdaMode};

/// Run Harmony 2.0 on a PCA embedding.
///
/// `pca` is `(n_cells, n_dims)` float32 row-major.
/// `batch_codes` is length-n_cells int32 (encoded 0..B-1).
///
/// Returns `(z_corrected, r, y, objective_harmony, converged_at_iter)`:
///   * `z_corrected` (n_cells, n_dims) float32
///   * `r` (n_clusters, n_cells) float32
///   * `y` (n_dims, n_clusters) float32
///   * `objective_harmony` (iters,) float32
///   * `converged_at_iter` int or None (None if max_iter reached)
#[pyfunction]
#[pyo3(signature = (
    pca,
    batch_codes,
    n_clusters=None,
    theta=2.0,
    sigma=0.1,
    lambda_=Some(1.0),
    alpha=0.2,
    max_iter=10,
    max_iter_cluster=20,
    epsilon_cluster=1e-3,
    epsilon_harmony=1e-2,
    block_size=0.05,
    seed=0,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn harmony_integrate<'py>(
    py: Python<'py>,
    pca: PyReadonlyArray2<'py, f32>,
    batch_codes: PyReadonlyArray1<'py, i32>,
    n_clusters: Option<usize>,
    theta: f32,
    sigma: f32,
    lambda_: Option<f32>,
    alpha: f32,
    max_iter: usize,
    max_iter_cluster: usize,
    epsilon_cluster: f32,
    epsilon_harmony: f32,
    block_size: f32,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Option<usize>,
)> {
    let pca_view = pca.as_array();
    let batch_view = batch_codes.as_array();
    let n_cells = pca_view.nrows();
    if batch_view.len() != n_cells {
        return Err(PyValueError::new_err(format!(
            "batch_codes length {} != pca rows {}",
            batch_view.len(),
            n_cells
        )));
    }
    if n_cells < 6 {
        return Err(PyValueError::new_err(
            "refusing to run Harmony with fewer than 6 cells",
        ));
    }

    // Match R's default: min(round(N/30), 100).
    let n_clusters = n_clusters.unwrap_or_else(|| {
        let rounded = ((n_cells as f32) / 30.0).round() as usize;
        rounded.clamp(1, 100)
    });

    // lambda=None → dynamic mode (matches RunHarmony(lambda=NULL) in R).
    let (lambda_mode, lambda_val) = match lambda_ {
        Some(v) => (LambdaMode::Fixed, v),
        None => (LambdaMode::Dynamic, 1.0),
    };

    let params = HarmonyParams {
        n_clusters,
        theta,
        sigma,
        lambda_mode,
        lambda: lambda_val,
        alpha,
        max_iter_harmony: max_iter,
        max_iter_cluster,
        epsilon_cluster,
        epsilon_harmony,
        block_size,
        window_size: 3,
        seed,
    };

    let result = py.allow_threads(|| harmony_run(pca_view, batch_view, &params));
    let obj = ndarray::Array1::from(result.objective_harmony);

    Ok((
        result.z_corrected.into_pyarray(py),
        result.r.into_pyarray(py),
        result.y.into_pyarray(py),
        obj.into_pyarray(py),
        result.converged_at_iter,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(harmony_integrate, m)?)?;
    Ok(())
}
