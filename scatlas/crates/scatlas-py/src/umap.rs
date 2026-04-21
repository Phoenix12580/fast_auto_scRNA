//! PyO3 binding for `scatlas_core::umap`.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scatlas_core::umap::{fit_ab as fit_ab_core, umap_from_connectivities, UmapParams};

/// UMAP layout optimization on a BBKNN-style symmetric CSR
/// connectivity matrix with a provided initial embedding.
///
/// Returns `(embedding, a, b, n_epochs_used)`.
///   * `embedding` — `(n_cells, n_components)` f32
///   * `a`, `b` — fitted kernel params
///   * `n_epochs_used` — actual epochs run (matches input if set)
#[pyfunction]
#[pyo3(signature = (
    connectivities_indptr,
    connectivities_indices,
    connectivities_data,
    n_cells,
    init_embedding,
    n_components=2,
    n_epochs=None,
    min_dist=0.5,
    spread=1.0,
    negative_sample_rate=5,
    repulsion_strength=1.0,
    learning_rate=1.0,
    seed=0,
    single_thread=false,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn umap_layout<'py>(
    py: Python<'py>,
    connectivities_indptr: PyReadonlyArray1<'py, u64>,
    connectivities_indices: PyReadonlyArray1<'py, u32>,
    connectivities_data: PyReadonlyArray1<'py, f32>,
    n_cells: usize,
    init_embedding: PyReadonlyArray2<'py, f32>,
    n_components: usize,
    n_epochs: Option<usize>,
    min_dist: f32,
    spread: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    learning_rate: f32,
    seed: u64,
    single_thread: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32, f32, usize)> {
    let indptr = connectivities_indptr
        .as_slice()
        .map_err(|_| PyValueError::new_err("indptr must be contiguous"))?
        .to_vec();
    let indices = connectivities_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("indices must be contiguous"))?
        .to_vec();
    let data = connectivities_data
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be contiguous"))?
        .to_vec();
    let init = init_embedding.as_array();
    if init.nrows() != n_cells {
        return Err(PyValueError::new_err(format!(
            "init_embedding rows {} ≠ n_cells {}",
            init.nrows(),
            n_cells
        )));
    }
    if init.ncols() != n_components {
        return Err(PyValueError::new_err(format!(
            "init_embedding cols {} ≠ n_components {}",
            init.ncols(),
            n_components
        )));
    }

    let params = UmapParams {
        n_components,
        n_epochs,
        min_dist,
        spread,
        negative_sample_rate,
        repulsion_strength,
        learning_rate,
        seed,
        single_thread,
    };

    let result = py.allow_threads(|| {
        umap_from_connectivities(&indptr, &indices, &data, n_cells, init, &params)
    });

    Ok((
        result.embedding.into_pyarray(py),
        result.a,
        result.b,
        result.n_epochs_used,
    ))
}

/// Fit (a, b) for the UMAP kernel. Exposed for diagnostic use.
#[pyfunction]
#[pyo3(signature = (min_dist=0.5, spread=1.0))]
pub fn fit_ab(min_dist: f32, spread: f32) -> (f32, f32) {
    fit_ab_core(min_dist, spread)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(umap_layout, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ab, m)?)?;
    Ok(())
}
