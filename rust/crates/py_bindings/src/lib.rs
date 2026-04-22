//! PyO3 bindings → compiled as `fast_auto_scrna._native`.
//!
//! Each pipeline stage registers a submodule here after V2-P1 migration:
//!   fast_auto_scrna._native.pca
//!   fast_auto_scrna._native.bbknn
//!   fast_auto_scrna._native.harmony
//!   fast_auto_scrna._native.umap
//!   fast_auto_scrna._native.fuzzy
//!   fast_auto_scrna._native.neighbors
//!   fast_auto_scrna._native.metrics
//!   fast_auto_scrna._native.rogue
//!   fast_auto_scrna._native.silhouette

use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Submodule registration will go here as kernels are ported.
    Ok(())
}
