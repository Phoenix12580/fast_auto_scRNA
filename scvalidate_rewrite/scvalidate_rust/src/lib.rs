//! scvalidate_rust — Rust acceleration kernels for scvalidate.
//!
//! Kernels:
//!   - `wilcoxon_ranksum_matrix(log_counts, mask1, mask2) -> pvals`  (M2)
//!   - `knockoff_threshold_offset1(w, fdr) -> threshold`  (M5, TODO)

use pyo3::prelude::*;

mod knockoff;
mod wilcoxon;

/// Probe — returns `true` so Python side can detect the extension is loaded.
#[pyfunction]
fn available() -> bool {
    true
}

#[pymodule]
fn scvalidate_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(available, m)?)?;
    m.add_function(wrap_pyfunction!(
        wilcoxon::wilcoxon_ranksum_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        knockoff::knockoff_threshold_offset1,
        m
    )?)?;
    Ok(())
}
