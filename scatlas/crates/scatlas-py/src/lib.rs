use pyo3::prelude::*;

mod ext;
mod fuzzy;
mod harmony;
mod metrics;
mod pca;
mod stats;
mod umap;

#[pymodule]
fn _scatlas_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let sys = py.import("sys")?;
    let sys_modules = sys.getattr("modules")?;

    let stats_mod = PyModule::new(py, "stats")?;
    stats::register(&stats_mod)?;
    m.add_submodule(&stats_mod)?;
    sys_modules.set_item("scatlas._scatlas_native.stats", &stats_mod)?;

    let ext_mod = PyModule::new(py, "ext")?;
    ext::register(&ext_mod)?;
    m.add_submodule(&ext_mod)?;
    sys_modules.set_item("scatlas._scatlas_native.ext", &ext_mod)?;

    let metrics_mod = PyModule::new(py, "metrics")?;
    metrics::register(&metrics_mod)?;
    m.add_submodule(&metrics_mod)?;
    sys_modules.set_item("scatlas._scatlas_native.metrics", &metrics_mod)?;

    let pp_mod = PyModule::new(py, "pp")?;
    pca::register(&pp_mod)?;
    m.add_submodule(&pp_mod)?;
    sys_modules.set_item("scatlas._scatlas_native.pp", &pp_mod)?;

    let tl_mod = PyModule::new(py, "tl")?;
    umap::register(&tl_mod)?;
    m.add_submodule(&tl_mod)?;
    sys_modules.set_item("scatlas._scatlas_native.tl", &tl_mod)?;

    // Harmony exposed as a function directly under ext (same namespace
    // as bbknn_kneighbors) for API symmetry; the Rust impl lives in its
    // own PyO3 module file for separation of concerns.
    harmony::register(&ext_mod)?;

    // Fuzzy simplicial set (BBKNN's connectivity step). Exposed under
    // `ext` because it's the companion to `bbknn_kneighbors`.
    fuzzy::register(&ext_mod)?;

    Ok(())
}
