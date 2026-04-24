//! PyO3 bindings → compiled as `fast_auto_scrna._native`.
//!
//! Each pipeline stage is a submodule; Python accesses them as e.g.
//! `fast_auto_scrna._native.pca.pca_csr(...)`.
//!
//! Migrated from v1 `scatlas-py` at V2-P1. The `stats` submodule was dropped
//! along with wilcoxon/knockoff; the remaining ROGUE bindings live under the
//! top-level `rogue` submodule. BBKNN was promoted out of the v1 `ext` grab
//! bag into its own submodule. `silhouette` added at GS-3 (V2-P6) — powers
//! the graph-silhouette resolution sweep that previously fell back to sklearn.

use pyo3::prelude::*;

mod bbknn;
mod fuzzy;
mod harmony;
mod metrics;
mod pca;
mod rogue;
mod silhouette;
mod umap;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let sys = py.import("sys")?;
    let sys_modules = sys.getattr("modules")?;

    // Stage 05 — PCA (randomized SVD + Gavish-Donoho selector).
    let pca_mod = PyModule::new(py, "pca")?;
    pca::register(&pca_mod)?;
    m.add_submodule(&pca_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.pca", &pca_mod)?;

    // Stage 06 — BBKNN (batch-balanced kNN).
    let bbknn_mod = PyModule::new(py, "bbknn")?;
    bbknn::register(&bbknn_mod)?;
    m.add_submodule(&bbknn_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.bbknn", &bbknn_mod)?;

    // Stage 06 — Harmony 2.
    let harmony_mod = PyModule::new(py, "harmony")?;
    harmony::register(&harmony_mod)?;
    m.add_submodule(&harmony_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.harmony", &harmony_mod)?;

    // Stage 07 — fuzzy_simplicial_set (neighbor-graph connectivities).
    let fuzzy_mod = PyModule::new(py, "fuzzy")?;
    fuzzy::register(&fuzzy_mod)?;
    m.add_submodule(&fuzzy_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.fuzzy", &fuzzy_mod)?;

    // Stage 08 — scIB metrics (LISI / graph_connectivity / kBET).
    let metrics_mod = PyModule::new(py, "metrics")?;
    metrics::register(&metrics_mod)?;
    m.add_submodule(&metrics_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.metrics", &metrics_mod)?;

    // Stage 09 — UMAP layout SGD.
    let umap_mod = PyModule::new(py, "umap")?;
    umap::register(&umap_mod)?;
    m.add_submodule(&umap_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.umap", &umap_mod)?;

    // Stage 11 — ROGUE purity.
    let rogue_mod = PyModule::new(py, "rogue")?;
    rogue::register(&rogue_mod)?;
    m.add_submodule(&rogue_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.rogue", &rogue_mod)?;

    // Stage 10 — silhouette (precomputed-distance kernel, powers the
    // graph-silhouette resolution sweep in cluster/resolution.py).
    let silhouette_mod = PyModule::new(py, "silhouette")?;
    silhouette::register(&silhouette_mod)?;
    m.add_submodule(&silhouette_mod)?;
    sys_modules.set_item("fast_auto_scrna._native.silhouette", &silhouette_mod)?;

    Ok(())
}
