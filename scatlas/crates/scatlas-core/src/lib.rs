//! scatlas-core — pure Rust kernels for scRNA-seq atlas computation.
//!
//! Consumed by `scatlas-py` (PyO3 wheel) and, optionally later, `scatlas-r`
//! (extendr_api). No Python or R types appear in this crate.

pub mod bbknn;
pub mod fuzzy;
pub mod harmony;
pub mod metrics;
pub mod pca;
pub mod stats;
pub mod umap;
