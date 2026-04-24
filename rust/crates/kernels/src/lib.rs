//! fast_auto_scRNA kernels — pure-Rust algorithm kernels.
//!
//! Consumed by the sibling `py_bindings` crate (PyO3 wheel). No Python types
//! appear here.
//!
//! Ported from v1 `scatlas-core` at commit `c1107e8` (V2-P1). `wilcoxon` /
//! `knockoff` were intentionally dropped — they were recall-only, and recall
//! was replaced by the graph-silhouette optimizer in v2. `silhouette` lands in
//! a later milestone (GS-3).

pub mod bbknn;
pub mod fuzzy;
pub mod harmony;
pub mod metrics;
pub mod pca;
pub mod rogue;
pub mod silhouette;
pub mod umap;
