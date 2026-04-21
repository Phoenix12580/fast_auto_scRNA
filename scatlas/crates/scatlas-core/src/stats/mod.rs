//! Robust statistics for single-cell DEG / FDR calibration.
//!
//! Ported from `scvalidate_rust` (scvalidate_rewrite/scvalidate_rust/src),
//! MIT-licensed. Kept pure-Rust with `ndarray` views; the PyO3 layer lives in
//! `scatlas-py::stats` and calls into these functions.

pub mod knockoff;
pub mod rogue;
pub mod wilcoxon;

pub use knockoff::knockoff_threshold_offset1;
pub use rogue::{calculate_rogue, entropy_table_csr, entropy_table_dense};
pub use wilcoxon::{wilcoxon_ranksum_matrix, WilcoxonError};
