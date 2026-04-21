//! scib-metrics parity kernels (Luecken et al. 2022).
//!
//! Covers the hot-loop integration metrics that operate on a k-NN graph
//! plus per-cell labels:
//!
//! * [`lisi`] — Local Inverse Simpson's Index (Korsunsky 2019). Used as
//!   iLISI over batch labels and cLISI over cell-type labels. Matches the
//!   `scib_metrics.ilisi_knn` / harmonypy formulation: Gaussian distance
//!   weights with per-cell perplexity calibration.
//! * [`graph_connectivity`] — mean per-label fraction of cells in the
//!   largest connected component of the same-label sub-graph.
//!
//! * [`kbet_chi2_per_cell`] — per-cell χ² goodness-of-fit statistic of
//!   neighbor batch composition against global; p-values and acceptance
//!   rate computed Python-side via `scipy.stats.chi2.sf`.
//!
//! Deferred (Python-only in scatlas):
//! * silhouette (ASW) — scikit-learn's `silhouette_samples` is already
//!   fast enough at typical scales to not warrant a Rust port.

pub mod graph_conn;
pub mod kbet;
pub mod lisi;

pub use graph_conn::graph_connectivity;
pub use kbet::kbet_chi2_per_cell;
pub use lisi::{lisi, LisiError};
