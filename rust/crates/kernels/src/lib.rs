//! fast_auto_scRNA kernels — pure-Rust algorithm kernels.
//!
//! Every module here is a single pipeline stage's compute core. PyO3 bindings
//! live in the sibling `py_bindings` crate.
//!
//! Stages (populated by V2-P1 migration):
//!   - pca
//!   - bbknn
//!   - harmony
//!   - umap
//!   - fuzzy        (fuzzy_simplicial_set)
//!   - neighbors    (plain kNN for the `none` integration route)
//!   - metrics      (lisi, graph_conn, kbet)
//!   - rogue        (entropy_table + calculate_rogue)
//!   - silhouette   (graph silhouette — replaces recall in v2)

// Modules will be declared here as they're migrated.
