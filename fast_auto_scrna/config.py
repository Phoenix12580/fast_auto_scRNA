"""Pipeline configuration — every knob in one place.

Ported from v1 ``scatlas_pipeline/pipeline.py`` at V2-P2 with all recall
fields stripped (recall replaced by graph-silhouette; see cluster/resolution).
"""
from __future__ import annotations

from dataclasses import dataclass, field


INTEGRATION_METHODS = ("none", "bbknn", "harmony")
"""Supported per-route integration methods. ``"none"`` = plain kNN on
X_pca (batch-effect baseline). ``"bbknn"`` = batch-balanced kNN
(graph-level batch correction). ``"harmony"`` = Harmony 2 on X_pca then
plain kNN on X_pca_harmony (embedding-level batch correction)."""


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration.

    Each ``integration`` route runs an independent downstream chain
    (kNN → UMAP → scIB → Leiden). When ``integration="all"``, every
    method in ``INTEGRATION_METHODS`` runs; results are stored under
    per-method keys (``X_umap_<method>``, ``leiden_<method>``, ...).
    """

    # --- Input
    input_h5ad: str
    batch_key: str = "orig.ident"
    out_h5ad: str | None = None

    # --- QC (01)
    min_cells: int = 3                    # genes in < min_cells cells → drop
    min_genes: int = 200                  # cells with < min_genes genes → drop
    max_pct_mt: float = 20.0              # cells with MT fraction > pct → drop
    mt_prefix: str = "MT-"

    # --- Normalization (02)
    target_sum: float = 1e4

    # --- HVG (03) — required, gates scale + PCA inputs
    hvg_n_top_genes: int = 2000
    hvg_flavor: str = "seurat_v3"         # VST on counts
    hvg_batch_aware: bool = True          # select HVGs per-batch then intersect

    # --- scale (04) — required, zeroes mean + clips z to [-10, 10]
    scale_max_value: float = 10.0
    scale_zero_center: bool = True

    # --- PCA (05)
    pca_n_comps: int | str = "auto"       # "auto" → Gavish-Donoho
    pca_n_power_iter: int = 7
    pca_random_state: int = 0

    # --- Integration route (06) ---------------------------------------------
    # "none" / "bbknn" / "harmony": run that single route.
    # "all": run every method in INTEGRATION_METHODS and produce comparison.
    integration: str = "bbknn"

    # BBKNN params (only used if integration in {"bbknn", "all"})
    neighbors_within_batch: int = 3
    bbknn_backend: str = "auto"           # "brute" / "hnsw" / "auto"

    # Harmony 2 params (only used if integration in {"harmony", "all"}).
    # Defaults diverge from R RunHarmony() in two places based on the
    # 157k-epithelia ablation from v1:
    #   * theta=4 (not R's 2) — at theta=2 Harmony stalls at iter 2 with
    #     iLISI=0.086; theta=4 keeps iterating to iter 9, iLISI=0.174.
    #   * max_iter=20 (not R's 10) — gives theta=4 enough runway.
    # For R-parity on a specific dataset, set theta=2, max_iter=10.
    harmony_n_clusters: int | None = None   # None → min(round(N/30), 100)
    harmony_max_iter: int = 20
    harmony_max_iter_cluster: int = 20
    harmony_theta: float = 4.0              # batch-diversity weight
    harmony_sigma: float = 0.1              # soft-cluster temperature
    harmony_lambda: float | None = 1.0      # None → dynamic α·E[k,b]
    harmony_alpha: float = 0.2              # dynamic-lambda scaling
    harmony_epsilon_cluster: float = 1e-3
    harmony_epsilon_harmony: float = 1e-2
    harmony_block_size: float = 0.05
    harmony_seed: int = 0

    # Plain kNN params (used for "none" + harmony's post-correction kNN).
    # Defaults match Seurat/SCOP scRNA convention (cosine, k=30).
    knn_n_neighbors: int = 30
    knn_metric: str = "cosine"

    # --- UMAP (07)
    run_umap: bool = True
    umap_min_dist: float = 0.5
    umap_spread: float = 1.0
    umap_n_epochs: int = 200
    umap_init: str = "pca"
    umap_random_state: int = 0

    # --- scIB metrics (08) — per-route
    run_metrics: bool = True
    label_key: str | None = None          # ground-truth cell type; None → Leiden
    # silhouette is O(N²) in sklearn — at 157k it's ~7 min per route.
    # Leave on for small / mid benchmarks; set False for atlas-scale runs.
    compute_silhouette: bool = True
    # Cluster-homogeneity metrics (ROGUE + SCCAF) — require Leiden first and
    # raw counts in layers['counts'].
    compute_homogeneity: bool = True
    # Optional auto-generated side-by-side comparison (needs >1 route).
    write_comparison_plot: str | None = None   # path to output PNG

    # --- Leiden (09) — per-route, auto-resolution
    # v1 defaults target MAJOR LINEAGE level (epithelia/immune/stromal/...),
    # not fine subtypes. Subclustering per lineage is a separate pass.
    run_leiden: bool = True
    leiden_resolutions: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5]
    )
    leiden_target_n: tuple[int, int] = (3, 10)    # pick smallest res giving k in [3, 10]
    leiden_n_iterations: int = 2

    # Resolution optimizer — "graph_silhouette" (default, data-driven) or
    # "target_n" (legacy heuristic).
    resolution_optimizer: str = "graph_silhouette"
    silhouette_n_subsample: int = 1000
    silhouette_n_iter: int = 100
    silhouette_stratify: bool = True  # stratify by first baseline leiden res

    def integration_methods(self) -> tuple[str, ...]:
        """Expand ``integration`` to the concrete list of routes to run."""
        if self.integration == "all":
            return INTEGRATION_METHODS
        if self.integration not in INTEGRATION_METHODS:
            raise ValueError(
                f"integration={self.integration!r} must be one of "
                f"{('all',) + INTEGRATION_METHODS}"
            )
        return (self.integration,)
