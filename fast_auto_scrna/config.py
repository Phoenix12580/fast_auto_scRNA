"""Pipeline configuration — every knob in one place.

Ported from v1 ``scatlas_pipeline/pipeline.py`` at V2-P2 with all recall
fields stripped (recall replaced by graph-silhouette; see cluster/resolution).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as _np


INTEGRATION_METHODS = ("bbknn", "harmony", "fastmnn", "scvi")
"""Supported per-route integration methods.
* ``bbknn`` — batch-balanced kNN (graph-level correction).
* ``harmony`` — Harmony 2 on X_pca, then plain kNN on X_pca_harmony.
* ``fastmnn`` — Haghverdi 2018 mutual nearest neighbors on X_pca,
  then plain kNN on the corrected embedding.

The ``"none"`` baseline route was removed 2026-04-25 — it added wall
without informing integration-method choice on real atlases."""


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
    # "bbknn" / "harmony" / "fastmnn": run that single route.
    # "all": run every method in INTEGRATION_METHODS and produce comparison.
    integration: str = "bbknn"

    # fastMNN params (only used if integration in {"fastmnn", "all"}).
    # Pure-Python port of Haghverdi 2018 + batchelor::fastMNN; uses our
    # hnswlib + numpy. Standard fastMNN uses k=20.
    fastmnn_n_neighbors: int = 20
    fastmnn_sigma_scale: float = 1.0
    fastmnn_n_threads: int = -1

    # scVI params (only used if integration in {"scvi", "all"}). Defaults
    # follow Gao et al. Cancer Cell 2024 (n_latent=30) + scvi-tools
    # standard hyperparameters. GPU is auto-detected via lightning;
    # CPU-only torch is the typical Windows-dev case (~30-60 min on 222k,
    # ~20 min on a budget Turing GPU like GTX 1660 SUPER).
    scvi_n_latent: int = 30
    scvi_n_hidden: int = 128
    scvi_n_layers: int = 1
    # ``None`` → scvi-tools heuristic
    # ``min(round((20000 / n_cells) * 400), 400)``: 36 epochs on 222k,
    # 18 on 444k, 400 on tiny data. Original 200 default 5× overshot 222k
    # (loss flat by epoch ~15 in our 2026-04-25 GPU bench). Set explicit
    # int to override.
    scvi_max_epochs: int | None = None
    # Early-stop guard: stop if val loss not improving for 45 epochs.
    # Cheap insurance against the heuristic still being too generous.
    scvi_early_stopping: bool = True
    scvi_gene_likelihood: str = "zinb"
    scvi_dispersion: str = "gene"
    scvi_use_hvg: bool = True
    scvi_accelerator: str = "auto"
    scvi_batch_size: int = 128
    scvi_seed: int = 0

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
    # Cluster-homogeneity metrics (ROGUE + SCCAF) — require Leiden first and
    # raw counts in layers['counts'].
    compute_homogeneity: bool = True
    # Embedding ASW silhouettes (label / batch / isolated_label) via the
    # scib-metrics package (JAX-jit'd, chunked). Adds ~5 min per route on
    # 222k cells. Default True to match the metric panel of Gao et al.
    # Cancer Cell 2024 (the reference benchmarking paper). Set False to
    # save ~5 min/route when ASW is not needed.
    compute_silhouette: bool = True
    # kBET acceptance — chi2 test of per-cell kNN batch composition vs
    # global. Removed from default panel 2026-04-25 because:
    #   1. BBKNN's batch-balanced kNN makes kBET nan-by-construction
    #      (constant neighbor composition violates the chi2 null);
    #   2. on heavily imbalanced atlases (e.g. 222k prostate, 20× batch
    #      ratio) kBET is uninformative for embedding-level methods
    #      (harmony/fastmnn/scvi all 0.001-0.008, no discrimination);
    #   3. iLISI captures the same intent (local batch mixing) without
    #      the chi2 fragility.
    # Set True to opt back in when batches are balanced or for diagnostic.
    compute_kbet: bool = False
    # Single plotting control. If set, every route writes into this dir:
    #   umap_<method>.png              (colored by batch / GT / Leiden)
    #   silhouette_curve_<method>.png  (graph-silhouette sweep)
    #   rogue_per_cluster_<method>.png (purity bars)
    #   scib_summary_<method>.png      (1-row metrics heatmap)
    # With >1 methods (e.g. integration='all') also:
    #   integration_comparison.png     (big UMAP grid, written pre-Phase-2)
    #   scib_heatmap.png               (methods × metrics)
    #   rogue_comparison.png           (per-cluster ROGUE grid)
    plot_dir: str | None = None

    # --- Leiden (09) — per-route, auto-resolution
    # v2-P9 default: single-stage 150-point knee picker on 0.01 step.
    # In v2-P9 Leiden only runs for the SELECTED integration method (one,
    # not three), so the 150-point full-accuracy sweep fits in budget.
    # Two-stage (knee_two_stage=True) remains as a fallback for
    # multi-route clustering or aggressive wall budgets.
    run_leiden: bool = True
    leiden_resolutions: list[float] = field(
        default_factory=lambda: [round(r, 2) for r in _np.arange(0.01, 1.51, 0.01)]
    )
    # Legacy clip, used only by target_n / conductance / graph_silhouette optimizers.
    # Ignored by the default "knee" optimizer.
    leiden_target_n: tuple[int, int] = (3, 10)
    leiden_n_iterations: int = 2

    # Resolution optimizer:
    #   "knee" (default v2-P8): PCA-style perpendicular-line elbow on the
    #       conductance-vs-resolution curve (mirrors rust/kernels/src/pca.rs
    #       perpendicular_elbow used for scree selection). Picks
    #       ``knee + knee_offset_steps`` to bias toward finer clustering.
    #       No k-range clip. Calibrated on 2026-04-24 audit: argmin-type
    #       pickers (conductance / silhouette / stability) all edge-pick
    #       trivial k=2-3 on trajectory/sub-lineage data (pancreas); the
    #       knee picker lands in the k=7-10 region which matches the
    #       "over-cluster then marker-merge" user workflow.
    #   "conductance" (v2-P7 legacy): argmin conductance + (3,10) clip.
    #       Worked on atlas-first-pass by luck (clip boundary). Fails on
    #       sub-lineage data.
    #   "graph_silhouette" (legacy): kept for back-compat.
    #   "target_n" (legacy heuristic): smallest res giving k in target.
    resolution_optimizer: str = "knee"
    # Knee picker offset: how many resolution steps past the detected knee
    # to take as the "picked" resolution. 3 steps × 0.01 = r_knee + 0.03.
    knee_offset_steps: int = 3
    # Knee detector:
    #   "first_plateau" (default v2-P8): "快速上升到平台的第一个点" —
    #       first index where (y[i] - y[0]) ≥ 10% × range AND local slope
    #       has dropped to 25% of the max seen so far. Matches user
    #       intuition that the picker should find the FIRST plateau entry
    #       after initial rapid rise, not the globally most-bowed point.
    #   "perp_elbow" (PCA-scree equivalent): global max perpendicular
    #       distance from the secant. Fallback; fails on multi-step curves
    #       where later jumps pull the global secant askew.
    knee_detector: str = "first_plateau"
    # Two-stage sweep — coarse (leiden_resolutions) then fine at knee.
    # v2-P9: default False since Leiden only runs for the selected method
    # once (not 3×), so single-stage 150-point is affordable and more
    # accurate. Set True to re-enable coarse+fine (for atlas-wide Leiden
    # on all routes, or tight wall budgets).
    knee_two_stage: bool = False
    knee_fine_step: float = 0.01
    knee_fine_half_width: float = 0.05

    # v2-P9: in multi-route mode (integration="all"), run Leiden + ROGUE +
    # SCCAF only for the selected method (not all three). Selection is
    # auto (highest scIB mean from Phase 2a) unless ``cluster_method`` is
    # specified.
    cluster_method: str | None = None
    # v2-P10: after winner Phase 2b finishes, optionally run a single
    # Leiden call at the winner's chosen resolution for each non-winner
    # route, then compute ROGUE + SCCAF. Adds ~12 min on 222k (3 non-winners
    # × ~4 min each, vs ~90 min if each ran its own 150-pt sweep). Useful
    # when you want to compare integration methods on equal footing
    # (same k from same resolution) — the standard scIB benchmarking
    # approach. Default off to preserve v2-P9's winner-only fast path.
    cluster_non_winners_at_winner_res: bool = False

    # v2-P9.1: CPU usage control for Leiden sweep workers.
    # ``max_leiden_workers=None`` → reserve 4 cores for OS / foreground
    # (e.g. video playback, browser). Set explicit int to cap tighter.
    # ``leiden_worker_priority`` lowers each worker's OS priority so the
    # host stays responsive while Leiden burns CPU; pure-stdlib impl
    # (ctypes on Windows, os.nice on Unix), no new deps.
    max_leiden_workers: int | None = None
    leiden_worker_priority: str | None = "below_normal"
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
