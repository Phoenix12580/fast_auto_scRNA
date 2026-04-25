"""Pipeline configuration — every knob in one place.

Ported from v1 ``scatlas_pipeline/pipeline.py`` at V2-P2 with all recall
fields stripped (recall replaced by graph-silhouette; see cluster/resolution).
"""
from __future__ import annotations

from dataclasses import dataclass


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
    # v2-P11: parallelize Phase 2a scIB across integration routes via
    # ProcessPoolExecutor. Numerically bit-identical to the sequential
    # path (same scib-metrics JAX kernels, same input arrays — verified
    # on 222k v2-P10 baseline, |Δ| ≤ 1.79e-7).
    #
    # DEFAULT FALSE because on BLAS-saturated CPU it's a wash or worse:
    # 222k 4-route bench (16-core WSL, 4 workers × 4 BLAS threads) was
    # 0.92× — JAX silhouette is BLAS-bound, splitting 16 threads 4-ways
    # slows each route ~4× so total wall doesn't move. Synthetic 500-cell
    # tests show 2–3× because matrices are too small for BLAS to dominate
    # — that win does not transfer to atlas data. Set True only when:
    #   * core count ≫ silhouette BLAS scaling cliff (probably 32+ cores
    #     where scib's JAX silhouette stops scaling per process), OR
    #   * silhouettes are off (cfg.compute_silhouette=False) so the
    #     non-BLAS bookkeeping dominates and parallel hides Python latency.
    # Real ASW speedup needs a different attack — see ROADMAP "GPU
    # silhouette" milestone.
    scib_parallel: bool = False
    # ``None`` → cap at min(n_routes, max(1, cpu_count - 4)) so 4 cores
    # stay reserved for OS / foreground apps. Set int to override.
    scib_max_workers: int | None = None
    # Single plotting control. If set, every route writes into this dir:
    #   umap_<method>.png               (colored by batch / GT / Leiden)
    #   champ_curve_<method>.png        (CHAMP modularity landscape + γ-range)
    #   rogue_per_cluster_<method>.png  (purity bars)
    #   scib_summary_<method>.png       (1-row metrics heatmap)
    # With >1 methods (e.g. integration='all') also:
    #   integration_comparison.png     (big UMAP grid, written pre-Phase-2)
    #   scib_heatmap.png               (methods × metrics)
    #   rogue_comparison.png           (per-cluster ROGUE grid)
    plot_dir: str | None = None

    # --- Leiden (09) — CHAMP resolution selector (Weir et al. 2017)
    # v2-P12: CHAMP is the only optimizer. The previous knee/conductance/
    # graph_silhouette/target_n options were removed — all relied on
    # heuristics fragile to sampling/range and required dense sweeps with
    # no statistical principle. CHAMP is deterministic, runs ~30 Leidens
    # vs 150, and on the 222k baseline aligns better with GT cell-type
    # structure (k=8 vs k=12). See cluster/champ.py module docstring.
    run_leiden: bool = True
    leiden_n_iterations: int = 2

    # CHAMP picker — defaults follow Weir 2017 with γ range narrowed to
    # atlas typical (knee at r≈0.2-0.4 sits comfortably inside [0.05, 1.50]).
    champ_n_partitions: int = 30
    champ_gamma_min: float = 0.05
    champ_gamma_max: float = 1.50
    # 'newman' (Newman-Girvan) or 'cpm' (Constant Potts Model — the
    # latter is resolution-limit-free per Fortunato-Barthelemy 2007;
    # CPM partitions need leidenalg.CPMVertexPartition so the
    # candidate set and scoring share an objective).
    champ_modularity: str = "newman"
    # 'log' (default, omicverse) — γ-space is scale-free under γ→cγ
    # so multiplicative width is the canonical metric.
    # 'linear' (Weir 2017 canonical) — additive γ_hi − γ_lo. Tends
    # to over-reward fine partitions on data with wide high-γ tails.
    # 'relative' — additive width / midpoint.
    champ_width_metric: str = "log"

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
