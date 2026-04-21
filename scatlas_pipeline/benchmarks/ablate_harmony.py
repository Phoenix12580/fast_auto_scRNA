"""Harmony-only ablation on 157k: compare old (max_iter=10, theta=2) vs
new (max_iter=20, theta=2) vs aggressive (max_iter=30, theta=4).

Shared prefix (load/QC/lognorm/HVG/scale/PCA) is computed once, then
three harmony runs reuse ``adata.obsm['X_pca']``. Only the harmony +
kNN + scIB portion repeats.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
import anndata as ad  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scatlas import ext as sc_ext, metrics as sc_metrics
from scatlas_pipeline.pipeline import _knn_and_fuzzy, _compute_scib_for_route, PipelineConfig
from scatlas_pipeline.pipeline import _lognorm, _qc_filter  # reuse helpers
from scatlas import pp


def run_once(adata, label: str, **harmony_kwargs) -> dict:
    t0 = time.perf_counter()
    ad_copy = adata.copy()  # harmony writes X_pca_harmony in-place
    sc_ext.harmony(ad_copy, batch_key="_batch", use_rep="X_pca",
                   key_added="X_pca_harmony", **harmony_kwargs)
    t_harmony = time.perf_counter() - t0

    converged = ad_copy.uns["harmony"].get("converged_at_iter", None)
    embed = ad_copy.obsm["X_pca_harmony"]

    # Plain kNN on corrected embedding (same as pipeline harmony route).
    dummy_batch = np.zeros(embed.shape[0], dtype=np.int32)
    t0 = time.perf_counter()
    knn, conn = _knn_and_fuzzy(
        embed, dummy_batch,
        neighbors_within_batch=15, backend="auto", metric="cosine",
    )
    t_knn = time.perf_counter() - t0

    # scIB core (silhouette off — same config as run_157k).
    from scatlas._scatlas_native.metrics import kbet_chi2 as _kb
    MAX = np.iinfo(np.uint32).max
    idx = knn["indices"].astype(np.int32)
    dst = knn["distances"].astype(np.float32).copy()
    pad = knn["indices"] == MAX
    if pad.any():
        rows = np.broadcast_to(
            np.arange(idx.shape[0], dtype=np.int32)[:, None], idx.shape,
        )
        idx = np.where(pad, rows, idx)
        dst[pad] = 0.0
    t0 = time.perf_counter()
    scib = sc_metrics.scib_score(
        idx, dst,
        batch_labels=adata.obs["_batch"].to_numpy(),
        label_labels=adata.obs["_batch"].to_numpy(),  # proxy without ground-truth
        embedding=None,
    )
    t_scib = time.perf_counter() - t0

    return {
        "label": label,
        "t_harmony": t_harmony, "t_knn": t_knn, "t_scib": t_scib,
        "converged": converged,
        "ilisi": scib["ilisi"], "kbet": scib["kbet_acceptance"],
        "graph_conn": scib["graph_connectivity"], "clisi": scib["clisi"],
        "mean": scib["mean"],
    }


def main() -> int:
    adata = ad.read_h5ad("/mnt/f/NMF_rewrite/epithelia_full.h5ad")
    print(f"loaded: {adata.n_obs} × {adata.n_vars}")

    cfg = PipelineConfig(input_h5ad="x", batch_key="orig.ident")
    adata = _qc_filter(adata, cfg)
    adata.obs["_batch"] = adata.obs["orig.ident"].astype(str)
    adata.layers["counts"] = adata.X.copy()
    adata.X = _lognorm(adata.X, 1e4)

    pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                             batch_key="_batch", layer="counts")
    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    pp.scale(ad_hvg, max_value=10)
    pp.pca(ad_hvg, n_comps="auto")
    adata.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
    print(f"shared prefix: X_pca {adata.obsm['X_pca'].shape}")

    trials = [
        ("old:max_iter=10,theta=2", dict(max_iter=10, theta=2.0)),
        ("new:max_iter=20,theta=2", dict(max_iter=20, theta=2.0)),
        ("aggressive:max_iter=30,theta=4", dict(max_iter=30, theta=4.0)),
        ("dynamic-lambda:max_iter=20", dict(max_iter=20, theta=2.0, lambda_=None)),
    ]
    results = []
    for label, kwargs in trials:
        print(f"\n── {label} ──")
        r = run_once(adata, label, **kwargs)
        print(f"  harmony {r['t_harmony']:.0f}s  knn {r['t_knn']:.0f}s  "
              f"scib {r['t_scib']:.0f}s")
        print(f"  converged @ iter {r['converged']}")
        print(f"  iLISI={r['ilisi']:.3f} kBET={r['kbet']:.3f} "
              f"cLISI={r['clisi']:.3f} graph={r['graph_conn']:.3f}  mean={r['mean']:.3f}")
        results.append(r)

    print("\n" + "=" * 70)
    print(f"{'label':<38s} {'iLISI':>7s} {'kBET':>7s} {'graph':>7s} {'iter':>6s} {'harm_s':>7s}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<38s} {r['ilisi']:>7.3f} {r['kbet']:>7.3f} "
              f"{r['graph_conn']:>7.3f} {str(r['converged']):>6s} {r['t_harmony']:>7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
