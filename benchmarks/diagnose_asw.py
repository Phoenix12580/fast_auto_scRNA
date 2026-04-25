"""ASW = 0 root-cause diagnosis (sprint #2, 2026-04-25).

Background: V2-P5 removed the three sklearn silhouettes (label / batch /
isolated_label) from the pipeline because they're O(N^2). Before they were
removed, ASW values were reportedly all ~0 on the 222k atlas. We now want to
confirm whether ASW = 0 was caused by:

  (1) embedding genuinely doesn't separate cell types (real biology / poor
      integration),
  (2) sklearn / our wrapper bug (e.g. wrong rescaling, wrong inputs), or
  (3) the metric just happens to return ~0 on this data shape (e.g. high-D
      embedding with overlapping convex hulls but local KNN structure intact).

Approach: re-run the three silhouettes on the latest 222k all-mode h5ad
(post-v2-P9 = v2p9 file). Use stratified sampling (>=100 cells per cell type)
to keep wall under a minute. Print mean + per-label breakdown so we can tell
whether the ~0 is global or driven by a few overlapping types.

Usage:
  python benchmarks/diagnose_asw.py \
      --h5ad benchmarks/out/smoke_222k_all_v2p9.h5ad \
      --label-key ct.main \
      --sample-per-label 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _stratified_sample(
    labels: np.ndarray, per_label: int, seed: int = 0,
) -> np.ndarray:
    """Pick up to ``per_label`` indices from each unique label."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for L in np.unique(labels):
        idx = np.where(labels == L)[0]
        if len(idx) > per_label:
            idx = rng.choice(idx, size=per_label, replace=False)
        out.append(idx)
    return np.concatenate(out)


def _label_silhouette(embedding: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Global label ASW rescaled to [0,1] + per-cell silhouette samples."""
    from sklearn.metrics import silhouette_samples

    if len(np.unique(labels)) < 2:
        return 1.0, np.full(len(labels), np.nan)
    samples = silhouette_samples(embedding, labels, metric="euclidean")
    s_mean = float(samples.mean())
    return float((s_mean + 1.0) / 2.0), samples


def _batch_silhouette(
    embedding: np.ndarray, batch_labels: np.ndarray, ct_labels: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Per-cell-type batch ASW. Returns (mean, per-type dict of 1-|s|)."""
    from sklearn.metrics import silhouette_samples

    per_type: dict[str, float] = {}
    for c in np.unique(ct_labels):
        mask = ct_labels == c
        bsub = batch_labels[mask]
        uniq, cnt = np.unique(bsub, return_counts=True)
        if len(uniq) < 2 or int(cnt.min()) < 2:
            continue
        s = silhouette_samples(embedding[mask], bsub, metric="euclidean")
        per_type[str(c)] = 1.0 - float(np.abs(s.mean()))
    if not per_type:
        return 1.0, per_type
    return float(np.mean(list(per_type.values()))), per_type


def _isolated_label_silhouette(
    embedding: np.ndarray, labels: np.ndarray, batch_labels: np.ndarray,
    iso_threshold: int | None = None,
) -> tuple[float, list[str]]:
    """scIB-style isolated-label ASW (binary in-label-vs-rest)."""
    from sklearn.metrics import silhouette_samples

    unique = np.unique(labels)
    bpl = np.array([len(np.unique(batch_labels[labels == L])) for L in unique])
    if iso_threshold is None:
        iso_threshold = int(bpl.min())
    iso = unique[bpl <= iso_threshold]
    if len(iso) == 0:
        return 1.0, []

    scores: list[float] = []
    iso_names: list[str] = []
    for L in iso:
        mask = (labels == L).astype(np.int32)
        if mask.sum() < 2 or mask.sum() == len(labels):
            continue
        s = silhouette_samples(embedding, mask, metric="euclidean")
        s_in = s[mask == 1].mean()
        scores.append((float(s_in) + 1.0) / 2.0)
        iso_names.append(str(L))
    if not scores:
        return 1.0, []
    return float(np.mean(scores)), iso_names


def diagnose_route(
    adata, route: str, label_key: str, batch_key: str,
    sample_per_label: int = 200, seed: int = 0, backend: str = "sklearn",
) -> dict:
    """Compute the 3 ASWs for a single route.

    backend='sklearn' uses stratified subsampling (200/label by default) +
    sklearn.metrics.silhouette_samples — fast (<1 min), small variance,
    diagnostic-only.

    backend='scib' uses the JAX-jitted chunked implementation from
    scib-metrics v0.5.1 on the FULL embedding — atlas-scale, semantics
    identical to sklearn ASW (paper-faithful). Wall ~minutes, no subsample.
    """
    rep_key = f"X_pca_{route}"
    if rep_key not in adata.obsm:
        print(f"  [{route}] missing {rep_key} — skipping")
        return {}
    if label_key not in adata.obs.columns:
        print(f"  [{route}] missing label_key={label_key!r} — skipping")
        return {}

    full_labels = np.asarray(adata.obs[label_key])
    full_batches = np.asarray(adata.obs[batch_key])
    full_emb = adata.obsm[rep_key]

    if backend == "scib":
        return _diagnose_route_scib(route, full_emb, full_labels, full_batches)

    idx = _stratified_sample(full_labels, sample_per_label, seed=seed)
    emb = full_emb[idx]
    lbl = full_labels[idx]
    bt = full_batches[idx]

    print(f"  [{route}] backend=sklearn  embedding={emb.shape}, "
          f"sample={len(idx)} cells, "
          f"labels={len(np.unique(lbl))}, batches={len(np.unique(bt))}")

    t0 = time.perf_counter()
    label_asw, samples = _label_silhouette(emb, lbl)
    t_label = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch_asw, per_type = _batch_silhouette(emb, bt, lbl)
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    iso_asw, iso_names = _isolated_label_silhouette(emb, lbl, bt)
    t_iso = time.perf_counter() - t0

    print(f"  [{route}] label_silhouette        = {label_asw:.4f}    "
          f"raw_mean={samples.mean():+.4f} (rescaled (raw+1)/2)  "
          f"  [t={t_label:.1f}s]")
    print(f"  [{route}] batch_silhouette        = {batch_asw:.4f}    "
          f"avg over {len(per_type)} cell types of 1-|mean(s)|  "
          f"  [t={t_batch:.1f}s]")
    print(f"  [{route}] isolated_label_asw      = {iso_asw:.4f}    "
          f"isolated labels (<=iso_threshold batches): {len(iso_names)}  "
          f"  [t={t_iso:.1f}s]")

    print(f"  [{route}] per-label silhouette breakdown (raw mean, count):")
    for L in np.unique(lbl):
        m = lbl == L
        print(f"    {str(L):20s} n={int(m.sum()):5d}  "
              f"mean_s={samples[m].mean():+.4f}  "
              f"std={samples[m].std():.4f}  "
              f"frac_pos={float((samples[m] > 0).mean()):.2f}")

    if per_type:
        print(f"  [{route}] per-cell-type batch ASW (1 = mixed, 0 = separated):")
        for c, v in sorted(per_type.items(), key=lambda kv: kv[1]):
            print(f"    {c:20s} batch_asw_within={v:.4f}")

    return {
        "backend": "sklearn",
        "label_silhouette": label_asw,
        "label_silhouette_raw_mean": float(samples.mean()),
        "batch_silhouette": batch_asw,
        "isolated_label": iso_asw,
        "iso_count": len(iso_names),
        "samples": samples,
        "labels": lbl,
        "per_type_batch": per_type,
        "sample_idx": idx,
    }


def _diagnose_route_scib(
    route: str, emb: np.ndarray, labels: np.ndarray, batches: np.ndarray,
) -> dict:
    """Full-data ASW via scib-metrics v0.5.1 (JAX-jitted, chunked)."""
    import scib_metrics as sm

    print(f"  [{route}] backend=scib-metrics v0.5.1  embedding={emb.shape}, "
          f"labels={len(np.unique(labels))}, batches={len(np.unique(batches))}")

    emb_f32 = np.ascontiguousarray(emb, dtype=np.float32)

    t0 = time.perf_counter()
    label_asw = float(sm.silhouette_label(emb_f32, labels, rescale=True))
    t_label = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch_asw = float(sm.silhouette_batch(emb_f32, labels, batches, rescale=True))
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    iso_asw = float(sm.isolated_labels(emb_f32, labels, batches, rescale=True))
    t_iso = time.perf_counter() - t0

    print(f"  [{route}] label_silhouette        = {label_asw:.4f}   [t={t_label:.1f}s]")
    print(f"  [{route}] batch_silhouette        = {batch_asw:.4f}   [t={t_batch:.1f}s]")
    print(f"  [{route}] isolated_label_asw      = {iso_asw:.4f}   [t={t_iso:.1f}s]")
    print(f"  [{route}] total ASW wall          = {t_label + t_batch + t_iso:.1f}s")

    return {
        "backend": "scib",
        "label_silhouette": label_asw,
        "label_silhouette_raw_mean": float("nan"),  # scib returns rescaled only
        "batch_silhouette": batch_asw,
        "isolated_label": iso_asw,
        "wall_seconds": t_label + t_batch + t_iso,
        "labels": labels,
        "samples": np.full(len(labels), np.nan),  # not exposed by scib
        "per_type_batch": {},
        "sample_idx": np.arange(len(labels)),
    }


def _load_obs_obsm_only(path: str):
    """Load only obs (categoricals) + obsm from h5ad via h5py.

    Sidesteps anndata version mismatches in uns (e.g. 'null' encoding for
    None-valued params not understood by older anndata reader).
    """
    import h5py
    import pandas as pd
    import anndata as ad

    with h5py.File(path, "r") as f:
        n_obs = f["obs/_index"].shape[0]
        # obs: read each column. Categoricals are stored as group with
        # 'codes' + 'categories'.
        obs_cols: dict[str, np.ndarray] = {}
        for k, v in f["obs"].items():
            if k == "_index":
                continue
            if isinstance(v, h5py.Group) and "codes" in v and "categories" in v:
                codes = v["codes"][:]
                cats = v["categories"][:]
                if cats.dtype.kind in ("S", "O"):
                    cats = np.array([
                        c.decode() if isinstance(c, bytes) else c for c in cats
                    ])
                obs_cols[k] = pd.Categorical.from_codes(codes, cats)
            elif isinstance(v, h5py.Dataset):
                arr = v[:]
                if arr.dtype.kind in ("S", "O"):
                    arr = np.array([
                        x.decode() if isinstance(x, bytes) else x for x in arr
                    ])
                obs_cols[k] = arr
        idx = f["obs/_index"][:]
        idx = np.array([x.decode() if isinstance(x, bytes) else x for x in idx])
        obs_df = pd.DataFrame(obs_cols, index=idx)

        n_vars = f["var/_index"].shape[0]
        var_idx = f["var/_index"][:]
        var_idx = np.array([x.decode() if isinstance(x, bytes) else x for x in var_idx])
        var_df = pd.DataFrame(index=var_idx)

        obsm: dict[str, np.ndarray] = {}
        if "obsm" in f:
            for k, v in f["obsm"].items():
                if isinstance(v, h5py.Dataset):
                    obsm[k] = v[:]

    adata = ad.AnnData(obs=obs_df, var=var_df)
    for k, v in obsm.items():
        adata.obsm[k] = v
    return adata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="benchmarks/out/smoke_222k_all_v2p9.h5ad")
    ap.add_argument("--label-key", default="ct.main")
    ap.add_argument("--batch-key", default="_batch")
    ap.add_argument("--routes", nargs="+", default=["none", "bbknn", "harmony"])
    ap.add_argument("--sample-per-label", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["sklearn", "scib"], default="sklearn",
                    help="sklearn = stratified subsample (fast diagnostic); "
                         "scib = scib-metrics JAX on full data (paper-faithful)")
    ap.add_argument("--plot", default="benchmarks/out/diagnose_asw.png")
    args = ap.parse_args()

    print(f"loading {args.h5ad} (obs + obsm only, via h5py) ...")
    t0 = time.perf_counter()
    adata = _load_obs_obsm_only(args.h5ad)
    print(f"  loaded {adata.n_obs} cells x {adata.n_vars} genes "
          f"in {time.perf_counter() - t0:.1f}s")
    print(f"  obs columns: {list(adata.obs.columns)[:15]}")
    print(f"  obsm keys:   {list(adata.obsm.keys())}")

    if args.label_key not in adata.obs.columns:
        print(f"\n[ERROR] label_key={args.label_key!r} not in obs. "
              f"Available: {list(adata.obs.columns)}")
        return 1
    if args.batch_key not in adata.obs.columns:
        print(f"\n[ERROR] batch_key={args.batch_key!r} not in obs. "
              f"Available: {list(adata.obs.columns)}")
        return 1

    print(f"\nlabel distribution ({args.label_key}):")
    vc = adata.obs[args.label_key].value_counts()
    for L, n in vc.items():
        print(f"  {str(L):20s} n={int(n):6d}")
    print(f"\nbatch distribution ({args.batch_key}):")
    vc = adata.obs[args.batch_key].value_counts()
    for B, n in vc.items():
        print(f"  {str(B):20s} n={int(n):6d}")

    results = {}
    for route in args.routes:
        print(f"\n{'=' * 72}\n[route] {route} (backend={args.backend})\n{'=' * 72}")
        results[route] = diagnose_route(
            adata, route, args.label_key, args.batch_key,
            sample_per_label=args.sample_per_label, seed=args.seed,
            backend=args.backend,
        )

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"{'route':10s}  {'label_asw':>10s}  {'raw_mean':>10s}  "
          f"{'batch_asw':>10s}  {'iso_asw':>10s}")
    for route, r in results.items():
        if not r:
            continue
        print(f"{route:10s}  {r['label_silhouette']:10.4f}  "
              f"{r['label_silhouette_raw_mean']:+10.4f}  "
              f"{r['batch_silhouette']:10.4f}  {r['isolated_label']:10.4f}")

    print("\nINTERPRETATION GUIDE:")
    print("  raw_mean ~ 0   AND  per-label means all ~ 0    → cause (1): no separation")
    print("  raw_mean ~ 0   BUT  per-label means split sign → cause (1b): partial sep, cancels")
    print("  raw_mean > 0.3 → embedding genuinely separates types — earlier ASW=0 was wrong")
    print("  batch_asw ~ 1  → batches well mixed within each type")
    print("  batch_asw ~ 0  → cells of the same type still cluster by batch")

    try:
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        _plot_per_label_silhouette(results, out, args.label_key)
        print(f"\n[plot] {out}")
    except Exception as e:
        print(f"[plot] failed: {type(e).__name__}: {e}")

    return 0


def _plot_per_label_silhouette(results: dict, out: Path, label_key: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    routes = [r for r, v in results.items() if v]
    if not routes:
        return

    all_labels = sorted(set().union(*[set(np.unique(results[r]["labels"])) for r in routes]))
    fig, axes = plt.subplots(1, len(routes), figsize=(4 * len(routes), 0.4 * len(all_labels) + 2),
                              sharey=True)
    if len(routes) == 1:
        axes = [axes]

    for ax, route in zip(axes, routes):
        r = results[route]
        means = []
        for L in all_labels:
            mask = r["labels"] == L
            means.append(r["samples"][mask].mean() if mask.any() else np.nan)
        means = np.asarray(means)
        colors = ["#3b82f6" if m > 0 else "#ef4444" for m in means]
        ax.barh(range(len(all_labels)), means, color=colors)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(range(len(all_labels)))
        ax.set_yticklabels([str(L) for L in all_labels], fontsize=7)
        ax.set_xlabel("per-label mean silhouette (raw)")
        ax.set_title(f"{route}\n"
                     f"label_asw={r['label_silhouette']:.3f}  "
                     f"raw={r['label_silhouette_raw_mean']:+.3f}",
                     fontsize=9)
    fig.suptitle(f"ASW per cell type ({label_key})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
