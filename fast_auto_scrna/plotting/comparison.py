"""Cross-route comparison plots. Ported from v1 ``pipeline.compare_*`` at V2-P2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import INTEGRATION_METHODS


SCIB_BATCH_METRICS = ("ilisi", "kbet_acceptance", "batch_silhouette")
SCIB_BIO_METRICS = ("clisi", "graph_connectivity", "label_silhouette", "isolated_label")
SCIB_HOMO_METRICS = ("rogue_mean", "sccaf")
SCIB_METRIC_DISPLAY = {
    "ilisi": "iLISI",
    "kbet_acceptance": "kBET",
    "batch_silhouette": "batch ASW",
    "clisi": "cLISI",
    "graph_connectivity": "graph conn",
    "label_silhouette": "label ASW",
    "isolated_label": "iso label",
    "rogue_mean": "ROGUE",
    "sccaf": "SCCAF",
}


def compare_integration_plot(
    adata, out_path: str | Path,
    *, label_key: str | None = None, methods: tuple[str, ...] | None = None,
    point_size: float = 4.0, dpi: int = 150,
) -> Path:
    """Side-by-side UMAP grid for all integration routes present on adata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if methods is None:
        methods = tuple(
            m for m in INTEGRATION_METHODS
            if f"X_umap_{m}" in adata.obsm
        )
    if not methods:
        raise ValueError("no per-route UMAP embeddings found on adata.obsm")

    color_key = label_key or "_batch"
    if color_key not in adata.obs.columns:
        raise ValueError(f"obs[{color_key!r}] not found — can't color")
    labels = adata.obs[color_key].astype(str).to_numpy()
    uniq = np.unique(labels)
    cmap = plt.colormaps["tab20"] if len(uniq) <= 20 else plt.colormaps["gist_ncar"]

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
    for ax, method in zip(axes[0], methods):
        emb = adata.obsm[f"X_umap_{method}"]
        for i, lab in enumerate(uniq):
            mask = labels == lab
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                s=point_size, alpha=0.75,
                color=cmap(i / max(len(uniq) - 1, 1)),
                label=lab if len(uniq) <= 12 else None,
            )
        scib = adata.uns.get(f"scib_{method}", {})
        sub = " / ".join(
            f"{k[:5]}={scib[k]:.2f}"
            for k in ("ilisi", "clisi", "graph_connectivity", "mean")
            if k in scib and isinstance(scib[k], (int, float))
        )
        ax.set_title(f"{method}\n{sub}" if sub else method, fontsize=10)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if len(uniq) <= 12:
            ax.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(f"integration comparison — colored by {color_key}", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def compare_scib_heatmap(
    adata, out_path: str | Path,
    *, methods: tuple[str, ...] | None = None,
    dpi: int = 150,
) -> Path:
    """scib-benchmark + Zhang-lab-style heatmap: methods × metrics, with
    three-category summary columns (Batch / Bio / Homogeneity / Overall)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if methods is None:
        methods = tuple(
            m for m in INTEGRATION_METHODS if f"scib_{m}" in adata.uns
        )
    if not methods:
        raise ValueError("no scib_<method> entries in adata.uns")

    all_metrics = (list(SCIB_BATCH_METRICS) + list(SCIB_BIO_METRICS)
                   + list(SCIB_HOMO_METRICS))
    rows: list[list[float | None]] = []
    for m in methods:
        scib = adata.uns.get(f"scib_{m}", {})
        row = [float(scib[k]) if k in scib and isinstance(scib[k], (int, float)) else None
               for k in all_metrics]
        batch_vals = [row[i] for i, k in enumerate(all_metrics)
                      if k in SCIB_BATCH_METRICS and row[i] is not None]
        bio_vals = [row[i] for i, k in enumerate(all_metrics)
                    if k in SCIB_BIO_METRICS and row[i] is not None]
        homo_vals = [row[i] for i, k in enumerate(all_metrics)
                     if k in SCIB_HOMO_METRICS and row[i] is not None]
        batch_s = float(np.mean(batch_vals)) if batch_vals else None
        bio_s = float(np.mean(bio_vals)) if bio_vals else None
        homo_s = float(np.mean(homo_vals)) if homo_vals else None
        parts = [(0.35, batch_s), (0.45, bio_s), (0.20, homo_s)]
        usable = [(w, v) for w, v in parts if v is not None]
        if usable:
            total_w = sum(w for w, _ in usable)
            overall = sum(w * v for w, v in usable) / total_w
        else:
            overall = None
        rows.append(row + [batch_s, bio_s, homo_s, overall])

    col_labels = [SCIB_METRIC_DISPLAY[k] for k in all_metrics] + [
        "Batch", "Bio", "Homo", "Overall"
    ]
    arr = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
    n_rows, n_cols = arr.shape

    fig_w = max(7.5, 0.95 * n_cols)
    fig_h = max(2.2, 0.65 * n_rows + 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.colormaps["viridis"].with_extremes(bad="#cccccc")
    im = ax.imshow(
        np.ma.masked_invalid(arr),
        cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0,
    )

    n_individual = len(all_metrics)
    ax.axvline(n_individual - 0.5, color="white", lw=2.5)

    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if np.isnan(v):
                txt = "—"
                color = "black"
            else:
                txt = f"{v:.2f}"
                color = "white" if v < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(methods), fontsize=10)

    nb = len(SCIB_BATCH_METRICS)
    nbi = len(SCIB_BIO_METRICS)
    nh = len(SCIB_HOMO_METRICS)
    ax.text((nb - 1) / 2, -0.85, "batch mixing ↑",
            ha="center", va="bottom", fontsize=9, color="#444",
            transform=ax.transData)
    ax.text(nb + (nbi - 1) / 2, -0.85, "bio conservation ↑",
            ha="center", va="bottom", fontsize=9, color="#444",
            transform=ax.transData)
    ax.text(nb + nbi + (nh - 1) / 2, -0.85, "cluster homogeneity ↑",
            ha="center", va="bottom", fontsize=9, color="#444",
            transform=ax.transData)
    ax.text(n_individual + 1.5, -0.85, "summary",
            ha="center", va="bottom", fontsize=9, color="#444",
            transform=ax.transData)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("score (higher = better)", fontsize=9)

    ax.set_title("scIB — integration method comparison", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compare_rogue_per_cluster(
    adata, out_path: str | Path,
    *, methods: tuple[str, ...] | None = None,
    top_n: int | None = None, dpi: int = 150,
) -> Path:
    """Per-cluster ROGUE bar plot — one panel per integration route."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if methods is None:
        methods = tuple(
            m for m in INTEGRATION_METHODS
            if f"rogue_per_cluster_{m}" in adata.uns
        )
    if not methods:
        raise ValueError("no rogue_per_cluster_<method> entries in adata.uns")

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(max(5.0, 5.5 * n), 5), squeeze=False)
    for ax, m in zip(axes[0], methods):
        pc = adata.uns.get(f"rogue_per_cluster_{m}", {})
        if not pc:
            ax.text(0.5, 0.5, f"no ROGUE for {m}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        items = sorted(pc.items(), key=lambda kv: kv[1], reverse=True)
        if top_n is not None and len(items) > 2 * top_n:
            items = items[:top_n] + items[-top_n:]
        labels = [str(k) for k, _ in items]
        values = np.array([v for _, v in items], dtype=float)
        colors = np.where(
            values >= 0.85, "#2a9d8f",
            np.where(values >= 0.70, "#e9c46a", "#e76f51"),
        )
        ax.bar(range(len(values)), values, color=colors)
        ax.axhline(0.85, color="#2a9d8f", linestyle="--", alpha=0.4, lw=1)
        ax.axhline(0.70, color="#e76f51", linestyle="--", alpha=0.4, lw=1)

        scib = adata.uns.get(f"scib_{m}", {})
        rogue_mean = scib.get("rogue_mean", None)
        title = m if rogue_mean is None else f"{m}  (mean={rogue_mean:.2f}, n={len(pc)})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("cluster (sorted by ROGUE)")
        ax.set_ylabel("ROGUE purity (↑)")
        ax.set_ylim(0, 1.05)
        if len(labels) <= 20:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks([0, len(labels) // 2, len(labels) - 1])
            ax.set_xticklabels(
                [labels[0], labels[len(labels) // 2], labels[-1]],
                fontsize=7,
            )

    fig.suptitle(
        "per-cluster ROGUE — green ≥ 0.85 pure · yellow 0.70-0.85 · red < 0.70 mixed",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def scib_comparison_table(adata, methods: tuple[str, ...]) -> list[dict]:
    """Build a list-of-dicts comparison table of all routes' scIB scores."""
    rows: list[dict] = []
    for m in methods:
        key = f"scib_{m}"
        if key not in adata.uns:
            continue
        rec = adata.uns[key]
        row: dict[str, Any] = {"method": m}
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                row[k] = round(float(v), 4)
        rows.append(row)
    return rows
