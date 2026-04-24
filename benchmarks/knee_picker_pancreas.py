"""Knee-detection picker on pancreas, fine 0.01 grid.

User's design:
  - sweep r ∈ [0.01, 0.02, ..., 1.00] (100 points)
  - detect knee on conductance curve (geometric Kneedle: point with
    max perpendicular distance from the [first, last] secant line)
  - pick r_knee + offset * step  (offset = 3-5 gives visual parsimony)

On pancreas (SubCellType 8 classes), check that picker hits r near 0.20-
0.30 (ground truth region, ARI 0.50-0.65), not r=0.03 (ARI 0.26).
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import os
import time
import pickle
import numpy as np
import pandas as pd
import anndata as ad
from concurrent.futures import ProcessPoolExecutor


H5AD        = "benchmarks/out/smoke_pancreas.h5ad"
REP         = "X_pca"
LABEL_COL   = "SubCellType"
RESOLUTIONS = [round(r, 2) for r in np.arange(0.01, 1.01, 0.01)]
KNEE_OFFSET_STEPS = 3   # how many step-sizes past the knee to pick
RESOLUTION_STEP = 0.01


_WORKER_G = None


def _worker_init(gp):
    global _WORKER_G
    _WORKER_G = pickle.loads(gp)


def _leiden_worker(args):
    r, seed = args
    import scanpy as sc
    import anndata as ad_
    import scipy.sparse as sp
    G = _WORKER_G
    n = G.shape[0]
    a = ad_.AnnData(X=sp.csr_matrix((n, 1)))
    sc.tl.leiden(
        a, resolution=r, key_added="l",
        adjacency=G, flavor="igraph", n_iterations=2,
        directed=False, random_state=seed,
    )
    return r, a.obs["l"].astype(int).to_numpy()


def leiden_sweep(G, resolutions):
    gp = pickle.dumps(G)
    args = [(r, 0) for r in resolutions]
    workers = min(len(args), os.cpu_count() or 1)
    out = {}
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(gp,),
    ) as ex:
        for r, lbl in ex.map(_leiden_worker, args):
            out[r] = lbl
    return out


def _mean_conductance(G, labels):
    from fast_auto_scrna.cluster.resolution import mean_conductance
    return mean_conductance(G, labels)


def detect_knee_kneedle(x: np.ndarray, y: np.ndarray) -> int:
    """Standard geometric Kneedle — global max perpendicular distance
    from the [first, last] secant. Finds the most "bowed-out" point.
    On multi-step curves this tends to hit the middle, not the first step.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_n = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    y_n = (y - y[0]) / max(y[-1] - y[0], 1e-12)
    dist = y_n - x_n
    return int(np.argmin(dist)) if np.mean(dist) < 0 else int(np.argmax(dist))


def detect_knee_log_kneedle(x: np.ndarray, y: np.ndarray) -> int:
    """Kneedle on log10(y). Low-cond plateau gets spread in log-space,
    so the knee at the first jump becomes the global max deviation.
    """
    x = np.asarray(x, dtype=np.float64)
    y_log = np.log10(np.maximum(np.asarray(y, dtype=np.float64), 1e-9))
    x_n = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    y_n = (y_log - y_log[0]) / max(y_log[-1] - y_log[0], 1e-12)
    dist = y_n - x_n
    return int(np.argmax(dist))   # concave-above in log space


def detect_knee_first_jump(y: np.ndarray, rel_threshold: float = 0.10) -> int:
    """First index i where y[i+1] - y[i] exceeds
    ``rel_threshold × (max(y) - min(y))``. Catches the first significant
    step in a multi-step curve.
    """
    y = np.asarray(y, dtype=np.float64)
    rng = y.max() - y.min()
    if rng <= 0.0:
        return 0
    diffs = np.diff(y)
    thr = rel_threshold * rng
    for i, d in enumerate(diffs):
        if d > thr:
            return i + 1
    return int(np.argmax(diffs)) + 1


def main():
    a = ad.read_h5ad(H5AD)
    G = a.obsp["connectivities"].tocsr()

    print(f"Leiden sweep, {len(RESOLUTIONS)} resolutions "
          f"({RESOLUTIONS[0]}..{RESOLUTIONS[-1]} step {RESOLUTION_STEP})...")
    t0 = time.perf_counter()
    labels_per_res = leiden_sweep(G, RESOLUTIONS)
    print(f"  leiden wall: {time.perf_counter() - t0:.1f}s")

    rows = []
    for r in RESOLUTIONS:
        lbl = labels_per_res[r]
        k = int(len(np.unique(lbl)))
        cond = _mean_conductance(G, lbl)
        rows.append({"resolution": r, "n_clusters": k, "conductance": cond})
    df = pd.DataFrame(rows)

    x = df["resolution"].to_numpy()
    y = df["conductance"].to_numpy()

    # ARI
    from sklearn.metrics import adjusted_rand_score
    manual = a.obs[LABEL_COL].astype(str).to_numpy()
    df["ari"] = [adjusted_rand_score(manual, labels_per_res[r])
                 for r in df["resolution"]]

    detectors = {
        "kneedle_linear":    detect_knee_kneedle(x, y),
        "kneedle_log":       detect_knee_log_kneedle(x, y),
        "first_jump_10pct":  detect_knee_first_jump(y, rel_threshold=0.10),
    }

    r_ari_best = float(df.loc[df["ari"].idxmax(), "resolution"])
    k_ari_best = int(df.loc[df["ari"].idxmax(), "n_clusters"])
    ari_best   = float(df["ari"].max())

    print(f"\n{'detector':>22s}  {'knee_r':>7} {'k_knee':>6}  "
          f"{'pick_r':>7} {'k_pick':>6}  {'ARI':>7}")
    picks = {}
    for name, idx in detectors.items():
        picked_idx = min(idx + KNEE_OFFSET_STEPS, len(df) - 1)
        picks[name] = picked_idx
        print(f"  {name:>22s}  "
              f"{x[idx]:>7.2f} {int(df['n_clusters'].iloc[idx]):>6d}  "
              f"{x[picked_idx]:>7.2f} "
              f"{int(df['n_clusters'].iloc[picked_idx]):>6d}  "
              f"{float(df['ari'].iloc[picked_idx]):>7.4f}")

    print(f"\nground-truth best: r={r_ari_best:.2f} k={k_ari_best} ARI={ari_best:.4f}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(x, y, "o-", color="darkorange", markersize=3,
             label="mean conductance", alpha=0.9)
    colors = {"kneedle_linear": "steelblue",
              "kneedle_log":    "purple",
              "first_jump_10pct": "red"}
    for name, idx in detectors.items():
        r_k = x[idx]
        pick_r = x[picks[name]]
        ax1.axvline(r_k, linestyle=":", linewidth=1.0, color=colors[name],
                    label=f"knee {name}→r={r_k:.2f}")
        ax1.axvline(pick_r, linestyle="--", linewidth=0.8, color=colors[name],
                    alpha=0.55,
                    label=f"picked({name})→r={pick_r:.2f} k={int(df['n_clusters'].iloc[picks[name]])}")
    ax1.set_xlabel("leiden resolution")
    ax1.set_ylabel("mean conductance")
    ax1.set_title("Knee detectors compared — pancreas, 0.01 step, offset=3")
    ax1.legend(loc="upper left", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, df["n_clusters"], "-", color="lightsteelblue",
             alpha=0.45, linewidth=0.8)
    ax2.set_ylabel("n_clusters", color="steelblue")
    plt.tight_layout()
    out = Path("benchmarks/out/knee_picker_pancreas.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
