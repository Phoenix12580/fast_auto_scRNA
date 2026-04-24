"""Compare knee detectors on the pancreas conductance curve.

Dump the knee_curve from smoke_pancreas_knee.h5ad, then apply several
detection strategies side-by-side:

  - perpendicular_elbow (current prod, global max offset from secant)
  - steepest_rise_plateau (new: max 1st deriv + first drop below α × peak)
  - second_deriv_min (plateau entry = most negative 2nd derivative)
  - first_significant_jump (first jump > threshold × total range)

No Leiden reruns. Picker = knee + offset_steps.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import anndata as ad


H5AD       = "benchmarks/out/smoke_pancreas_knee.h5ad"
LABEL_COL  = "SubCellType"
OFFSET     = 3


def perpendicular_elbow(y):
    """Current prod — global max perpendicular distance from secant."""
    y = np.asarray(y, dtype=np.float64)
    L = len(y)
    if L < 3: return 0
    x1, y1 = 0.0, float(y[0])
    x2, y2 = float(L-1), float(y[-1])
    dx = x2 - x1
    dy = y2 - y1
    line_norm = max(float(np.hypot(dx, dy)), 1e-20)
    px = np.arange(L, dtype=np.float64)
    num = np.abs(dy * px - dx * y + x2 * y1 - y2 * x1)
    dist = num / line_norm
    dist[0] = 0.0; dist[-1] = 0.0
    return int(np.argmax(dist))


def steepest_rise_plateau(y, window=5, alpha=0.30):
    """Find the steepest-rise point, then first index after it where local
    slope drops below `alpha × peak_slope`. That's the plateau entry after
    the first rapid rise.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2 * window: return 0
    # local slopes via centered window
    slopes = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo >= 2:
            slopes[i] = (y[hi-1] - y[lo]) / (hi - 1 - lo)
    # mild smoothing
    k = np.ones(3) / 3.0
    slopes = np.convolve(slopes, k, mode="same")
    peak_idx = int(np.argmax(slopes))
    thr = alpha * slopes[peak_idx]
    for i in range(peak_idx, n):
        if slopes[i] < thr:
            return i
    return peak_idx


def second_deriv_min(y, smooth_window=5):
    """Plateau entry = most-negative second derivative (curve decelerates
    fastest = transition from rise to plateau)."""
    y = np.asarray(y, dtype=np.float64)
    if smooth_window > 1:
        k = np.ones(smooth_window) / smooth_window
        y = np.convolve(y, k, mode="same")
    d1 = np.gradient(y)
    d2 = np.gradient(d1)
    return int(np.argmin(d2))


def first_significant_jump(y, rel_threshold=0.05):
    """First index where single-step jump exceeds rel_threshold × range."""
    y = np.asarray(y, dtype=np.float64)
    rng = y.max() - y.min()
    if rng <= 0: return 0
    thr = rel_threshold * rng
    for i, d in enumerate(np.diff(y)):
        if d > thr:
            return i + 1
    return int(np.argmax(np.diff(y))) + 1


def fraction_rise_threshold(y, target_fraction=0.25):
    """First index where y crosses y[0] + fraction × total_range."""
    y = np.asarray(y, dtype=np.float64)
    rng = y.max() - y.min()
    if rng <= 0: return 0
    thr = y[0] + target_fraction * rng
    for i, v in enumerate(y):
        if v >= thr:
            return i
    return len(y) - 1


def perpendicular_elbow_log(y):
    """Kneedle on log10(y). Initial low plateau gets spread in log-space,
    first jump becomes the dominant deviation."""
    y = np.asarray(y, dtype=np.float64)
    L = len(y)
    if L < 3: return 0
    yl = np.log10(np.maximum(y, 1e-9))
    x1, y1 = 0.0, float(yl[0])
    x2, y2 = float(L-1), float(yl[-1])
    dx, dy = x2 - x1, y2 - y1
    line_norm = max(float(np.hypot(dx, dy)), 1e-20)
    px = np.arange(L, dtype=np.float64)
    num = np.abs(dy * px - dx * yl + x2 * y1 - y2 * x1)
    dist = num / line_norm
    dist[0] = 0.0; dist[-1] = 0.0
    return int(np.argmax(dist))


def first_plateau_after_rise(y, window=5, min_rise_ratio=0.10, low_slope_ratio=0.25):
    """Find first index where (a) we've climbed min_rise_ratio of total range
    AND (b) local slope has dropped to low_slope_ratio × max_slope_so_far.
    Interprets 'first plateau entry after the first real rise'."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < window: return 0
    slopes = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        if hi - lo >= 2:
            slopes[i] = (y[hi-1] - y[lo]) / (hi - 1 - lo)
    max_seen = 0.0
    rng = y.max() - y.min()
    if rng <= 0: return 0
    min_rise = min_rise_ratio * rng
    for i in range(n):
        if slopes[i] > max_seen:
            max_seen = slopes[i]
        if (y[i] - y[0]) >= min_rise and max_seen > 0 \
           and slopes[i] < low_slope_ratio * max_seen:
            return i
    return int(np.argmax(slopes))


DETECTORS = {
    "perp_elbow (current)":   lambda y: perpendicular_elbow(y),
    "perp_elbow_log":         lambda y: perpendicular_elbow_log(y),
    "fraction_rise_25%":      lambda y: fraction_rise_threshold(y, 0.25),
    "fraction_rise_20%":      lambda y: fraction_rise_threshold(y, 0.20),
    "fraction_rise_30%":      lambda y: fraction_rise_threshold(y, 0.30),
    "first_plateau_after_rise": lambda y: first_plateau_after_rise(
        y, window=5, min_rise_ratio=0.10, low_slope_ratio=0.25
    ),
    "first_jump_5pct":        lambda y: first_significant_jump(y, 0.05),
}


def main():
    a = ad.read_h5ad(H5AD)
    c = a.uns["knee_curve_none"]
    resolutions = np.asarray(c["resolution"])
    conds       = np.asarray(c["conductance"])
    ks          = np.asarray(c["n_clusters"])

    from sklearn.metrics import adjusted_rand_score
    manual = a.obs[LABEL_COL].astype(str).to_numpy()

    def ari_at(r):
        col = f"leiden_none_r{r:.2f}"
        if col not in a.obs.columns:
            return float("nan")
        return adjusted_rand_score(manual, a.obs[col].astype(int).to_numpy())

    print(f"{'detector':>24}  {'knee_r':>7} {'k_knee':>6}  "
          f"{'pick_r':>7} {'k_pick':>6}  {'ARI':>7}")
    for name, fn in DETECTORS.items():
        idx = fn(conds)
        pick_idx = min(idx + OFFSET, len(resolutions) - 1)
        print(f"{name:>24}  {resolutions[idx]:>7.2f} "
              f"{int(ks[idx]):>6}  "
              f"{resolutions[pick_idx]:>7.2f} {int(ks[pick_idx]):>6}  "
              f"{ari_at(resolutions[pick_idx]):>7.4f}")

    # anchor ground truth for reference
    print()
    r_grid = sorted(set(resolutions.tolist()))
    aris   = [ari_at(r) for r in r_grid]
    best_i = int(np.nanargmax(aris))
    print(f"ground-truth best (in grid): r={r_grid[best_i]:.2f} "
          f"k={int(a.obs[f'leiden_none_r{r_grid[best_i]:.2f}'].nunique())} "
          f"ARI={aris[best_i]:.4f}")


if __name__ == "__main__":
    main()
