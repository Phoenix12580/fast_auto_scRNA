"""Python port of PaulingLiu/ROGUE (R, master, R/ROGUE.R).

Ported from v1 ``scvalidate_rewrite/scvalidate/rogue_py/core.py`` at
V2-P2. Each public function maps 1:1 to an R function; line numbers
reference ``docs/r_reference/rogue_ROGUE.R`` in the v1 repo:

    Entropy             L11–L23   -> entropy_table
    entropy_fit         L38–L68   -> entropy_fit
    SE_fun              L90–L97   -> se_fun
    matr.filter         L218–L226 -> filter_matrix
    CalculateRogue      L245–L271 -> calculate_rogue
    ent.toli            L287–L310 -> _remove_top_outliers
    rogue               L331–L369 -> rogue_per_cluster
    DetermineK          L410–L419 -> determine_k

Speed notes: ``entropy_table`` uses numpy broadcasting on sparse CSR
via ``.mean(axis=1)``. ``entropy_fit`` delegates loess to
``skmisc.loess`` (R-bit-compat) when installed, else statsmodels lowess
(~1% deviation).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests


def entropy_table(
    expr, r: float = 1.0, gene_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-gene expected-vs-observed entropy table. R ``Entropy`` L11–L23."""
    n_genes = expr.shape[0]

    if sp.issparse(expr):
        log1p = expr.copy()
        log1p.data = np.log(log1p.data + 1.0)
        entropy = np.asarray(log1p.mean(axis=1)).ravel()
        row_mean = np.asarray(expr.mean(axis=1)).ravel()
    else:
        expr_arr = np.asarray(expr)
        entropy = np.log(expr_arr + 1.0).mean(axis=1)
        row_mean = expr_arr.mean(axis=1)

    mean_expr = np.log(row_mean + r)

    if gene_names is None:
        gene_names = np.arange(n_genes).astype(str)

    return pd.DataFrame(
        {"Gene": gene_names, "mean_expr": mean_expr, "entropy": entropy}
    )


def _loess_predict(
    x: np.ndarray, y: np.ndarray, x_new: np.ndarray, span: float,
) -> np.ndarray:
    """Fit loess on (x, y) and predict at x_new. Prefers ``skmisc.loess``."""
    try:
        from skmisc.loess import loess as _skmisc_loess
    except ImportError:
        warnings.warn(
            "skmisc.loess not installed — falling back to statsmodels.lowess. "
            "Install `scikit-misc` for R parity.",
            stacklevel=3,
        )
        from statsmodels.nonparametric.smoothers_lowess import lowess
        order = np.argsort(x)
        x_sorted = x[order]
        smoothed = lowess(y[order], x_sorted, frac=span, return_sorted=False)
        return np.interp(x_new, x_sorted, smoothed)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_new_clamped = np.clip(x_new, x_min, x_max)

    for trial_span in (span, min(0.75, span * 1.5), 0.9):
        try:
            lo = _skmisc_loess(x, y, span=trial_span, degree=2, family="gaussian")
            lo.fit()
            pred = lo.predict(x_new_clamped, stderror=False).values
            return np.asarray(pred)
        except ValueError as err:
            msg = str(err).lower()
            if "singular" in msg or "extrapolation" in msg:
                continue
            raise
    from statsmodels.nonparametric.smoothers_lowess import lowess
    order = np.argsort(x)
    x_sorted = x[order]
    smoothed = lowess(y[order], x_sorted, frac=min(0.9, span), return_sorted=False)
    return np.interp(x_new, x_sorted, smoothed)


def entropy_fit(
    ent: pd.DataFrame,
    span: float = 0.5,
    mt_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Fit entropy ~ mean_expr via 3-pass trimmed loess. R ``entropy_fit`` L38–L68."""
    mask = np.isfinite(ent["mean_expr"].to_numpy()) & (ent["entropy"].to_numpy() > 0)
    work = ent.loc[mask].reset_index(drop=True).copy()

    x_full = work["mean_expr"].to_numpy()
    y_full = work["entropy"].to_numpy()

    # Pass 1
    prd = _loess_predict(x_full, y_full, x_full, span=span)
    ds = prd - y_full
    with np.errstate(invalid="ignore"):
        pv = 1.0 - stats.norm.cdf(ds, loc=np.mean(ds), scale=np.std(ds, ddof=1))
    keep = pv > 0.1

    # Pass 2
    prd = _loess_predict(x_full[keep], y_full[keep], x_full, span=span)
    ds = prd - y_full
    finite = np.isfinite(ds)
    ds_f = ds[finite]
    with np.errstate(invalid="ignore"):
        pv = 1.0 - stats.norm.cdf(ds_f, loc=np.mean(ds_f), scale=np.std(ds_f, ddof=1))
    keep2 = np.zeros_like(finite, dtype=bool)
    keep2[np.where(finite)[0][pv > 0.1]] = True

    # Pass 3 — final fit on twice-trimmed points
    prd = _loess_predict(x_full[keep2], y_full[keep2], x_full, span=span)
    ds = prd - y_full
    finite = np.isfinite(ds)

    out = work.loc[finite].copy()
    out["fit"] = prd[finite]
    out["ds"] = ds[finite]
    ds_vals = out["ds"].to_numpy()
    with np.errstate(invalid="ignore"):
        out["p_value"] = 1.0 - stats.norm.cdf(
            ds_vals, loc=np.mean(ds_vals), scale=np.std(ds_vals, ddof=1)
        )
    pvals = out["p_value"].to_numpy()
    nan_mask = ~np.isfinite(pvals)
    p_adj = np.full_like(pvals, np.nan)
    if (~nan_mask).any():
        _, p_adj_valid, _, _ = multipletests(pvals[~nan_mask], method=mt_method)
        p_adj[~nan_mask] = p_adj_valid
    out["p_adj"] = p_adj
    out = out.sort_values("ds", ascending=False).reset_index(drop=True)
    return out


def se_fun(
    expr,
    span: float = 0.5,
    r: float = 1.0,
    mt_method: str = "fdr_bh",
    if_adj: bool = True,
    gene_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Entropy + entropy_fit in one call. R ``SE_fun`` L90–L97."""
    ent = entropy_table(expr, r=r, gene_names=gene_names)
    fit = entropy_fit(ent, span=span, mt_method=mt_method)
    if not if_adj:
        fit["p_adj"] = fit["p_value"]
    return fit


def filter_matrix(expr, min_cells: int = 10, min_genes: int = 10):
    """Drop low-coverage genes and cells. R ``matr.filter`` L218–L226."""
    if sp.issparse(expr):
        nnz = (expr > 0).astype(np.int32)
        cells_per_gene = np.asarray(nnz.sum(axis=1)).ravel()
        genes_per_cell = np.asarray(nnz.sum(axis=0)).ravel()
    else:
        arr = np.asarray(expr)
        cells_per_gene = (arr > 0).sum(axis=1)
        genes_per_cell = (arr > 0).sum(axis=0)

    gene_keep = cells_per_gene >= min_cells
    cell_keep = genes_per_cell >= min_genes

    if sp.issparse(expr):
        return expr[gene_keep, :][:, cell_keep]
    return expr[np.ix_(gene_keep, cell_keep)]


def calculate_rogue(
    se_table: pd.DataFrame,
    platform: Literal["UMI", "full-length"] | None = None,
    cutoff: float = 0.05,
    k: float | None = None,
    features: Sequence[str] | None = None,
) -> float:
    """Compute a single ROGUE score from an S-E table. R ``CalculateRogue`` L245–L271."""
    if k is None:
        if platform == "UMI":
            k_val = 45.0
        elif platform == "full-length":
            k_val = 500.0
        else:
            raise ValueError(
                "Must provide `k` or `platform` in {'UMI', 'full-length'}"
            )
    else:
        k_val = float(k)

    if features is not None:
        sub = se_table[se_table["Gene"].isin(list(features))]
        sig_value = np.nansum(np.abs(sub["ds"].to_numpy()))
    else:
        mask = (
            (se_table["p_adj"].to_numpy() < cutoff)
            & (se_table["p_value"].to_numpy() < cutoff)
        )
        sig_value = np.nansum(np.abs(se_table["ds"].to_numpy()[mask]))

    return 1.0 - sig_value / (sig_value + k_val)


def _remove_top_outliers(
    ent: pd.DataFrame,
    expr,
    n: int = 2,
    span: float = 0.5,
    r: float = 1.0,
    mt_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Remove top-n outlier cells per significant gene, refit. R ``ent.toli`` L287–L310."""
    sig_genes = ent.loc[ent["p_adj"] < 0.05, "Gene"].tolist()
    ng = len(sig_genes)
    if ng == 0:
        return ent

    gene_to_row = {g: i for i, g in enumerate(ent["Gene"].tolist())}
    rows = [gene_to_row[g] for g in sig_genes if g in gene_to_row]
    if sp.issparse(expr):
        sub = np.asarray(expr[rows, :].todense())
    else:
        sub = np.asarray(expr)[rows, :]

    sub_sorted = np.sort(sub, axis=1)[:, ::-1]
    trimmed = sub_sorted[:, n:]
    new_mean = np.log(trimmed.mean(axis=1) + r)
    new_entr = np.log(trimmed + 1).mean(axis=1)

    mean_cut = ent["mean_expr"].min()

    out = ent.copy()
    for g, mnew, enew in zip(sig_genes, new_mean, new_entr):
        out.loc[out["Gene"] == g, "mean_expr"] = mnew
        out.loc[out["Gene"] == g, "entropy"] = enew

    out = out.drop(columns=["p_adj"], errors="ignore")
    out = out[out["mean_expr"] > mean_cut].reset_index(drop=True)
    out = entropy_fit(out, span=span, mt_method=mt_method)
    return out


@dataclass
class RoguePerClusterResult:
    matrix: pd.DataFrame


def rogue_per_cluster(
    expr,
    labels: Sequence,
    samples: Sequence,
    platform: Literal["UMI", "full-length"] | None = None,
    k: float | None = None,
    min_cell_n: int = 10,
    remove_outlier_n: int = 2,
    span: float = 0.5,
    r: float = 1.0,
    do_filter: bool = False,
    min_cells: int = 10,
    min_genes: int = 10,
    mt_method: str = "fdr_bh",
    gene_names: Sequence[str] | None = None,
) -> RoguePerClusterResult:
    """Compute ROGUE for every (cluster × sample) cell subset. R ``rogue`` L331–L369."""
    labels = np.asarray(list(labels))
    samples = np.asarray(list(samples))
    n_cells = expr.shape[1]
    if len(labels) != n_cells or len(samples) != n_cells:
        raise ValueError(
            f"labels ({len(labels)}) / samples ({len(samples)}) must match cells ({n_cells})"
        )

    unique_clusters = pd.unique(labels)
    unique_samples = pd.unique(samples)

    matrix = pd.DataFrame(
        np.full((len(unique_samples), len(unique_clusters)), np.nan),
        index=unique_samples,
        columns=unique_clusters,
    )

    for cluster in unique_clusters:
        for sample in unique_samples:
            sel = (labels == cluster) & (samples == sample)
            if sel.sum() < min_cell_n:
                continue
            sub_expr = (
                expr[:, np.where(sel)[0]] if sp.issparse(expr) else expr[:, sel]
            )
            if do_filter:
                sub_expr = filter_matrix(
                    sub_expr, min_cells=min_cells, min_genes=min_genes,
                )
                if sub_expr.shape[0] == 0 or sub_expr.shape[1] == 0:
                    continue
                sub_genes = gene_names
            else:
                sub_genes = gene_names
            se = se_fun(
                sub_expr, span=span, r=r, mt_method=mt_method, gene_names=sub_genes,
            )
            se = _remove_top_outliers(
                se, sub_expr, n=remove_outlier_n, span=span, r=r, mt_method=mt_method,
            )
            matrix.loc[sample, cluster] = calculate_rogue(
                se, platform=platform, k=k,
            )

    return RoguePerClusterResult(matrix=matrix)


def determine_k(
    expr,
    span: float = 0.5,
    r: float = 1.0,
    mt_method: str = "fdr_bh",
    if_adj: bool = True,
    gene_names: Sequence[str] | None = None,
) -> float:
    """Data-driven K selection on a heterogeneous reference dataset. R ``DetermineK`` L410–L419."""
    ent = se_fun(
        expr, span=span, r=r, mt_method=mt_method, if_adj=if_adj, gene_names=gene_names,
    )
    sig_ds = ent.loc[ent["p_adj"] < 0.05, "ds"].sum()
    return float(sig_ds / 2.0)
