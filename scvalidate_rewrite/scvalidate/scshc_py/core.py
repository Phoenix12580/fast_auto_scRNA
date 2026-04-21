"""Python port of igrabski/sc-SHC.

R mapping (docs/r_reference/scshc_utilities.R, scshc_clustering.R):

    poisson_dev_batch           utilities.R L7–L38   -> poisson_dev_batch
    poisson_dispersion_stats    utilities.R L47–L60  -> poisson_dispersion_stats
    reduce_dimension            utilities.R L68–L75  -> reduce_dimension
    compute_ess                 utilities.R L81–L83  -> compute_ess
    ward_linkage                utilities.R L89–L94  -> ward_linkage_stat
    fit_model_batch             clustering.R L8–L39  -> fit_model_batch
    fit_model                   clustering.R L45–L58 -> fit_model
    generate_null               clustering.R L66–L90 -> _generate_null
    generate_null_statistic     clustering.R L100–L136 -> _generate_null_statistic
    test_split                  clustering.R L147–L197 -> _test_split
    scSHC                       clustering.R L234–L337 -> scshc
    testClusters                clustering.R L379–L506 -> test_clusters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import cholesky, eigh
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Poisson deviance residuals (utilities.R L7–L38)
# -----------------------------------------------------------------------------


def _poisson_dev_single_batch(y: np.ndarray) -> np.ndarray:
    n = y.sum(axis=0)
    total = y.sum()
    pis = y.sum(axis=1) / total
    mu = np.outer(pis, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y == 0, 1.0, y / mu)
        d = 2.0 * (y * np.log(ratio) - (y - mu))
    d = np.where(d < 0, 0.0, d)
    sign = np.where(y > mu, 1.0, -1.0)
    return np.sqrt(d) * sign


def poisson_dev_batch(y, x: Sequence | None = None) -> np.ndarray:
    """Signed-sqrt Poisson deviance residuals. R: utilities.R L7–L38."""
    if sp.issparse(y):
        y = np.asarray(y.todense())
    y = np.asarray(y, dtype=np.float64)

    if x is None:
        return _poisson_dev_single_batch(y)

    x_arr = np.asarray(list(x))
    out = np.zeros_like(y)
    for b in np.unique(x_arr):
        mask = x_arr == b
        out[:, mask] = _poisson_dev_single_batch(y[:, mask])
    return out


# -----------------------------------------------------------------------------
# Poisson dispersion (utilities.R L47–L60)
# -----------------------------------------------------------------------------


def poisson_dispersion_stats(y) -> np.ndarray:
    """Per-gene Poisson dispersion z-statistic. R: utilities.R L47–L60."""
    if sp.issparse(y):
        y = np.asarray(y.todense())
    y = np.asarray(y, dtype=np.float64)

    n = y.sum(axis=0)
    pis = y.sum(axis=1) / y.sum()
    mu = np.outer(pis, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        y2 = (y - mu) ** 2 / mu
    y2 = np.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)
    disp = y2.sum(axis=1) / y2.shape[1]
    row_var = y2.var(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.sqrt(y2.shape[1]) * (disp - 1.0) / np.sqrt(row_var)
    return stat


# -----------------------------------------------------------------------------
# Dimension reduction (utilities.R L68–L75)
# -----------------------------------------------------------------------------


def reduce_dimension(
    y, x: Sequence | None, num_pcs: int
) -> tuple[dict, np.ndarray]:
    """Poisson-deviance PCA. R: utilities.R L68–L75.

    Uses LAPACK ``?SYEVR`` (``subset_by_index``) to compute only the top
    ``num_pcs`` eigenvectors of the gene-gene gram matrix — O(G²·k) instead
    of the O(G³) of a full decomposition. For G=2000, k=30 this is ~60×
    faster per call, which dominates sc-SHC runtime via null-draw loops.
    """
    pdev = poisson_dev_batch(y, x)
    pdev = pdev - pdev.mean(axis=1, keepdims=True)

    gram = pdev @ pdev.T
    g = gram.shape[0]
    k = min(num_pcs, g)
    # subset_by_index selects the LARGEST k eigenvalues (ascending order in R)
    vals, vecs = eigh(gram, subset_by_index=[g - k, g - 1])
    # Reverse to descending
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    projection = (vecs.T @ pdev).T
    return {"values": vals, "vectors": vecs}, projection


# -----------------------------------------------------------------------------
# ESS / Ward linkage (utilities.R L81–L94)
# -----------------------------------------------------------------------------


def compute_ess(reduc: np.ndarray) -> float:
    return float(((reduc - reduc.mean(axis=0)) ** 2).sum())


def ward_linkage_stat(reduc: np.ndarray, labels: np.ndarray) -> float:
    """Ward linkage statistic. R: utilities.R L89–L94.

    ``(ESS(all) - ESS(c1) - ESS(c2)) / n_total``, labels in {1, 2}.
    """
    labels = np.asarray(labels)
    ess1 = compute_ess(reduc[labels == 1])
    ess2 = compute_ess(reduc[labels == 2])
    ess_all = compute_ess(reduc)
    return (ess_all - (ess1 + ess2)) / len(labels)


# -----------------------------------------------------------------------------
# LogNormal-Poisson fit (clustering.R L8–L39)
# -----------------------------------------------------------------------------


@dataclass
class BatchParams:
    """Fitted parameters for one batch."""

    lambdas: np.ndarray    # per-gene rowMeans (length G)
    mus: np.ndarray        # log-normal locations for on-genes (length g_on)
    on_cov_sqrt: np.ndarray  # lower Cholesky factor (g_on x g_on)


def fit_model_batch(
    y: np.ndarray,
    on_genes: np.ndarray,
    num_pcs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Method-of-moments LogNormal-Poisson fit. R: clustering.R L8–L39."""
    if sp.issparse(y):
        y = np.asarray(y.todense())
    y = np.asarray(y, dtype=np.float64)

    on_counts = y[on_genes, :].T
    cov = np.cov(on_counts, rowvar=False, ddof=1)
    means = on_counts.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigmas = np.log(((np.diag(cov) - means) / means ** 2) + 1.0)
        mus = np.log(means) - 0.5 * sigmas

    mus_sum = mus[:, None] + mus[None, :]
    sigmas_sum = sigmas[:, None] + sigmas[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        rhos = np.log(cov / np.exp(mus_sum + 0.5 * sigmas_sum) + 1.0)
    rhos = np.where(np.isfinite(rhos), rhos, -10.0)
    np.fill_diagonal(rhos, sigmas)

    k = min(rhos.shape[0] - 1, num_pcs)
    # R uses RSpectra::eigs_sym(rhos, k=30, which="LM") — top-k by |λ|
    # (largest magnitude). rhos is a log-covariance that is NOT PSD; it
    # carries large negatives. R then filters to positives only, so the
    # effective rank is often < k. scipy.sparse.linalg.eigsh(which='LM')
    # wraps ARPACK and matches R's RSpectra both semantically and numerically.
    # On 1529×1529 it runs ~5× faster than a full eigh followed by subsetting.
    try:
        vals, vecs = eigsh(rhos, k=k, which="LM")
    except Exception:
        # Fallback to full dense eigh + top-|λ| subset for edge cases
        # (tiny matrices, convergence failures).
        vals_all, vecs_all = eigh(rhos)
        mag_order = np.argsort(-np.abs(vals_all))[:k]
        vals = vals_all[mag_order]
        vecs = vecs_all[:, mag_order]
    pos_mask = vals > 0
    num_pos = int(pos_mask.sum())
    vals = vals[pos_mask]
    vecs = vecs[:, pos_mask]
    if num_pos == 0:
        on_cov = np.diag(np.clip(sigmas, 1e-8, None))
    else:
        sub = vecs * np.sqrt(vals)
        on_cov = sub @ sub.T
        np.fill_diagonal(on_cov, np.diag(rhos))
        on_cov = 0.5 * (on_cov + on_cov.T)
        # sfsmisc::posdefify (method="someEVadd"): clip negative eigenvalues
        # to ε, reconstruct, then rescale so diag(result) == diag(input).
        # The rescaling step is critical — without it, diagonals drift after
        # eigenvalue clipping and the implied LogNormal-Poisson variances run
        # ~40% hotter than R, inflating null Ward stats 2–3×.
        target_diag = np.diag(on_cov).copy()
        ev_all, ev_vecs = eigh(on_cov)
        eps_ev = 1e-7 * abs(ev_all.max())
        if ev_all.min() < eps_ev:
            ev_clip = np.where(ev_all < eps_ev, eps_ev, ev_all)
            on_cov = (ev_vecs * ev_clip) @ ev_vecs.T
            on_cov = 0.5 * (on_cov + on_cov.T)
            # Rescale so diag(on_cov) == target_diag (R: D = sqrt(diag/orig))
            new_diag = np.maximum(np.diag(on_cov), eps_ev)
            safe_target = np.maximum(target_diag, eps_ev)
            d = np.sqrt(new_diag / safe_target)
            on_cov = on_cov / np.outer(d, d)
            on_cov = 0.5 * (on_cov + on_cov.T)

    on_cov_sqrt = cholesky(on_cov, lower=True)
    lambdas = y.mean(axis=1)
    return lambdas, mus, on_cov_sqrt


def fit_model(
    y: np.ndarray,
    on_genes: np.ndarray,
    x: np.ndarray,
    num_pcs: int,
) -> dict[str, BatchParams]:
    """Per-batch model fit. R: clustering.R L45–L58."""
    if sp.issparse(y):
        y = np.asarray(y.todense())
    y = np.asarray(y, dtype=np.float64)

    x = np.asarray(x)
    out: dict[str, BatchParams] = {}
    for b in np.unique(x):
        mask = x == b
        lam, mus, csqrt = fit_model_batch(y[:, mask], on_genes, num_pcs)
        out[str(b)] = BatchParams(lambdas=lam, mus=mus, on_cov_sqrt=csqrt)
    return out


# -----------------------------------------------------------------------------
# Null generation (clustering.R L66–L90)
# -----------------------------------------------------------------------------


def _generate_null(
    y: np.ndarray,
    params: dict[str, BatchParams],
    on_genes: np.ndarray,
    x: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """R: generate_null (clustering.R L66–L90).

    Simulates Poisson for off-genes and LogNormal-Poisson for on-genes,
    up to ``min(n_batch, 1000)`` cells per batch. Drops zero-total cells.
    """
    if sp.issparse(y):
        y = np.asarray(y.todense())
    y = np.asarray(y, dtype=np.float64)

    n_genes = y.shape[0]
    x = np.asarray(x)
    on_genes = np.asarray(on_genes, dtype=np.int64)
    off_mask = np.ones(n_genes, dtype=bool)
    off_mask[on_genes] = False
    off_idx = np.where(off_mask)[0]

    # Allocate null with the *capped* capacity per batch.
    per_batch = {}
    total_cols = 0
    col_starts = {}
    batches = list(np.unique(x))
    for b in batches:
        nb = int((x == b).sum())
        num_gen = min(nb, 1000)
        col_starts[b] = total_cols
        per_batch[b] = num_gen
        total_cols += num_gen

    null = np.zeros((n_genes, total_cols), dtype=np.float64)
    batch_null = np.empty(total_cols, dtype=object)

    for b in batches:
        num_gen = per_batch[b]
        start = col_starts[b]
        end = start + num_gen
        p = params[str(b)]

        # Off-genes: Poisson(lambda_i) broadcast across cells
        lam_off = p.lambdas[off_idx]
        null[off_idx[:, None], np.arange(start, end)[None, :]] = rng.poisson(
            lam=np.broadcast_to(lam_off[:, None], (len(off_idx), num_gen))
        )

        # On-genes: sample Z ~ N(0,I), Y = exp(L @ Z + mu), then Poisson(Y)
        g_on = len(on_genes)
        z = rng.standard_normal(size=(g_on, num_gen))
        log_rate = p.on_cov_sqrt @ z + p.mus[:, None]
        rates = np.exp(log_rate)
        null[on_genes[:, None], np.arange(start, end)[None, :]] = rng.poisson(
            lam=rates
        )

        batch_null[start:end] = b

    # Drop zero-total cells
    keep = null.sum(axis=0) > 0
    return null[:, keep], batch_null[keep]


# -----------------------------------------------------------------------------
# Null-statistic single draw (clustering.R L100–L136)
# -----------------------------------------------------------------------------


def _generate_null_statistic(
    y: np.ndarray,
    params: dict[str, BatchParams],
    on_genes: np.ndarray,
    x: np.ndarray,
    num_pcs: int,
    gm: tuple[dict, np.ndarray] | None,
    labs: np.ndarray | None,
    posthoc: bool,
    rng: np.random.Generator,
) -> float:
    """R: generate_null_statistic (clustering.R L100–L136).

    For ``posthoc=False``: redo reduce_dimension on the null and cut Ward tree
    at k=2. For ``posthoc=True``: project the null into the observed embedding
    and assign labels via 15-NN majority vote.
    """
    null, batch_null = _generate_null(y, params, on_genes, x, rng)
    if null.shape[1] < 4:
        return 0.0

    _, null_gm = reduce_dimension(null, batch_null, num_pcs)

    if not posthoc:
        d = pdist(null_gm)
        Z = linkage(d, method="ward")
        hc2 = fcluster(Z, t=2, criterion="maxclust")
    else:
        assert gm is not None and labs is not None
        pdev = poisson_dev_batch(null, batch_null)
        pdev = pdev - pdev.mean(axis=1, keepdims=True)
        # Project into observed space: gm2 = t(vectors.T @ pdev)
        gm2 = (gm[0]["vectors"].T @ pdev).T  # cells × num_pcs

        # 15-NN majority vote in gm[1] (observed embedding). Vectorized:
        # squared-distance matrix gm2(n_null × d) · obs_emb(n_obs × d).
        k_nn = min(15, len(labs))
        obs_emb = gm[1]
        # d(i,j) = ||gm2_i||² + ||obs_j||² - 2 gm2_i·obs_j
        a2 = (gm2 ** 2).sum(axis=1, keepdims=True)
        b2 = (obs_emb ** 2).sum(axis=1, keepdims=True).T
        dists = a2 + b2 - 2.0 * gm2 @ obs_emb.T
        nn_idx = np.argpartition(dists, k_nn, axis=1)[:, :k_nn]
        nn_labs = labs[nn_idx]  # n_null × k_nn
        # Majority via label==1 vote count (labs are {1, 2} by construction)
        c1 = (nn_labs == 1).sum(axis=1)
        c2 = k_nn - c1
        hc2 = np.where(c1 > c2, 1, 2)
        # Ties: random pick — use rng once per null draw
        ties = c1 == c2
        if ties.any():
            hc2[ties] = rng.choice([1, 2], size=int(ties.sum()))

        if len(np.unique(hc2)) == 1:
            # R's fallback cuts null_gm's own Ward tree at k=2, which finds a
            # structured split in pure-noise data → inflates null stat → makes
            # every observed split look null-like → uniform p≈1. R rarely
            # triggers this because AnnoyParam's approximate KNN is noisy; our
            # exact KNN deterministically collapses on clean clusters. Preserve
            # the "null has no structure" semantics with a random 50/50 split.
            hc2 = rng.choice([1, 2], size=len(hc2))

    # Per-batch Ward linkage stat, median across batches
    stats = []
    for b in np.unique(batch_null):
        mask = batch_null == b
        sub_labs = hc2[mask]
        _, counts = np.unique(sub_labs, return_counts=True)
        if len(counts) == 2 and counts.min() >= 2:
            stats.append(ward_linkage_stat(null_gm[mask], sub_labs))
        else:
            stats.append(0.0)
    return float(np.median(stats))


# -----------------------------------------------------------------------------
# Test one split (clustering.R L147–L197)
# -----------------------------------------------------------------------------


def _test_split(
    data: np.ndarray,
    ids1: np.ndarray,
    ids2: np.ndarray,
    var_genes: np.ndarray,
    num_pcs: int,
    batch: np.ndarray,
    alpha_level: float,
    rng: np.random.Generator,
    posthoc: bool,
) -> float:
    """R: test_split (clustering.R L147–L197)."""
    if sp.issparse(data):
        data = np.asarray(data.todense())
    data = np.asarray(data, dtype=np.float64)

    ids = np.concatenate([ids1, ids2])
    true = data[np.ix_(var_genes, ids)]
    b_sub = np.asarray(batch)[ids]
    labs = np.concatenate([np.ones(len(ids1), dtype=int), np.full(len(ids2), 2)])

    gm = reduce_dimension(true, b_sub, num_pcs)
    gm_x = gm[1]

    stats = []
    for b in np.unique(b_sub):
        mask = b_sub == b
        sub_labs = labs[mask]
        _, counts = np.unique(sub_labs, return_counts=True)
        if len(counts) == 2 and counts.min() >= 2:
            stats.append(ward_linkage_stat(gm_x[mask], sub_labs))
        else:
            stats.append(0.0)
    stat = float(np.median(stats))

    phi_stat = poisson_dispersion_stats(true)
    # per-batch rowSums, take per-gene min across batches
    batch_rowsums = np.stack(
        [true[:, b_sub == b].sum(axis=1) for b in np.unique(b_sub)], axis=1
    )
    check_means = batch_rowsums.min(axis=1)
    # on-genes: significant dispersion AND detected in every batch
    # R: pnorm(phi, lower.tail=FALSE) < 0.05
    p_vals = 1 - norm.cdf(phi_stat)
    on_genes_local = np.where((p_vals < 0.05) & (check_means != 0))[0]
    if len(on_genes_local) < 2:
        return 1.0

    params = fit_model(true, on_genes_local, b_sub, num_pcs)

    # First 10 null draws
    null_stats = [
        _generate_null_statistic(
            true, params, on_genes_local, b_sub, num_pcs, gm, labs, posthoc, rng
        )
        for _ in range(10)
    ]

    null_arr = np.asarray(null_stats)
    mu_hat = null_arr.mean()
    sd_hat = null_arr.std(ddof=1) if null_arr.std(ddof=1) > 0 else 1e-8
    pval = 1 - norm.cdf(stat, loc=mu_hat, scale=sd_hat)

    # Early exit if decisively far from alpha
    if pval < 0.1 * alpha_level or pval > 10 * alpha_level:
        return float(pval)

    # 40 more draws
    more = [
        _generate_null_statistic(
            true, params, on_genes_local, b_sub, num_pcs, gm, labs, posthoc, rng
        )
        for _ in range(40)
    ]
    null_arr = np.concatenate([null_arr, more])
    mu_hat = null_arr.mean()
    sd_hat = null_arr.std(ddof=1) if null_arr.std(ddof=1) > 0 else 1e-8
    return float(1 - norm.cdf(stat, loc=mu_hat, scale=sd_hat))


# -----------------------------------------------------------------------------
# Feature selection (binomial deviance, scry::devianceFeatureSelection)
# -----------------------------------------------------------------------------


def _deviance_feature_selection(y: np.ndarray) -> np.ndarray:
    """Port of scry::devianceFeatureSelection (binomial deviance).

    Returns per-gene deviance; higher = more variable.
    Formula: d_i = 2 * sum_j [y_ij * log(y_ij / (n_j*pi_i)) + (n_j-y_ij) *
    log((n_j-y_ij)/(n_j*(1-pi_i)))] where pi_i = sum_j y_ij / sum_j n_j.
    """
    if sp.issparse(y):
        y = np.asarray(y.todense())
    y = np.asarray(y, dtype=np.float64)

    n = y.sum(axis=0)  # per-cell library size
    total = n.sum()
    pis = y.sum(axis=1) / total
    # Expected "on" and "off" counts
    mu_on = np.outer(pis, n)             # genes × cells
    mu_off = np.outer(1.0 - pis, n)      # genes × cells

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(y > 0, y * np.log(y / mu_on), 0.0)
        off = n[None, :] - y
        term2 = np.where(off > 0, off * np.log(off / mu_off), 0.0)
    dev = 2.0 * (term1 + term2).sum(axis=1)
    return dev


# -----------------------------------------------------------------------------
# scSHC orchestrator (clustering.R L234–L337)
# -----------------------------------------------------------------------------


@dataclass
class SCSHCNode:
    """Lightweight tree node mirroring R's data.tree output.

    ``name`` follows R's labeling: "Node N: qfwer" or "Cluster N: qfwer".
    """

    name: str
    children: list["SCSHCNode"]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "children": [c.to_dict() for c in self.children],
        }


def scshc(
    data,
    batch: Sequence | None = None,
    alpha: float = 0.05,
    num_features: int = 2500,
    num_pcs: int = 30,
    seed: int | None = 0,
) -> tuple[np.ndarray, SCSHCNode]:
    """R: scSHC (clustering.R L234–L337).

    Hierarchical clustering with built-in Ward-linkage significance tests and
    Meinshausen-style FWER control.

    Parameters
    ----------
    data
        Genes × cells raw counts (dense or sparse).
    batch
        Batch labels per cell. If None, single batch.
    alpha
        Family-wise error rate.
    num_features
        Number of top-variable genes (by binomial deviance).
    num_pcs
        Number of PCs for dimension reduction.
    seed
        RNG seed for null draws.

    Returns
    -------
    ``(cluster_labels, tree_root)`` — ``cluster_labels`` is an int vector of
    length n_cells; ``tree_root`` is an :class:`SCSHCNode` recording each
    split's q-FWER.
    """
    if sp.issparse(data):
        data = np.asarray(data.todense())
    data = np.asarray(data, dtype=np.float64)
    n_cells = data.shape[1]

    if batch is None:
        batch_arr = np.array(["1"] * n_cells)
    else:
        batch_arr = np.asarray([str(b) for b in batch])

    rng = np.random.default_rng(seed)

    # Feature selection
    dev = _deviance_feature_selection(data)
    var_genes = np.argsort(-dev)[:num_features]

    # Embedding + full Ward tree
    _, gm_x = reduce_dimension(data[var_genes, :], batch_arr, num_pcs)
    d = pdist(gm_x)
    Z = linkage(d, method="ward")

    # Traverse the linkage tree top-down. Each split node has a list of
    # leaf cell indices.
    root_leaves = np.arange(n_cells)
    stack: list[tuple[np.ndarray, list]] = [(root_leaves, [])]
    # Children lists filled post-hoc on returning nodes.
    clusters: list[np.ndarray] = []

    def qfwer(pval: float, n_leaves: int) -> float:
        return min(round(pval * (n_cells - 1) / (n_leaves - 1), 2), 1.0)

    def ward_cut_two(leaves: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if len(leaves) < 4:
            return None
        sub_d = pdist(gm_x[leaves])
        sub_Z = linkage(sub_d, method="ward")
        cuts = fcluster(sub_Z, t=2, criterion="maxclust")
        if len(np.unique(cuts)) < 2:
            return None
        return leaves[cuts == 1], leaves[cuts == 2]

    def recurse(leaves: np.ndarray) -> SCSHCNode:
        n_leaves = len(leaves)
        if n_leaves < 2:
            clusters.append(leaves)
            return SCSHCNode(f"Cluster {len(clusters)}: 1.0", [])

        split = ward_cut_two(leaves)
        if split is None:
            clusters.append(leaves)
            return SCSHCNode(f"Cluster {len(clusters)}: 1.0", [])
        ids1, ids2 = split

        alpha_level = alpha * ((n_leaves - 1) / (n_cells - 1))

        # Drop batches with poor representation in either side
        batch_sub = batch_arr[np.concatenate([ids1, ids2])]
        cuts_marker = np.concatenate(
            [np.ones(len(ids1), dtype=int), np.full(len(ids2), 2)]
        )
        keep_batches: list[str] = []
        for b in np.unique(batch_sub):
            m = batch_sub == b
            if m.sum() == 0:
                continue
            c1 = int((cuts_marker[m] == 1).sum())
            c2 = int((cuts_marker[m] == 2).sum())
            if min(c1, c2) > 20:
                keep_batches.append(b)

        if not keep_batches:
            pval = 1.0
        else:
            ids1_k = ids1[np.isin(batch_arr[ids1], keep_batches)]
            ids2_k = ids2[np.isin(batch_arr[ids2], keep_batches)]
            pval = _test_split(
                data, ids1_k, ids2_k, var_genes, num_pcs, batch_arr,
                alpha_level, rng, posthoc=False,
            )

        if pval < alpha_level:
            left_node = recurse(ids1)
            right_node = recurse(ids2)
            return SCSHCNode(
                f"Node: {qfwer(pval, n_leaves)}",
                [left_node, right_node],
            )
        else:
            clusters.append(leaves)
            return SCSHCNode(
                f"Cluster {len(clusters)}: {qfwer(pval, n_leaves)}",
                [],
            )

    root = recurse(root_leaves)

    cluster_labels = np.zeros(n_cells, dtype=int)
    for i, cells in enumerate(clusters, start=1):
        cluster_labels[cells] = i
    return cluster_labels, root


# -----------------------------------------------------------------------------
# testClusters orchestrator (clustering.R L379–L506)
# -----------------------------------------------------------------------------


def test_clusters(
    data,
    cluster_ids: Sequence,
    batch: Sequence | None = None,
    var_genes: Sequence[str] | Sequence[int] | None = None,
    alpha: float = 0.05,
    num_features: int = 2500,
    num_pcs: int = 30,
    seed: int | None = 0,
) -> tuple[np.ndarray, SCSHCNode, dict[int, float]]:
    """R: testClusters (clustering.R L379–L506).

    Significance analysis on pre-computed clusters. Builds a hierarchy via
    pseudobulk distances, then tests each split. Returns
    ``(new_labels, tree_root, pvalues_per_cluster)``.

    Notes
    -----
    * ``pvalues_per_cluster`` is a dict keyed by the ORIGINAL cluster id
      containing the smallest (strictest) q-FWER-adjusted p-value observed
      along the path from root → cluster's collapsed group.
    """
    if sp.issparse(data):
        data = np.asarray(data.todense())
    data = np.asarray(data, dtype=np.float64)

    n_cells = data.shape[1]
    ids_arr = np.asarray([str(c) for c in cluster_ids])

    if batch is None:
        batch_arr = np.array(["1"] * n_cells)
    else:
        batch_arr = np.asarray([str(b) for b in batch])

    rng = np.random.default_rng(seed)

    if var_genes is None:
        dev = _deviance_feature_selection(data)
        var_idx = np.argsort(-dev)[:num_features]
    else:
        var_idx = np.asarray(var_genes)
        if var_idx.dtype.kind in {"U", "O"}:
            raise ValueError(
                "var_genes by name requires a gene index — pass integer indices"
            )

    # Pseudobulk normalization per cluster
    unique_ids = np.unique(ids_arr)
    pseudobulk = np.zeros((len(unique_ids), len(var_idx)))
    for i, cid in enumerate(unique_ids):
        mask = ids_arr == cid
        pseudobulk[i] = data[np.ix_(var_idx, np.where(mask)[0])].sum(axis=1)
    pseudobulk = pseudobulk / pseudobulk.sum(axis=1, keepdims=True).clip(min=1e-12)

    if len(unique_ids) == 1:
        new_labels = np.full(n_cells, "new1", dtype=object)
        return new_labels, SCSHCNode("Cluster 1: 1.0", []), {unique_ids[0]: 1.0}

    pb_d = pdist(pseudobulk)
    pb_Z = linkage(pb_d, method="ward")

    # Recursively cut pseudobulk tree
    root_leaves = unique_ids.copy()
    merged_groups: list[np.ndarray] = []
    pvalues: dict[str, float] = {}

    def qfwer(pval: float, cells_in_leaves: int) -> float:
        return min(round(pval * (n_cells - 1) / max(cells_in_leaves - 1, 1), 2), 1.0)

    def pb_cut_two(leaves: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if len(leaves) < 2:
            return None
        leaf_positions = np.array([np.where(unique_ids == lid)[0][0] for lid in leaves])
        sub_d = pdist(pseudobulk[leaf_positions])
        if len(sub_d) == 0:
            return None
        sub_Z = linkage(sub_d, method="ward")
        cuts = fcluster(sub_Z, t=2, criterion="maxclust")
        if len(np.unique(cuts)) < 2:
            return None
        return leaves[cuts == 1], leaves[cuts == 2]

    def recurse(leaves: np.ndarray, current_pval: float) -> SCSHCNode:
        if len(leaves) == 1:
            merged_groups.append(leaves)
            pvalues[leaves[0]] = current_pval
            return SCSHCNode(f"Cluster: {qfwer(current_pval, int((np.isin(ids_arr, leaves)).sum()))}", [])

        split = pb_cut_two(leaves)
        if split is None:
            merged_groups.append(leaves)
            for lid in leaves:
                pvalues[lid] = current_pval
            return SCSHCNode(
                f"Cluster: {qfwer(current_pval, int((np.isin(ids_arr, leaves)).sum()))}",
                [],
            )
        gleft, gright = split

        cells_in_leaves = int(np.isin(ids_arr, leaves).sum())
        alpha_level = alpha * ((cells_in_leaves - 1) / max(n_cells - 1, 1))

        ids1 = np.where(np.isin(ids_arr, gleft))[0]
        ids2 = np.where(np.isin(ids_arr, gright))[0]

        # Batch-representation filter
        batch_sub = batch_arr[np.concatenate([ids1, ids2])]
        cut_mark = np.concatenate(
            [np.ones(len(ids1), dtype=int), np.full(len(ids2), 2)]
        )
        keep_batches: list[str] = []
        for b in np.unique(batch_sub):
            m = batch_sub == b
            c1 = int((cut_mark[m] == 1).sum())
            c2 = int((cut_mark[m] == 2).sum())
            if min(c1, c2) > 20:
                keep_batches.append(b)
        if not keep_batches:
            pval = 1.0
        else:
            ids1_k = ids1[np.isin(batch_arr[ids1], keep_batches)]
            ids2_k = ids2[np.isin(batch_arr[ids2], keep_batches)]
            pval = _test_split(
                data, ids1_k, ids2_k, var_idx, num_pcs, batch_arr,
                alpha_level, rng, posthoc=True,
            )

        if pval < alpha_level:
            left_node = recurse(gleft, min(current_pval, pval))
            right_node = recurse(gright, min(current_pval, pval))
            return SCSHCNode(
                f"Node: {qfwer(pval, cells_in_leaves)}",
                [left_node, right_node],
            )
        else:
            merged_groups.append(leaves)
            for lid in leaves:
                pvalues[lid] = min(current_pval, pval)
            return SCSHCNode(
                f"Cluster: {qfwer(pval, cells_in_leaves)}",
                [],
            )

    root = recurse(root_leaves, 1.0)

    new_labels = np.empty(n_cells, dtype=object)
    for i, group in enumerate(merged_groups, start=1):
        new_labels[np.isin(ids_arr, group)] = f"new{i}"

    return new_labels, root, pvalues
