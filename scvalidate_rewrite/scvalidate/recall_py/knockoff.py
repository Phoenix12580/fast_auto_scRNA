"""ZIP / NB parameter estimation + knockoff sampling + Barber-Candès filter.

R mapping (docs/r_reference/):

    estimate_zi_poisson            recall_zip.R L16–L31  -> estimate_zip
    rzipoisson                     recall_zip.R L46–L59  -> sample_zip
    estimate_negative_binomial     recall_nb.R  L13–L186 -> estimate_nb
    get_seurat_obj_with_artifi...  recall_main.R L18–L81 -> generate_knockoff_matrix
    knockoff::knockoff.threshold   (CRAN knockoff)       -> knockoff_threshold_offset1

Speed notes:

* ZIP estimator is closed-form (Lambert W) — vectorized over genes. In R this
  is called gene-by-gene via ``lapply``. Here we batch across genes, expecting
  ~3–5× speedup even without Rust.
* NB MLE uses ``scipy.stats.nbinom.fit`` per-gene. This is the main hotspot
  and the primary target for Rust-ization in v0.4 (PyO3 + rayon over genes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import lambertw


# -----------------------------------------------------------------------------
# ZIP estimation (recall_zip.R L16–L31)
# -----------------------------------------------------------------------------


@dataclass
class ZIPParams:
    lambda_hat: float
    pi_hat: float


def estimate_zip(data: np.ndarray) -> ZIPParams:
    """Closed-form MoM-style estimator for zero-inflated Poisson.

    R: ``estimate_zi_poisson`` (recall_zip.R L16–L31).

    Derivation: gamma = x_bar / (1 - p_zero_obs);
    lambda = W_0(-gamma * exp(-gamma)) + gamma.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    if n == 0:
        raise ValueError("empty data for estimate_zip")
    num_zeros = int((data == 0).sum())
    r0 = num_zeros / n
    x_bar = float(data.mean())

    if r0 == 0.0:
        # No excess zeros → degenerate to Poisson
        return ZIPParams(lambda_hat=x_bar, pi_hat=0.0)
    if r0 == 1.0:
        # All zeros
        return ZIPParams(lambda_hat=0.0, pi_hat=1.0)

    gamma = x_bar / (1.0 - r0)
    # lamW::lambertW0 — principal branch
    w = lambertw(-gamma * np.exp(-gamma), k=0).real
    lambda_hat = float(w + gamma)
    # Numerical guards: lambda_hat must be finite and >= x_bar (pi_hat ≥ 0)
    if not np.isfinite(lambda_hat) or lambda_hat < x_bar:
        # Degenerate — fall back to Poisson with observed mean.
        return ZIPParams(lambda_hat=x_bar, pi_hat=0.0)
    pi_hat = float(1.0 - x_bar / lambda_hat) if lambda_hat > 0 else 0.0
    pi_hat = float(np.clip(pi_hat, 0.0, 1.0))
    return ZIPParams(lambda_hat=lambda_hat, pi_hat=pi_hat)


def sample_zip(n: int, lambda_hat: float, pi_hat: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from ZIP(lambda, pi).

    R: ``rzipoisson`` (recall_zip.R L46–L59). R's per-sample for-loop is
    replaced by vectorized masking here (~10× faster without Rust).
    """
    lam = lambda_hat if np.isfinite(lambda_hat) and lambda_hat > 0 else 0.0
    pi = pi_hat if np.isfinite(pi_hat) else 0.0
    pi = float(np.clip(pi, 0.0, 1.0))
    zero_mask = rng.random(n) < pi
    out = rng.poisson(lam=lam, size=n)
    out[zero_mask] = 0
    return out.astype(np.int64)


# -----------------------------------------------------------------------------
# NB estimation (recall_nb.R L13–L186)
# -----------------------------------------------------------------------------


@dataclass
class NBParams:
    size: float  # R's "size" = dispersion n (nbinom of r successes)
    mu: float


def _nb_nll(params, data):
    size, mu = params
    if size <= 0 or mu <= 0:
        return np.inf
    # scipy parameterization: nbinom(n, p) where p = size/(size+mu)
    p = size / (size + mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -stats.nbinom.logpmf(data, size, p).sum()


def _nb_mme(data: np.ndarray) -> NBParams | None:
    """Method of moments fallback."""
    m = data.mean()
    v = data.var(ddof=1)
    if m <= 0 or v <= m:
        return None
    size = m * m / (v - m)
    return NBParams(size=float(size), mu=float(m))


def estimate_nb(data: np.ndarray, verbose: bool = False) -> NBParams:
    """Negative binomial parameter fit with MLE → MME fallback chain.

    R: ``estimate_negative_binomial`` (recall_nb.R L13–L186). R tries six
    methods in order (MLE Nelder-Mead, MLE default, MME, MSE, QME, MGE);
    we condense to MLE → MME since scipy exposes these cleanly. If both
    fail we raise, matching R's final ``stop()``.
    """
    data = np.asarray(data, dtype=np.int64)
    if data.size == 0:
        raise ValueError("empty data for estimate_nb")

    # MLE via scipy.optimize — parameterize (size>0, mu>0). Use MME as init.
    mme = _nb_mme(data)
    if mme is not None:
        from scipy.optimize import minimize

        res = minimize(
            _nb_nll,
            x0=[mme.size, mme.mu],
            args=(data,),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500},
        )
        if res.success and np.all(np.isfinite(res.x)) and res.x[0] > 0 and res.x[1] > 0:
            return NBParams(size=float(res.x[0]), mu=float(res.x[1]))
        if verbose:
            print("[estimate_nb] MLE failed, falling back to MME")
        return mme

    raise RuntimeError(
        "All negative binomial estimation methods failed (data may be constant)."
    )


def sample_nb(n: int, size: float, mu: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from NB(size, mu).

    R: ``stats::rnbinom(n, size=size, mu=mu)``. numpy's ``negative_binomial``
    uses (n_success, p) → convert: p = size / (size + mu).
    """
    p = size / (size + mu)
    return rng.negative_binomial(n=size, p=p, size=n).astype(np.int64)


# -----------------------------------------------------------------------------
# Knockoff matrix generator (recall_main.R L18–L81 subset)
# -----------------------------------------------------------------------------


def generate_knockoff_matrix(
    counts_gxc,
    null_method: str = "ZIP",
    seed: int | None = None,
    verbose: bool = False,
):
    """Generate a genes × cells knockoff matrix matching the input distribution.

    R: ``get_seurat_obj_with_artificial_variables`` (recall_main.R L18–L81),
    restricted to null_method in {"ZIP", "NB"} (copula variants are skipped
    per plan — they don't scale past 50K cells).

    Parameters
    ----------
    counts_gxc
        Genes × cells raw counts (sparse or dense).
    null_method
        "ZIP" (default) or "NB".
    seed
        Random seed for reproducibility.

    Returns
    -------
    knockoff_gxc
        Same shape/dtype as input, with per-gene samples drawn from the
        fitted null distribution.
    """
    rng = np.random.default_rng(seed)
    if sp.issparse(counts_gxc):
        counts_dense = np.asarray(counts_gxc.todense())
    else:
        counts_dense = np.asarray(counts_gxc)

    n_genes, n_cells = counts_dense.shape
    out = np.zeros_like(counts_dense, dtype=np.int64)

    for i in range(n_genes):
        row = counts_dense[i]
        if null_method == "ZIP":
            params = estimate_zip(row)
            out[i] = sample_zip(n_cells, params.lambda_hat, params.pi_hat, rng)
        elif null_method == "NB":
            try:
                params = estimate_nb(row, verbose=verbose)
                out[i] = sample_nb(n_cells, params.size, params.mu, rng)
            except RuntimeError:
                # Degenerate gene — fall back to Poisson with observed mean
                out[i] = rng.poisson(lam=row.mean(), size=n_cells).astype(np.int64)
        else:
            raise ValueError(
                f"null_method={null_method!r} not supported in Python port. "
                f"Use 'ZIP' or 'NB'; copula variants are excluded per plan."
            )

    return out


# -----------------------------------------------------------------------------
# Knockoff threshold (Barber-Candès offset=1)
# -----------------------------------------------------------------------------


try:
    import scvalidate_rust as _rust
    _RUST_KNOCKOFF_THRESHOLD = _rust.knockoff_threshold_offset1
except ImportError:
    _RUST_KNOCKOFF_THRESHOLD = None


def knockoff_threshold_offset1(w: np.ndarray, fdr: float = 0.05) -> float:
    """Barber-Candès knockoff threshold with offset=1.

    R: ``knockoff::knockoff.threshold(W, fdr=q, offset=1)``. The formula::

        T = min { t > 0 : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q }

    If no such t exists, returns ``np.inf`` (no genes selected).

    When ``scvalidate_rust`` is available, delegates to the Rust kernel
    (binary-search counting, O(N log N) per call vs the pure-Python O(T·N)).
    """
    w = np.asarray(w, dtype=np.float64)
    if _RUST_KNOCKOFF_THRESHOLD is not None:
        return float(_RUST_KNOCKOFF_THRESHOLD(w, float(fdr)))

    ts = np.sort(np.unique(np.abs(w[w != 0])))
    for t in ts:
        num = 1 + np.sum(w <= -t)
        denom = max(1, np.sum(w >= t))
        if num / denom <= fdr:
            return float(t)
    return float("inf")
