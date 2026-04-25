"""scVI integration route — variational autoencoder batch correction.

Wraps ``scvi-tools`` SCVI on raw counts (``adata.layers['counts']``),
restricted to HVGs (``adata.var['highly_variable']``) for atlas-scale
tractability. GPU is auto-detected via lightning's ``accelerator='auto'``;
the typical Windows dev box ships CPU-only torch — expect ~30-60 min CPU
on 222k cells × 2k HVGs × 200 epochs (vs ~2-5 min on a single GPU).

Returns the latent representation as a ``(n_cells, n_latent)`` ndarray.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def scvi_train(
    adata,
    *,
    batch_key: str = "_batch",
    n_latent: int = 30,
    n_hidden: int = 128,
    n_layers: int = 1,
    max_epochs: int | None = None,
    early_stopping: bool = True,
    gene_likelihood: str = "zinb",
    dispersion: str = "gene",
    use_hvg: bool = True,
    accelerator: str = "auto",
    batch_size: int = 128,
    seed: int = 0,
    enable_progress_bar: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train scVI on ``adata`` and return ``(latent, info)``.

    ``latent`` is a ``(n_cells, n_latent)`` float32 array suitable for
    downstream kNN / UMAP. ``info`` is a small diagnostic dict.

    Inputs assumed:
      * raw counts in ``adata.layers['counts']`` (added by Stage 02);
      * batch labels in ``adata.obs[batch_key]``;
      * if ``use_hvg`` and ``adata.var['highly_variable']`` exists,
        training is restricted to that subset.
    """
    import scvi

    scvi.settings.seed = seed

    if use_hvg and "highly_variable" in adata.var:
        mask = adata.var["highly_variable"].to_numpy(dtype=bool)
        adata_train = adata[:, mask].copy()
    else:
        adata_train = adata.copy()

    counts_layer = "counts" if "counts" in adata_train.layers else None

    scvi.model.SCVI.setup_anndata(
        adata_train,
        batch_key=batch_key,
        layer=counts_layer,
    )
    model = scvi.model.SCVI(
        adata_train,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        gene_likelihood=gene_likelihood,
        dispersion=dispersion,
    )
    train_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "accelerator": accelerator,
        "enable_progress_bar": enable_progress_bar,
    }
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    if early_stopping:
        train_kwargs["early_stopping"] = True
        # Need a periodic val check for early-stopping to fire.
        train_kwargs["check_val_every_n_epoch"] = 1
    else:
        train_kwargs["check_val_every_n_epoch"] = None

    model.train(**train_kwargs)

    latent = model.get_latent_representation().astype(np.float32, copy=False)
    actual_epochs = (
        int(model.history["elbo_train"].shape[0])
        if hasattr(model, "history") and "elbo_train" in model.history
        else -1
    )
    info = {
        "n_latent": int(n_latent),
        "n_hidden": int(n_hidden),
        "n_layers": int(n_layers),
        "max_epochs": int(max_epochs) if max_epochs is not None else None,
        "actual_epochs": actual_epochs,
        "early_stopping": bool(early_stopping),
        "gene_likelihood": str(gene_likelihood),
        "n_train_genes": int(adata_train.n_vars),
        "n_train_cells": int(adata_train.n_obs),
        "accelerator": str(accelerator),
    }
    return latent, info
