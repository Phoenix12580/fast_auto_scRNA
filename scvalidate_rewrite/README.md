# scvalidate

Automated single-cell clustering validation in Python.

Fuses three published R methods into a single AnnData-native pipeline:

- **recall** (lcrawlab) — knockoff artificial-variable calibration for over-clustering detection
- **sc-SHC** (Grabski) — hierarchical FWER-controlled significance testing
- **ROGUE** (Pauling Liu) — entropy-based cluster purity metric

## Design goals (priority order)

1. **Accuracy (non-negotiable)** — Python output must match R originals. verdict consistency ≥ 0.95.
2. **Automation** — one call on an AnnData + cluster labels → per-cluster report.
3. **Speed** — vectorized NumPy/SciPy core, selective Rust hotspots in v0.4.

## Status

- v0.1 — rogue_py complete; recall_py / scshc_py skeletons with R line-level TODOs
- v0.3 (current) — full Python three-piece + fuse layer. Epithelia 10k
  parity: ROGUE Pearson r=0.996 (bit-match after `matr.filter` fix);
  scSHC ARI=1.0 vs R; recall diverges from R due to
  `max_iterations=20` cap (Py=21 clusters, R=1). Biological ARI vs
  subtype: recall 0.26, scSHC 0.00 → scSHC now defaults to OFF in
  `fuse_report` (pass `enable_scshc=True` to re-enable for
  mixed-lineage data). Overall 5.9× slower than R end-to-end; recall
  is the bottleneck.
- v0.4 target — Rust knockoff sampler + pairwise W-statistic for
  recall, ≥ R parity on speed.

## Layout

```
scvalidate/
  recall_py/     # knockoff generation + iterative resolution reduction
  scshc_py/      # Poisson-deviance PCA + hierarchical + null MoM
  rogue_py/      # entropy + 3-pass loess + S-E metric
  fuse/          # two-layer fused report
  io/            # AnnData adapters
docs/r_reference/  # verbatim R source from upstream, for line-by-line porting
tests/reference/   # R-original outputs on benchmark datasets, for parity tests
benchmark/         # user-supplied ground-truth data
```

## Rationale (why these three, why fuse)

Single-method blind spots:
- recall alone tends to split small real subpopulations
- sc-SHC alone tends to collapse close clusters
- ROGUE alone is a post-hoc score, not a gate

Two-layer fusion:
- **Layer 1 (math gate)**: recall_pass AND sc-SHC_p < α — both must agree
- **Layer 2 (bio score)**: ROGUE purity + marker richness — continuous 0–1

See individual module docstrings and `docs/r_reference/` for algorithmic provenance.
