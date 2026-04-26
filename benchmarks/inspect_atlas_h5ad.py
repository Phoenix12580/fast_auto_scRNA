"""One-shot metadata dump for the v2-P12 222k atlas h5ad."""
import anndata as ad


a = ad.read_h5ad("benchmarks/out/smoke_222k_v2p10.h5ad", backed="r")
print(f"shape: {a.shape}")
print()

print("=== obs (cell metadata) ===")
for c in ["_batch", "ct.main", "ct.sub", "ct.sub.epi",
         "leiden_bbknn", "leiden_bbknn_champ",
         "leiden_harmony", "leiden_fastmnn", "leiden_scvi"]:
    if c in a.obs.columns:
        print(f"  {c:24s}  unique={a.obs[c].nunique():>4}")
print()

print("=== obsm (low-dim embeddings) ===")
for k in sorted(a.obsm.keys()):
    print(f"  {k:24s}  shape={a.obsm[k].shape}")
print()

print("=== obsp (graph connectivities) ===")
for k in sorted(a.obsp.keys()):
    g = a.obsp[k]
    print(f"  {k:24s}  shape={g.shape}  nnz={g.nnz}")
print()

print("=== uns (results/metadata) ===")
print("per-route scIB scores:")
for m in ("bbknn", "harmony", "fastmnn", "scvi"):
    sk = f"scib_{m}"
    if sk in a.uns:
        s = a.uns[sk]
        nan = float("nan")
        print(f"  {m:8s}  ilisi={s.get('ilisi', nan):.3f}  "
              f"clisi={s.get('clisi', nan):.3f}  "
              f"label_sil={s.get('label_silhouette', nan):.3f}  "
              f"batch_sil={s.get('batch_silhouette', nan):.3f}  "
              f"iso_sil={s.get('isolated_label', nan):.3f}  "
              f"mean={s.get('mean', nan):.3f}")
print()

print("picked resolutions:")
for k in sorted(a.uns.keys()):
    if "resolution" in k and not k.endswith("_source"):
        v = a.uns[k]
        if isinstance(v, (int, float)):
            print(f"  {k:40s}  {v}")
print()

print("CHAMP curve hull (bbknn):")
if "champ_curve_bbknn" in a.uns:
    cc = a.uns["champ_curve_bbknn"]
    n_part = len(cc["origin_resolution"])
    n_hull = sum(cc["on_hull"])
    picked = [i for i, p in enumerate(cc["is_picked"]) if p][0]
    print(f"  n_partitions={n_part}, n_on_hull={n_hull}")
    print(f"  picked: γ={cc['origin_resolution'][picked]:.3f}  "
          f"k={cc['n_clusters'][picked]}  "
          f"γ_admissible=[{cc['gamma_lo'][picked]:.3f}, {cc['gamma_hi'][picked]:.3f}]")
    print(f"  modularity={cc['modularity']}  width_metric={cc['width_metric']}")
print()

print("pipeline timings:")
if "fast_auto_scrna_timings" in a.uns:
    t = a.uns["fast_auto_scrna_timings"]
    total = sum(v for v in t.values() if isinstance(v, (int, float)))
    print(f"  total wall: {total:.1f}s = {total/60:.1f} min")
