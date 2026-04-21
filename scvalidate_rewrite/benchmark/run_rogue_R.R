# R ROGUE on pancreas_sub, per leiden cluster — for direct comparison with
# the Python port's per-cluster ROGUE scores.
suppressMessages({
  library(SeuratObject); library(Matrix); library(tibble); library(ROGUE)
})

load("F:/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")
counts <- as.matrix(GetAssayData(pancreas_sub, assay="RNA", layer="counts"))
cl_df <- read.csv("F:/NMF_rewrite/scvalidate_rewrite/benchmark/scvalidate_clusters.csv",
                  stringsAsFactors=FALSE, row.names=1)
cells <- colnames(counts)
leiden <- as.character(cl_df[cells, "leiden"])

out <- data.frame(cluster=character(), rogue=numeric(),
                  stringsAsFactors=FALSE)
for (cid in sort(unique(leiden))) {
  mask <- leiden == cid
  if (sum(mask) < 10) next
  sub <- counts[, mask]
  # Filter low-coverage (defaults: 10 cells, 10 genes)
  sub_f <- matr.filter(sub, min.cells=10, min.genes=10)
  ent <- SE_fun(sub_f, span=0.5)
  r <- CalculateRogue(ent, platform="UMI")
  out <- rbind(out, data.frame(cluster=cid, rogue=r))
  cat(sprintf("R leiden %s: %d cells, ROGUE=%.4f\n", cid, sum(mask), r))
}
write.csv(out, "F:/NMF_rewrite/scvalidate_rewrite/benchmark/rogue_R.csv", row.names=FALSE)
cat("Wrote rogue_R.csv\n")
