# Call R's internal scSHC::test_split on the exact Python root split:
# ids1 = leiden ∈ {0,1,2,4,5,6} (887 cells)
# ids2 = leiden ∈ {3,7}         (113 cells)
# Expected R p-value ≈ 0.00 (R's benchmark kept 7 of 8 leaves → root split passed).
suppressMessages({
  library(SeuratObject); library(Matrix); library(scSHC)
})

# Windows sequential patch
seq_mclapply <- function(X, FUN, ..., mc.cores = 1L, mc.preschedule = TRUE,
                         mc.set.seed = TRUE, mc.silent = FALSE,
                         mc.cleanup = TRUE, mc.allow.recursive = TRUE,
                         affinity.list = NULL) lapply(X, FUN, ...)
imports_env <- parent.env(asNamespace("scSHC"))
if (exists("mclapply", envir = imports_env, inherits = FALSE)) {
  unlockBinding("mclapply", imports_env)
  assign("mclapply", seq_mclapply, envir = imports_env)
  lockBinding("mclapply", imports_env)
}
assignInNamespace("mclapply", seq_mclapply, ns = "parallel")

load("F:/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")
counts <- GetAssayData(pancreas_sub, assay = "RNA", layer = "counts")

cl_df <- read.csv("F:/NMF_rewrite/scvalidate_rewrite/benchmark/scvalidate_clusters.csv",
                  stringsAsFactors = FALSE, row.names = 1)
cells <- colnames(counts)
leiden <- as.integer(cl_df[cells, "leiden"])

# Same root split as Python
ids1 <- which(leiden %in% c(0L, 1L, 2L, 4L, 5L, 6L))
ids2 <- which(leiden %in% c(3L, 7L))
cat("ids1:", length(ids1), " ids2:", length(ids2), "\n")

# Top 2000 var genes (same as Python)
dev <- scry::devianceFeatureSelection(as(counts, "CsparseMatrix"))
var_genes <- rownames(counts)[order(dev, decreasing = TRUE)[1:2000]]
batch <- rep("1", ncol(counts))

# Call R's internal test_split directly
set.seed(0)
p <- scSHC:::test_split(
  data = as(counts, "CsparseMatrix"),
  ids1 = ids1, ids2 = ids2,
  var.genes = var_genes, num_PCs = 30, batch = batch,
  alpha_level = 0.05 * (length(c(ids1, ids2)) - 1) / (ncol(counts) - 1),
  cores = 1L, posthoc = TRUE
)
cat(sprintf("R test_split p-value for root split: %.4f\n", p))
