# Run R's sc-SHC testClusters on the same baseline Leiden as Python,
# to check whether the 8->1 collapse is faithful to R (parity) or a port bug.
suppressMessages({
  library(SeuratObject)
  library(Matrix)
  library(scSHC)
})

# scSHC's test_split calls mclapply() via its imports (locked after library()).
# On Windows mclapply with cores>1 is unsupported, so override inside scSHC's
# namespace by unlocking the binding.
seq_mclapply <- function(X, FUN, ..., mc.cores = 1L, mc.preschedule = TRUE,
                         mc.set.seed = TRUE, mc.silent = FALSE,
                         mc.cleanup = TRUE, mc.allow.recursive = TRUE,
                         affinity.list = NULL) lapply(X, FUN, ...)
# scSHC's test_split resolves mclapply via its imports env (locked after load).
imports_env <- parent.env(asNamespace("scSHC"))
if (exists("mclapply", envir = imports_env, inherits = FALSE)) {
  unlockBinding("mclapply", imports_env)
  assign("mclapply", seq_mclapply, envir = imports_env)
  lockBinding("mclapply", imports_env)
  cat("Patched mclapply in scSHC imports env.\n")
} else {
  cat("mclapply NOT found in scSHC imports env; listing contents:\n")
  print(ls(imports_env))
}
assignInNamespace("mclapply", seq_mclapply, ns = "parallel")

load("F:/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")
counts <- GetAssayData(pancreas_sub, assay = "RNA", layer = "counts")
cells <- colnames(counts)

# Use Python's baseline Leiden labels so both ports see identical inputs
cl_df <- read.csv("F:/NMF_rewrite/scvalidate_rewrite/benchmark/scvalidate_clusters.csv",
                  stringsAsFactors = FALSE, row.names = 1)
stopifnot(all(cells %in% rownames(cl_df)))
leiden <- as.integer(cl_df[cells, "leiden"])
names(leiden) <- cells

cat("N cells:", length(leiden), "\n")
cat("Leiden table:\n"); print(table(leiden))

t0 <- Sys.time()
set.seed(0)
res <- testClusters(
  as(counts, "CsparseMatrix"),
  cluster_ids = leiden,
  alpha        = 0.05,
  num_features = 2000,
  num_PCs      = 30,
  parallel     = FALSE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("R scSHC wall-clock: %.1fs\n", elapsed))

new_labs <- res[[1]]
cat("R scSHC merged clusters: ", length(unique(leiden)), "->",
    length(unique(new_labs)), "\n")
cat("Merged label table:\n"); print(table(new_labs))

# Save for direct diff with Python output
out <- data.frame(cell = cells, leiden = leiden, scshc_R = new_labs,
                  row.names = NULL)
write.csv(out, "F:/NMF_rewrite/scvalidate_rewrite/benchmark/scshc_R.csv", row.names = FALSE)
cat("Wrote F:/NMF_rewrite/scvalidate_rewrite/benchmark/scshc_R.csv\n")
