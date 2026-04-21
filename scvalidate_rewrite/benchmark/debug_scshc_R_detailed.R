# Dump R's observed + 10 null Ward stats for the same root split Python uses,
# so we can tell which step in our Python port drifts.
suppressMessages({
  library(SeuratObject); library(Matrix); library(scSHC); library(scry)
})

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
counts <- as(counts, "CsparseMatrix")

cl_df <- read.csv("F:/NMF_rewrite/scvalidate_rewrite/benchmark/scvalidate_clusters.csv",
                  stringsAsFactors = FALSE, row.names = 1)
cells <- colnames(counts)
leiden <- as.integer(cl_df[cells, "leiden"])

ids1 <- which(leiden %in% c(0L, 1L, 2L, 4L, 5L, 6L))
ids2 <- which(leiden %in% c(3L, 7L))
cat("ids1:", length(ids1), " ids2:", length(ids2), "\n")

dev <- scry::devianceFeatureSelection(counts)
var.genes <- rownames(counts)[order(dev, decreasing = TRUE)[1:2000]]
batch <- rep("1", ncol(counts))

# Pull scSHC internals into this env
rd   <- scSHC:::reduce_dimension
wl   <- scSHC:::ward_linkage
pds  <- scSHC:::poisson_dispersion_stats
fm   <- scSHC:::fit_model
gns  <- scSHC:::generate_null_statistic

cell1s <- counts[, ids1]
cell2s <- counts[, ids2]
true <- cbind(cell1s, cell2s)
batch_sub <- batch[c(ids1, ids2)]
labs <- c(rep(1, length(ids1)), rep(2, length(ids2)))

set.seed(0)
gm <- rd(true[var.genes, ], batch_sub, 30)
gm_sub.x <- gm[[2]]
cat("R reduce_dimension projection: ", nrow(gm_sub.x), "x", ncol(gm_sub.x), "\n")
cat("R proj col means (first 5): ", round(colMeans(gm_sub.x)[1:5], 4), "\n")
cat("R proj abs range: [", round(min(abs(gm_sub.x)), 4), ",",
    round(max(abs(gm_sub.x)), 4), "]\n")
cat("R proj col-sd (first 5): ", round(apply(gm_sub.x, 2, sd)[1:5], 4), "\n")

obs_stat <- wl(gm_sub.x, labs)
cat(sprintf("R observed Ward stat: %.4f\n", obs_stat))

phi <- pds(true[var.genes, ])
check_means <- matrixStats::rowMins(sapply(unique(batch_sub), function(b)
  Matrix::rowSums(true[var.genes, batch_sub == b])))
on_genes <- which(pnorm(phi, lower.tail = FALSE) < 0.05 & check_means != 0)
cat("R on-genes:", length(on_genes), "\n")

params <- fm(true[var.genes, ], on_genes, batch_sub, 30)
cat("R fit_model: lambdas[[1]] head=", round(params[[1]][[1]][1:5], 4),
    " mus[[1]] head=", round(params[[2]][[1]][1:5], 4), "\n")

set.seed(0)
null_vals <- sapply(1:10, function(i)
  gns(true[var.genes, ], params, on_genes, batch_sub, 30, gm, labs, TRUE))
cat("R null stats (10 draws):\n")
print(round(null_vals, 4))
cat(sprintf("R null mean=%.4f sd=%.4f\n",
    mean(null_vals), sd(null_vals)))
cat(sprintf("R z = (obs - null_mean)/null_sd = %.4f\n",
    (obs_stat - mean(null_vals)) / sd(null_vals)))
cat(sprintf("R 1-pnorm(obs, mu, sd) = %.4f\n",
    1 - pnorm(obs_stat, mean = mean(null_vals), sd = sd(null_vals))))
