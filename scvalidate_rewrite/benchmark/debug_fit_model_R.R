# Dump R's fit_model_batch on_cov_sqrt statistics for direct comparison.
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

load("F:/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")
counts <- GetAssayData(pancreas_sub, assay = "RNA", layer = "counts")
counts <- as(counts, "CsparseMatrix")

cl_df <- read.csv("F:/NMF_rewrite/scvalidate_rewrite/benchmark/scvalidate_clusters.csv",
                  stringsAsFactors = FALSE, row.names = 1)
cells <- colnames(counts)
leiden <- as.integer(cl_df[cells, "leiden"])

ids1 <- which(leiden %in% c(0L, 1L, 2L, 4L, 5L, 6L))
ids2 <- which(leiden %in% c(3L, 7L))

dev <- scry::devianceFeatureSelection(counts)
var.genes <- rownames(counts)[order(dev, decreasing = TRUE)[1:2000]]
batch <- rep("1", ncol(counts))

true <- cbind(counts[, ids1], counts[, ids2])
batch_sub <- batch[c(ids1, ids2)]

pds <- scSHC:::poisson_dispersion_stats
fmb <- scSHC:::fit_model_batch

phi <- pds(true[var.genes, ])
check_means <- matrixStats::rowMins(sapply(unique(batch_sub), function(b)
  Matrix::rowSums(true[var.genes, batch_sub == b])))
on_genes <- which(pnorm(phi, lower.tail = FALSE) < 0.05 & check_means != 0)

set.seed(0)
res <- fmb(true[var.genes, ], on_genes, 30)
lambdas <- res[[1]]
mus <- res[[2]]
on_cov_sqrt <- as.matrix(res[[3]])

cat("R fit_model_batch on_cov_sqrt:", nrow(on_cov_sqrt), "x",
    ncol(on_cov_sqrt), "\n")
cat("R on_cov_sqrt fro:", round(norm(on_cov_sqrt, "F"), 2), "\n")
cat("R diag range: [", round(min(diag(on_cov_sqrt)), 4), ",",
    round(max(diag(on_cov_sqrt)), 4), "]\n")
on_cov <- on_cov_sqrt %*% t(on_cov_sqrt)
cat("R implied on-gene variance: mean=", round(mean(diag(on_cov)), 4),
    " max=", round(max(diag(on_cov)), 4),
    " min=", round(min(diag(on_cov)), 4), "\n")
cat("R mus: mean=", round(mean(mus), 4),
    " range=[", round(min(mus), 4), ",", round(max(mus), 4), "]\n")

# Also: check variance of off-gene Poisson draws (should just be lambdas)
cat("R lambdas[1:5]:", round(lambdas[1:5], 4), "\n")
cat("R sum(on_genes) length:", length(on_genes), "\n")

# Dump one null draw's on-gene col sums to compare
gn <- scSHC:::generate_null
set.seed(0)
null_set <- gn(true[var.genes, ],
               list(list("1"=lambdas), list("1"=mus), list("1"=on_cov_sqrt)),
               on_genes, batch_sub)
# Actually, the list structure:
params <- list(list(lambdas), list(mus), list(on_cov_sqrt))
names(params[[1]]) <- "1"; names(params[[2]]) <- "1"; names(params[[3]]) <- "1"
set.seed(0)
null_set <- gn(true[var.genes, ], params, on_genes, batch_sub)
null_mat <- null_set[[1]]
on_col_sums <- colSums(null_mat[on_genes, ])
real_on_col_sums <- colSums(true[var.genes[on_genes], ])
cat("R real on-gene col_sum: mean=", round(mean(real_on_col_sums), 1),
    " median=", round(median(real_on_col_sums), 1), "\n")
cat("R null on-gene col_sum: mean=", round(mean(on_col_sums), 1),
    " median=", round(median(on_col_sums), 1), "\n")
