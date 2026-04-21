# R reference on the epithelia subsample — scSHC + ROGUE + recall.
# Reads the exact same cell barcodes used by Python (subsample_cells.txt) so
# every module sees an identical input.
suppressMessages({
  library(qs); library(SeuratObject); library(Seurat); library(Matrix)
  library(scSHC); library(tibble); library(ROGUE); library(recall)
})

BENCH <- "F:/NMF_rewrite/scvalidate_rewrite/benchmark/epithelia"
cat("=== R reference benchmark on epithelia subsample ===\n")

# -------- patch mclapply in scSHC & parallel (Windows) ----------------------
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
# Also patch inside recall if it uses future / parallel
try(assignInNamespace("mclapply", seq_mclapply, ns = "future"), silent = TRUE)

# -------- load Python subsample + baseline Leiden ---------------------------
cat("[load] full Seurat object ...\n")
t0 <- Sys.time()
obj <- qs::qread("F:/NMF_rewrite/seurat_epithelia.qs")
cat(sprintf("  %.1fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

picked_cells <- readLines(file.path(BENCH, "subsample_cells.txt"))
cat(sprintf("  subsample: %d cells\n", length(picked_cells)))

# Counts for subsample only
counts_full <- GetAssayData(obj, assay = "RNA", layer = "counts")
counts <- counts_full[, picked_cells]
cat(sprintf("  sub counts: %d genes x %d cells\n", nrow(counts), ncol(counts)))

# Baseline Leiden: use Python's labels so both ports see identical partitions
cl_df <- read.csv(file.path(BENCH, "py_clusters.csv"),
                  stringsAsFactors = FALSE, row.names = 1)
stopifnot(all(picked_cells %in% rownames(cl_df)))
leiden <- as.integer(cl_df[picked_cells, "leiden"])
names(leiden) <- picked_cells
cat("Leiden table:\n"); print(table(leiden))

timing <- list()

# ----------------------------- scSHC ---------------------------------------
cat("\n[scSHC] testClusters ...\n")
t0 <- Sys.time()
set.seed(0)
scshc_res <- testClusters(
  as(counts, "CsparseMatrix"),
  cluster_ids   = leiden,
  alpha         = 0.05,
  num_features  = 2000,
  num_PCs       = 30,
  parallel      = FALSE
)
timing$scshc <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  %.1fs. %d -> %d clusters\n", timing$scshc,
            length(unique(leiden)), length(unique(scshc_res[[1]]))))
scshc_df <- data.frame(cell = picked_cells, leiden = leiden,
                       scshc_R = scshc_res[[1]], stringsAsFactors = FALSE)
write.csv(scshc_df, file.path(BENCH, "R_scshc.csv"), row.names = FALSE)

# Extract per-leiden p-values if the scSHC return structure exposes them.
# scSHC's testClusters returns list(new_labels, tree). We reconstruct by
# running testClusters once more with trace if needed — skip for now; parity
# check is crosstab-based.

# ----------------------------- ROGUE ---------------------------------------
cat("\n[ROGUE] per-leiden ...\n")
t0 <- Sys.time()
rogue_out <- data.frame(cluster = integer(), n = integer(),
                        rogue_R = numeric(), stringsAsFactors = FALSE)
for (cid in sort(unique(leiden))) {
  mask <- leiden == cid
  if (sum(mask) < 10) { next }
  sub <- counts[, mask]
  sub_f <- matr.filter(as.matrix(sub), min.cells = 10, min.genes = 10)
  if (nrow(sub_f) < 50 || ncol(sub_f) < 10) {
    rogue_out <- rbind(rogue_out,
      data.frame(cluster = cid, n = sum(mask), rogue_R = NA_real_))
    next
  }
  ent <- SE_fun(sub_f, span = 0.5)
  r <- CalculateRogue(ent, platform = "UMI")
  cat(sprintf("  leiden %d: %d cells, rogue=%.4f\n", cid, sum(mask), r))
  rogue_out <- rbind(rogue_out,
    data.frame(cluster = cid, n = sum(mask), rogue_R = r))
}
timing$rogue <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.csv(rogue_out, file.path(BENCH, "R_rogue.csv"), row.names = FALSE)
cat(sprintf("  ROGUE total: %.1fs\n", timing$rogue))

# ----------------------------- recall --------------------------------------
cat("\n[recall] FindClustersRecall ...\n")
# Build a Seurat object with the subsample counts
sub_obj <- CreateSeuratObject(counts = counts)
t0 <- Sys.time()
recall_out <- tryCatch({
  # Installed recall 0.0.0 (patched w/o scDesign3) signature:
  # (seurat_obj, resolution_start, reduction_percentage, num_clusters_start,
  #  dims, algorithm, null_method, assay, cores, shared_memory_max, verbose)
  sub_obj <- NormalizeData(sub_obj, verbose = FALSE)
  sub_obj <- FindVariableFeatures(sub_obj, nfeatures = 2000, verbose = FALSE)
  sub_obj <- ScaleData(sub_obj, verbose = FALSE)
  sub_obj <- RunPCA(sub_obj, npcs = 30, verbose = FALSE)
  FindClustersRecall(
    sub_obj,
    resolution_start      = 0.8,
    reduction_percentage  = 0.2,
    dims                  = 1:30,
    algorithm             = "louvain",
    null_method           = "ZIP",
    verbose               = TRUE
  )
}, error = function(e) {
  cat("recall ERROR: ", conditionMessage(e), "\n")
  NULL
})
timing$recall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  recall: %.1fs\n", timing$recall))
if (!is.null(recall_out)) {
  # recall_out is expected to be a Seurat object with updated Idents
  recall_labs <- as.integer(Idents(recall_out))
  names(recall_labs) <- colnames(recall_out)
  recall_df <- data.frame(cell = picked_cells,
                          recall_R = recall_labs[picked_cells],
                          stringsAsFactors = FALSE)
  write.csv(recall_df, file.path(BENCH, "R_recall.csv"), row.names = FALSE)
  cat(sprintf("  recall n_clusters = %d\n", length(unique(recall_labs))))
} else {
  writeLines("recall failed", file.path(BENCH, "R_recall.error.txt"))
}

# -------- save timing -------------------------------------------------------
tj <- file(file.path(BENCH, "R_timing.json"), "w")
writeLines(jsonlite::toJSON(timing, auto_unbox = TRUE, pretty = TRUE), tj)
close(tj)

cat("\n=== R reference done ===\n")
cat(sprintf("  scSHC : %.1fs\n", timing$scshc))
cat(sprintf("  ROGUE : %.1fs\n", timing$rogue))
cat(sprintf("  recall: %.1fs\n", timing$recall))
cat(sprintf("  total : %.1fs\n",
            sum(unlist(timing[c("scshc","rogue","recall")]))))
