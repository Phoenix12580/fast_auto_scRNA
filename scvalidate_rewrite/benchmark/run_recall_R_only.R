# Run R recall only on the epithelia subsample (signature fix).
suppressMessages({
  library(qs); library(SeuratObject); library(Seurat); library(Matrix)
  library(recall); library(jsonlite)
})
BENCH <- "F:/NMF_rewrite/scvalidate_rewrite/benchmark/epithelia"

obj <- qs::qread("F:/NMF_rewrite/seurat_epithelia.qs")
picked_cells <- readLines(file.path(BENCH, "subsample_cells.txt"))
counts_full <- GetAssayData(obj, assay = "RNA", layer = "counts")
counts <- counts_full[, picked_cells]
rm(obj); gc()

sub_obj <- CreateSeuratObject(counts = counts)
sub_obj <- NormalizeData(sub_obj, verbose = FALSE)
sub_obj <- FindVariableFeatures(sub_obj, nfeatures = 2000, verbose = FALSE)
sub_obj <- ScaleData(sub_obj, verbose = FALSE)
sub_obj <- RunPCA(sub_obj, npcs = 30, verbose = FALSE)

cat("[recall] start\n")
t0 <- Sys.time()
recall_out <- tryCatch({
  FindClustersRecall(
    sub_obj,
    resolution_start     = 0.8,
    reduction_percentage = 0.2,
    dims                 = 1:30,
    algorithm            = "louvain",
    null_method          = "ZIP",
    verbose              = TRUE
  )
}, error = function(e) {
  cat("recall ERROR: ", conditionMessage(e), "\n")
  NULL
})
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[recall] done in %.1fs\n", elapsed))

if (!is.null(recall_out)) {
  recall_labs <- as.integer(Idents(recall_out))
  names(recall_labs) <- colnames(recall_out)
  recall_df <- data.frame(cell = picked_cells,
                          recall_R = recall_labs[picked_cells],
                          stringsAsFactors = FALSE)
  write.csv(recall_df, file.path(BENCH, "R_recall.csv"), row.names = FALSE)
  cat(sprintf("[recall] n_clusters = %d\n", length(unique(recall_labs))))
  tj <- jsonlite::fromJSON(file.path(BENCH, "R_timing.json"))
  tj$recall <- elapsed
  writeLines(jsonlite::toJSON(tj, auto_unbox = TRUE, pretty = TRUE),
             file.path(BENCH, "R_timing.json"))
}
cat("=== done ===\n")
