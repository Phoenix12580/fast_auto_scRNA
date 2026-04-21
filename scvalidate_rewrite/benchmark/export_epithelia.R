# Export F:/NMF_rewrite/seurat_epithelia.qs to the benchmark format used by
# scvalidate: counts.mtx (genes x cells), genes.tsv, cells.tsv, metadata.csv.
# Writes into benchmark/epithelia/ to keep it separate from pancreas_sub.
suppressMessages({
  library(qs); library(SeuratObject); library(Matrix)
})

out_dir <- "F:/NMF_rewrite/scvalidate_rewrite/benchmark/epithelia"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading seurat_epithelia.qs (this file is ~1 GB, may take a minute)\n")
t0 <- Sys.time()
obj <- qs::qread("F:/NMF_rewrite/seurat_epithelia.qs")
cat(sprintf("  loaded in %.1fs, class: %s\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            paste(class(obj), collapse = ",")))

cat("Object summary:\n")
print(obj)
cat("Assays:", paste(names(obj@assays), collapse=", "), "\n")

# Determine which assay holds raw counts
assay_name <- DefaultAssay(obj)
cat(sprintf("Default assay: %s\n", assay_name))

# Pull counts. For Assay5 use layer="counts"; for legacy Assay use slot="counts".
ay <- obj[[assay_name]]
cls <- class(ay)[1]
cat(sprintf("Assay class: %s\n", cls))
# SeuratObject 5.x deprecates `slot`; use `layer` for both Assay and Assay5.
counts <- GetAssayData(obj, assay = assay_name, layer = "counts")
cat(sprintf("counts: %d genes x %d cells, class=%s\n",
            nrow(counts), ncol(counts), paste(class(counts), collapse=",")))

# Metadata
meta <- obj@meta.data
cat("Meta columns:\n"); print(colnames(meta))
cat(sprintf("n_cells_meta = %d\n", nrow(meta)))

# Write outputs
cat("Writing counts.mtx ...\n")
Matrix::writeMM(as(counts, "CsparseMatrix"),
                file.path(out_dir, "counts.mtx"))
writeLines(rownames(counts), file.path(out_dir, "genes.tsv"))
writeLines(colnames(counts), file.path(out_dir, "cells.tsv"))
write.csv(meta, file.path(out_dir, "metadata.csv"), row.names = TRUE)

cat("\nDone. Files in:\n")
cat(list.files(out_dir, full.names = TRUE), sep = "\n")
