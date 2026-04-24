# Convert pancreas_sub.rda (Seurat) → raw parts readable from Python.
# Writes data/pancreas_sub_parts/:
#   counts.mtx  — sparse raw counts (genes × cells)
#   obs.csv     — cell metadata (incl. CellType/SubCellType)
#   var.csv     — gene names
# Python then assembles AnnData and saves as h5ad.

suppressMessages(library(Seurat))
suppressMessages(library(Matrix))

load("F:/NMF_rewrite/pancreas_sub.rda")
obj <- pancreas_sub
DefaultAssay(obj) <- "RNA"

out_dir <- "data/pancreas_sub_parts"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

counts <- LayerData(obj, assay = "RNA", layer = "counts")
writeMM(counts, file.path(out_dir, "counts.mtx"))

write.csv(obj@meta.data, file.path(out_dir, "obs.csv"),
          row.names = TRUE, quote = TRUE)
writeLines(rownames(counts), file.path(out_dir, "var.txt"))
writeLines(colnames(counts), file.path(out_dir, "obs_names.txt"))

cat("counts:", dim(counts)[1], "genes x", dim(counts)[2], "cells\n")
cat("wrote parts to:", out_dir, "\n")
