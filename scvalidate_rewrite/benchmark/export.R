suppressMessages({
  library(SeuratObject)
  library(Matrix)
})
load("F:/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")

# Export counts matrix as sparse mtx (Seurat v5 Assay5)
counts <- GetAssayData(pancreas_sub, assay = "RNA", layer = "counts")
writeMM(counts, "F:/NMF_rewrite/scvalidate_rewrite/benchmark/counts.mtx")
writeLines(rownames(counts), "F:/NMF_rewrite/scvalidate_rewrite/benchmark/genes.tsv")
writeLines(colnames(counts), "F:/NMF_rewrite/scvalidate_rewrite/benchmark/cells.tsv")

# Export metadata
write.csv(pancreas_sub@meta.data,
          "F:/NMF_rewrite/scvalidate_rewrite/benchmark/metadata.csv",
          row.names = TRUE)

cat("Export complete:\n")
cat("  counts.mtx:", dim(counts)[1], "x", dim(counts)[2], "\n")
cat("  metadata.csv:", nrow(pancreas_sub@meta.data), "rows,",
    ncol(pancreas_sub@meta.data), "cols\n")
cat("  CellType levels:",
    length(unique(pancreas_sub@meta.data$CellType)), "\n")
cat("  SubCellType levels:",
    length(unique(pancreas_sub@meta.data$SubCellType)), "\n")
