load("F:/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")
sink("F:/NMF_rewrite/scvalidate_rewrite/benchmark/inspect_out.txt")
cat("Objects loaded:\n"); print(ls())
obj_name <- setdiff(ls(), c("obj_name", "obj"))[1]
obj <- get(obj_name)
cat("\nObject name:", obj_name, "\n")
cat("Class:", class(obj), "\n")
print(obj)
cat("\nMeta columns:\n")
print(colnames(obj@meta.data))
cat("\nMeta head:\n")
print(head(obj@meta.data))
if ("CellType" %in% colnames(obj@meta.data)) {
  cat("\nCellType table:\n"); print(table(obj@meta.data$CellType))
}
if ("SubCellType" %in% colnames(obj@meta.data)) {
  cat("\nSubCellType table:\n"); print(table(obj@meta.data$SubCellType))
}
sink()
