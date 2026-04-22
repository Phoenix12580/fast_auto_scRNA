#!/usr/bin/env Rscript
# Convert Seurat .qs → sparse counts HDF5 + metadata CSV.
#
# SeuratDisk is broken against SeuratObject 5.0+ (internally uses the
# now-defunct `slot=` argument). Bypass it entirely: dump dgCMatrix
# components + obs dataframe to files, let Python reassemble via scanpy.
#
# Run (Windows):
#   Rscript.exe F:/NMF_rewrite/fast_auto_scRNA_v1/scripts/convert_qs_to_h5ad.R

suppressPackageStartupMessages({
  library(qs)
  library(Matrix)
  library(hdf5r)
  library(data.table)
})

IN   <- "F:/NMF_rewrite/StepF.All_Cells_Seurat_Object.qs"
OUT_H5  <- "F:/NMF_rewrite/_stepf_conv/counts_csc.h5"
OUT_OBS <- "F:/NMF_rewrite/_stepf_conv/obs.csv"
OUT_GENES <- "F:/NMF_rewrite/_stepf_conv/genes.txt"
OUT_CELLS <- "F:/NMF_rewrite/_stepf_conv/cells.txt"
dir.create(dirname(OUT_H5), showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[conv] reading %s ...\n", IN))
t0 <- Sys.time()
obj <- qread(IN)
cat(sprintf("[conv] loaded Seurat in %.1fs (cells=%d)\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            ncol(obj)))

# Seurat 5 accessor — returns dgCMatrix in CSC form (genes × cells)
counts <- obj[["RNA"]]$counts
stopifnot(inherits(counts, "dgCMatrix"))
cat(sprintf("[conv] counts: %d genes × %d cells, nnz=%d (%.2f%% sparse)\n",
            nrow(counts), ncol(counts), length(counts@x),
            100 * (1 - length(counts@x) / (as.numeric(nrow(counts)) * ncol(counts)))))

# Write CSC triplet to HDF5
t0 <- Sys.time()
if (file.exists(OUT_H5)) file.remove(OUT_H5)
h5 <- H5File$new(OUT_H5, mode = "w")
h5$create_dataset("data",    counts@x, dtype = h5types$H5T_NATIVE_FLOAT)
h5$create_dataset("indices", counts@i, dtype = h5types$H5T_NATIVE_INT32)
h5$create_dataset("indptr",  counts@p, dtype = h5types$H5T_NATIVE_INT32)
h5$create_dataset("shape",   counts@Dim, dtype = h5types$H5T_NATIVE_INT32)
h5$close_all()
cat(sprintf("[conv] HDF5 triplet write: %.1fs → %s (%.2f GB)\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            OUT_H5, file.info(OUT_H5)$size / 1e9))

# Write obs, gene names, cell barcodes
t0 <- Sys.time()
fwrite(cbind(cell = colnames(counts), obj@meta.data), OUT_OBS)
writeLines(rownames(counts), OUT_GENES)
writeLines(colnames(counts), OUT_CELLS)
cat(sprintf("[conv] obs/genes/cells write: %.1fs\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

cat(sprintf("[conv] done. Run Python assembly next:\n"))
cat(sprintf("  python F:/NMF_rewrite/fast_auto_scRNA_v1/scripts/assemble_h5ad.py\n"))
