# Install lcrawlab/recall on Windows.
# Previous blocker: recall's DESCRIPTION references scDesign3 via non-standard
# Remotes syntax that `remotes` does not parse. Workaround: install scDesign3
# via Bioconductor first, then install recall WITHOUT processing Remotes.
#
# Rationale for including scDesign3:
#   The copula branch of recall depends on it. We do not port the copula
#   branch in Python, but the R package needs scDesign3 to load.

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

cat("=== Installing Bioconductor deps (scDesign3 and common SC deps) ===\n")

bioc_pkgs <- c("scDesign3", "SingleCellExperiment", "SummarizedExperiment",
               "scran", "scater", "scuttle")
for (pkg in bioc_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s ...\n", pkg))
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  } else {
    cat(sprintf("  %s already installed.\n", pkg))
  }
}

cat("\n=== Installing CRAN deps ===\n")
cran_deps <- c("knockoff", "glmnet", "Seurat", "SeuratObject", "Matrix",
               "harmony", "dplyr", "tibble", "remotes", "devtools",
               "countsplit")
for (pkg in cran_deps) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s ...\n", pkg))
    install.packages(pkg)
  } else {
    cat(sprintf("  %s already installed.\n", pkg))
  }
}

cat("\n=== Installing lcrawlab/recall (skipping Remotes parsing) ===\n")
# Use install_github with dependencies=TRUE but upgrade="never" + Remotes=FALSE
# via the low-level remotes API. If the Remotes field still errors, fall back
# to downloading the tarball and installing directly.

ok <- tryCatch({
  remotes::install_github("lcrawlab/recall",
                          dependencies = TRUE,
                          upgrade = "never",
                          build = FALSE,
                          quiet = FALSE)
  TRUE
}, error = function(e) {
  cat("install_github failed: ", conditionMessage(e), "\n")
  FALSE
})

if (!ok) {
  cat("\nFalling back: clone + R CMD INSTALL manually\n")
  tmpd <- tempfile("recall_src_")
  dir.create(tmpd)
  zipfile <- file.path(tmpd, "recall.zip")
  download.file("https://github.com/lcrawlab/recall/archive/refs/heads/main.zip",
                zipfile, mode = "wb")
  unzip(zipfile, exdir = tmpd)
  pkg_path <- file.path(tmpd, "recall-main")
  # Strip the Remotes line from DESCRIPTION so install does not try to parse it
  desc_file <- file.path(pkg_path, "DESCRIPTION")
  desc_lines <- readLines(desc_file)
  in_remotes <- FALSE
  keep <- rep(TRUE, length(desc_lines))
  for (i in seq_along(desc_lines)) {
    line <- desc_lines[i]
    if (grepl("^Remotes:", line)) {
      in_remotes <- TRUE
      keep[i] <- FALSE
      next
    }
    if (in_remotes) {
      # Continuation lines start with whitespace; a new field starts at col 1.
      if (grepl("^[A-Za-z]", line)) {
        in_remotes <- FALSE
      } else {
        keep[i] <- FALSE
      }
    }
  }
  writeLines(desc_lines[keep], desc_file)
  cat("Stripped Remotes: from DESCRIPTION. Installing ...\n")
  install.packages(pkg_path, repos = NULL, type = "source")
}

cat("\n=== Checking recall install ===\n")
if (requireNamespace("recall", quietly = TRUE)) {
  cat("recall installed. Version:", as.character(packageVersion("recall")), "\n")
  cat("Exported functions:\n")
  print(head(getNamespaceExports("recall"), 20))
} else {
  cat("!! recall NOT installed\n")
  quit(status = 1)
}
