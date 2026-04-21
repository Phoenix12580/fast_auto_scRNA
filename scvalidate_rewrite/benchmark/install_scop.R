# Install SCOP and inspect its example datasets.
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# SCOP is on GitHub (zhanghao-njmu/SCOP)
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

if (!requireNamespace("SCOP", quietly = TRUE)) {
  cat("Installing SCOP from GitHub...\n")
  remotes::install_github("zhanghao-njmu/SCOP",
                          dependencies = TRUE, upgrade = "never")
}

library(SCOP)
cat("SCOP loaded.\n")
cat("Data objects shipped with SCOP:\n")
print(data(package = "SCOP")$results[, c("Item", "Title")])
