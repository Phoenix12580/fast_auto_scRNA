# Third-attempt recall install. Uses R's DCF parser to cleanly rewrite
# DESCRIPTION (drops scDesign3 from Imports, drops Remotes entirely).
options(repos = c(CRAN = "https://cloud.r-project.org"))

for (pkg in c("lamW", "knockoff", "countsplit")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s from CRAN ...\n", pkg))
    install.packages(pkg)
  }
}

tmpd <- tempfile("recall_src_")
dir.create(tmpd)
zipfile <- file.path(tmpd, "recall.zip")
download.file("https://github.com/lcrawlab/recall/archive/refs/heads/main.zip",
              zipfile, mode = "wb")
unzip(zipfile, exdir = tmpd)
pkg_path <- file.path(tmpd, "recall-main")

desc_file <- file.path(pkg_path, "DESCRIPTION")
desc <- read.dcf(desc_file)
cat("Before patch, fields:\n")
print(colnames(desc))

# Strip scDesign3 from Imports
if ("Imports" %in% colnames(desc)) {
  imp <- desc[1, "Imports"]
  parts <- strsplit(imp, ",")[[1]]
  parts <- trimws(parts)
  parts <- parts[!grepl("^scDesign3(\\s|\\(|$)", parts)]
  desc[1, "Imports"] <- paste(parts, collapse = ",\n    ")
}
# Drop Remotes entirely
if ("Remotes" %in% colnames(desc)) {
  desc <- desc[, setdiff(colnames(desc), "Remotes"), drop = FALSE]
}

write.dcf(desc, desc_file, indent = 4, width = 80, keep.white = TRUE)
cat("\n--- PATCHED DESCRIPTION ---\n")
cat(readLines(desc_file), sep = "\n")

# Delete R/copula.R (only user of scDesign3)
copula_file <- file.path(pkg_path, "R", "copula.R")
if (file.exists(copula_file)) file.remove(copula_file)

# Stub any remaining scDesign3 refs in other R/ files
for (f in list.files(file.path(pkg_path, "R"), full.names = TRUE,
                     pattern = "\\.R$")) {
  txt <- readLines(f, warn = FALSE)
  if (any(grepl("scDesign3", txt, fixed = TRUE))) {
    cat(sprintf("\nstubbing scDesign3 refs in %s:\n", basename(f)))
    for (i in seq_along(txt)) {
      if (grepl("scDesign3", txt[i], fixed = TRUE)) {
        cat(sprintf("  L%d: %s\n", i, txt[i]))
        txt[i] <- paste0("# stub: ", txt[i])
      }
    }
    writeLines(txt, f)
  }
}

# Strip copula exports from NAMESPACE
ns_file <- file.path(pkg_path, "NAMESPACE")
if (file.exists(ns_file)) {
  ns <- readLines(ns_file, warn = FALSE)
  ns2 <- ns[!grepl("copula", ns, ignore.case = TRUE)]
  writeLines(ns2, ns_file)
}

cat("\n=== Installing ===\n")
install.packages(pkg_path, repos = NULL, type = "source")

if (requireNamespace("recall", quietly = TRUE)) {
  cat("\nOK. recall version:", as.character(packageVersion("recall")), "\n")
  cat("Exports:\n"); print(getNamespaceExports("recall"))
} else {
  cat("\n!! still broken\n"); quit(status = 1)
}
