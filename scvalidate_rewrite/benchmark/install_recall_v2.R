# Second-attempt recall install. Strategy:
#   1. Install simple deps (lamW, knockoff, countsplit) from CRAN.
#   2. Download recall source.
#   3. Strip the copula.R source file and all references to scDesign3 from the
#      package — the copula branch is the only user of scDesign3, and we do
#      not port or exercise it.
#   4. R CMD INSTALL the patched source.
# This is what we would do if we forked lcrawlab/recall to remove the scDesign3
# hard dep, just done locally without a fork.

options(repos = c(CRAN = "https://cloud.r-project.org"))

# Deps we still need
for (pkg in c("lamW", "knockoff", "countsplit")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s from CRAN ...\n", pkg))
    install.packages(pkg)
  } else {
    cat(sprintf("  %s ok\n", pkg))
  }
}

# Download recall source
tmpd <- tempfile("recall_nocopula_")
dir.create(tmpd)
zipfile <- file.path(tmpd, "recall.zip")
download.file("https://github.com/lcrawlab/recall/archive/refs/heads/main.zip",
              zipfile, mode = "wb")
unzip(zipfile, exdir = tmpd)
pkg_path <- file.path(tmpd, "recall-main")
cat("Source at:", pkg_path, "\n")
cat("R/ files:\n"); print(list.files(file.path(pkg_path, "R")))

# --- Patch DESCRIPTION -------------------------------------------------------
desc_file <- file.path(pkg_path, "DESCRIPTION")
desc_lines <- readLines(desc_file)
cat("\n--- ORIGINAL DESCRIPTION ---\n")
cat(desc_lines, sep = "\n")

# Strip Remotes: block and remove scDesign3 from Imports/Depends
in_multi <- NULL
new_lines <- character()
for (line in desc_lines) {
  if (grepl("^Remotes:", line)) {
    in_multi <- "remotes"; next
  }
  if (!is.null(in_multi) && grepl("^[A-Za-z]", line)) {
    in_multi <- NULL
  }
  if (!is.null(in_multi)) next  # drop the Remotes continuation
  # Strip ", scDesign3" or "scDesign3," token from Imports continuation lines
  line2 <- gsub(",\\s*scDesign3(\\s*\\([^)]*\\))?", "", line)
  line2 <- gsub("scDesign3(\\s*\\([^)]*\\))?\\s*,\\s*", "", line2)
  line2 <- gsub("^\\s*scDesign3(\\s*\\([^)]*\\))?\\s*$", "", line2)
  new_lines <- c(new_lines, line2)
}
# Drop any now-empty continuation lines
keep <- !(nchar(trimws(new_lines)) == 0 & seq_along(new_lines) > 1 &
          grepl("^\\s", c("x", head(new_lines, -1))))  # heuristic; safe-ish
writeLines(new_lines, desc_file)
cat("\n--- PATCHED DESCRIPTION ---\n")
cat(readLines(desc_file), sep = "\n")

# --- Patch R/ sources --------------------------------------------------------
# Delete copula.R entirely; it's the only file that uses scDesign3.
copula_file <- file.path(pkg_path, "R", "copula.R")
if (file.exists(copula_file)) {
  cat("\nDeleting R/copula.R\n")
  file.remove(copula_file)
}

# Find any other files that reference scDesign3 and stub out the references.
r_files <- list.files(file.path(pkg_path, "R"), full.names = TRUE,
                      pattern = "\\.R$")
for (f in r_files) {
  txt <- readLines(f, warn = FALSE)
  if (any(grepl("scDesign3", txt, fixed = TRUE))) {
    cat(sprintf("  scDesign3 refs in %s — stubbing lines:\n", basename(f)))
    for (i in seq_along(txt)) {
      if (grepl("scDesign3", txt[i], fixed = TRUE)) {
        cat(sprintf("    L%d: %s\n", i, txt[i]))
        txt[i] <- sprintf("# stubbed: %s", txt[i])
      }
    }
    writeLines(txt, f)
  }
}

# Also check if copula entry points are exported in NAMESPACE; strip if so.
ns_file <- file.path(pkg_path, "NAMESPACE")
if (file.exists(ns_file)) {
  ns <- readLines(ns_file)
  ns2 <- ns[!grepl("copula", ns, ignore.case = TRUE)]
  if (length(ns2) != length(ns)) {
    cat("\nStripping copula exports from NAMESPACE\n")
    writeLines(ns2, ns_file)
  }
}

cat("\n=== Installing patched recall ===\n")
install.packages(pkg_path, repos = NULL, type = "source")

cat("\n=== Verify ===\n")
if (requireNamespace("recall", quietly = TRUE)) {
  cat("OK. recall version:", as.character(packageVersion("recall")), "\n")
  cat("Exports:\n")
  print(getNamespaceExports("recall"))
} else {
  cat("!! still broken\n")
  quit(status = 1)
}
