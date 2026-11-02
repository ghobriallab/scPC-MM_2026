# ============================================================================
# Purpose:      One-shot installer for the R packages of the analysis scripts: doublet scores (02_preprocessing, called
#               through rpy2), Numbat and clone graphs (04_genomics), DESeq2 (05_differential_expression), survival
#               models (10_commpass) and the SHazaM threshold used by dandelion (03_malignant_cells), at the versions
#               the published results were generated with where recorded.
# Inputs:       None (a package already present at the pinned version is skipped).
# Outputs:      Installed R packages, listed in CRAN, CRAN_ANY and BIOC below.
# Dependencies: R 4.1 with internet access to CRAN and Bioconductor; 'remotes' and 'BiocManager' are installed
#               automatically if missing.
# ============================================================================

CRAN <- c(survival = "3.5-8", numbat = "1.1.0", nnet = "7.3-20")
CRAN_ANY <- c("argparse", "plyr", "Seurat", "tidygraph", "shazam")
BIOC_VERSION <- "3.14"
BIOC <- c("DESeq2", "BiocParallel", "scuttle", "scran", "scater", "bluster", "scDblFinder", "scds")

REPO <- "https://cloud.r-project.org"
for (p in c("remotes", "BiocManager")) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = REPO)
}

BiocManager::install(version = BIOC_VERSION, ask = FALSE, update = FALSE)
BiocManager::install(BIOC, ask = FALSE, update = FALSE)

for (pkg in CRAN_ANY) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = REPO)
}

for (pkg in names(CRAN)) {
  want <- CRAN[[pkg]]
  have <- tryCatch(as.character(packageVersion(pkg)), error = function(e) NA_character_)
  if (!identical(have, gsub("-", ".", want))) {
    remotes::install_version(pkg, version = want, repos = REPO, upgrade = "never")
  }
}

cat("\nAll packages installed.\n")
