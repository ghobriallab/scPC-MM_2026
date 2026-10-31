# ============================================================================
# Purpose:      Shared paths for every R script: input, result and figure directories, so that no script holds a
#               machine-specific path. R scripts are run from their own folder and source("../config.R").
# Inputs:       Optional environment variables SCPCMM_DATA, SCPCMM_RESULTS and SCPCMM_FIGURES; the defaults are
#               data/, results/ and figures/ at the repository root.
# Outputs:      REPO_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR, SEED and the helpers data_path() and result_dir().
# Dependencies: R base.
# ============================================================================
REPO_DIR <- normalizePath(file.path(getwd(), ".."))
DATA_DIR <- Sys.getenv("SCPCMM_DATA", file.path(REPO_DIR, "data"))
RESULTS_DIR <- Sys.getenv("SCPCMM_RESULTS", file.path(REPO_DIR, "results"))
FIGURES_DIR <- Sys.getenv("SCPCMM_FIGURES", file.path(REPO_DIR, "figures"))

SEED <- 42

# An input file under DATA_DIR, e.g. data_path("commpass", "MMRF_CoMMpass_IA22_BM_1_metadata.tsv")
data_path <- function(...) file.path(DATA_DIR, ...)

# The output folder of one analysis step, created if needed
result_dir <- function(step) {
  d <- file.path(RESULTS_DIR, step)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}
