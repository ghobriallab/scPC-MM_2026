# ============================================================================
# Purpose:      Pseudobulk DESeq2 comparisons of malignant vs normal BM PCs at each disease stage with SMM split by the
#               IMWG 2/20/20 risk group (Fig. 2B), with the functions of 02_deseq2_functions.R and the design of
#               03_deseq2_stage_comparisons.R (patients with both a malignant and a normal pseudobulk at the stage,
#               condition suffix _mp). Each SMM pseudobulk takes the risk group of its samples (SMM_stage_strict of the
#               cohort table, after the clinical-record correction of three PB samples, as in risk.py); pseudobulks of
#               one patient that then share tissue, risk group and PC population are summed, as
#               01_pseudobulk_matrices.py sums the cells of one group.
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulks
#                 (<tissue>.<stage>.<record id>.<tumor.k | normal>).
#               data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240329.csv: sample_type, disease_status, disease_categs,
#                 redcap_id, assay_type.
#               data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id, sample_type, disease_status,
#                 SMM_stage_strict.
# Outputs:      results/differential_expression/pseudobulk_imwg_risk/BM_<stage>_tumor_mp_n<N>_vs_BM_<stage>_normal_mp_n<M>.res.csv
#                 (DESeq2 results) for MGUS, LRSMM, IRSMM, HRSMM and MM.
# Dependencies: R 4.1 with DESeq2, plyr (environment.R); sources ../config.R and 02_deseq2_functions.R.
# ============================================================================
source("../config.R")      # data_path(), result_dir()
source("02_deseq2_functions.R")

SMM <- c("LRSMM", "IRSMM", "HRSMM")
OUT <- file.path(result_dir("differential_expression"), "pseudobulk_imwg_risk")

mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)
meta <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-metadata-GEX5-240329.csv"), as.is = TRUE)
cohort <- read.csv(data_path("metadata", "tumor-cohort-metadata-final-240325.csv"), as.is = TRUE)

# IMWG risk group of each (record, tissue, recorded risk group); one group per key
co <- unique(cohort[cohort$disease_status %in% SMM, c("redcap_id", "sample_type", "disease_status", "SMM_stage_strict")])
key <- paste(co$redcap_id, co$sample_type, co$disease_status)
stopifnot(!any(duplicated(key)))
imwg <- setNames(co$SMM_stage_strict, key)
# risk group of three PB samples of two patients from the clinical records (the cell table holds LRSMM)
pt <- setNames(cohort$patient_id, cohort$redcap_id)
clinical <- c("Pt-90 PB LRSMM" = "IRSMM", "Pt-95 PB LRSMM" = "HRSMM")
to_imwg <- function(rec, tissue, stage) {
  k1 <- paste(pt[rec], tissue, stage)
  stage <- ifelse(k1 %in% names(clinical), clinical[k1], stage)
  unname(imwg[paste(rec, tissue, stage)])
}

# sample table: risk group of each SMM row of the cohort, then one row per patient, tissue and group
in_cohort <- meta$redcap_id %in% cohort$redcap_id
cat("sample-table rows outside the cohort table:", sum(!in_cohort), "\n")
meta <- meta[in_cohort, ]
smm_row <- meta$disease_categs %in% SMM
new <- to_imwg(meta$redcap_id[smm_row], meta$sample_type[smm_row], meta$disease_categs[smm_row])
stopifnot(!any(is.na(new)))
meta$disease_categs[smm_row] <- new
meta <- unique(meta)
cat("sample table rows:", nrow(meta), "\n")

# pseudobulk columns (names as read.csv gives them, as in 03_deseq2_stage_comparisons.R): <tissue>.<stage>.<record>.<rest>;
# SMM columns of cohort records take the IMWG group and columns that become one are summed; other columns are kept as
# they are (create_mat only matches the columns of the sample-table rows)
parts <- strsplit(colnames(mat), ".", fixed = TRUE)
tissue <- sapply(parts, "[[", 1)
stage <- sapply(parts, "[[", 2)
rec <- sapply(parts, "[[", 3)
rest <- sapply(parts, function(p) paste(p[-(1:3)], collapse = "."))
smm_col <- (stage %in% SMM) & (rec %in% cohort$redcap_id)
new <- to_imwg(rec[smm_col], tissue[smm_col], stage[smm_col])
stopifnot(!any(is.na(new)))
stage[smm_col] <- new
new_name <- colnames(mat)
new_name[smm_col] <- paste(tissue, stage, rec, rest, sep = ".")[smm_col]
summed <- sapply(split(seq_along(new_name), new_name), function(i) rowSums(mat[, i, drop = FALSE]))
cat("pseudobulks:", ncol(mat), "-> ", ncol(summed), "( summed groups:", sum(table(new_name) > 1), ")\n")
mat <- as.data.frame(summed)

# malignant vs normal BM PCs at each stage, patients with both pseudobulks (as in 03_deseq2_stage_comparisons.R)
matched_tn <- function(tumor, normal) {
  pat_t <- sapply(strsplit(tumor$meta$name, ".tumor"), "[[", 1)
  pat_n <- sapply(strsplit(normal$meta$name, ".normal"), "[[", 1)
  overlap <- intersect(pat_t, pat_n)
  t_names <- tumor$meta$name[pat_t %in% overlap]
  n_names <- normal$meta$name[pat_n %in% overlap]
  tumor$mat <- tumor$mat[, t_names]
  tumor$meta <- tumor$meta[t_names, ]
  tumor$meta$condition <- paste0(tumor$meta$condition, "_mp")
  normal$mat <- normal$mat[, n_names]
  normal$meta <- normal$meta[n_names, ]
  normal$meta$condition <- paste0(normal$meta$condition, "_mp")
  list(t = tumor, n = normal)
}

for (st in c("MGUS", SMM, "MM")) {
  sel <- meta[(meta$sample_type == "BM") & (meta$disease_categs == st), ]
  bm_t <- create_mat(mat, sel, paste0("BM_", st, "_tumor"), "tumor")
  bm_n <- create_mat(mat, sel, paste0("BM_", st, "_normal"), "normal")
  mp <- matched_tn(bm_t, bm_n)
  cat(st, ": malignant pseudobulks", nrow(mp$t$meta), "| normal pseudobulks", nrow(mp$n$meta), "\n")
  detect_DE(mp$t, mp$n, OUT)
}
