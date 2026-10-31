# ============================================================================
# Purpose:      Pseudobulk DESeq2 comparisons of BM PCs within each cytogenetic group of patients (Methods,
#               Differential expression analysis, pseudobulk level), with the functions of 02_deseq2_functions.R. For
#               each group (Other, HRD, t(4;14), t(11;14), t(12;14), t(6;14), t(14;16), t(14;20), t(8;14)) and for the
#               t(11;14) patients at each stage (MGUS, SMM, LRSMM, IRSMM, HRSMM, MM): malignant vs normal PCs (all
#               patients, and patients with both pseudobulks, condition suffix _mp) and normal PCs vs healthy-donor PCs.
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulks.
#               data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240907.csv: sample_type, disease_status, disease_categs
#                 (HD = healthy donor), redcap_id, assay_type, cytogenetics.
# Outputs:      data/pseudobulk/result/GEX5_only/cytogenetics/BM_<group>_tumor_n<N>_vs_BM_<group>_normal_n<M>.res.csv,
#                 BM_<group>_tumor_mp_n<N>_vs_BM_<group>_normal_mp_n<M>.res.csv and
#                 BM_<group>_normal_n<N>_vs_BM_donor_n<M>.res.csv (DESeq2 results; group t(11;14) = t1114, and
#                 t1114_<stage> for the stages).
# Dependencies: R 4.1 with DESeq2, plyr (environment.R); sources ../config.R and 02_deseq2_functions.R.
# ============================================================================
source("../config.R")      # data_path()
source("02_deseq2_functions.R")

mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)
meta <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-metadata-GEX5-240907.csv"), as.is = TRUE)
OUT <- data_path("pseudobulk", "result", "GEX5_only", "cytogenetics")

# patients with both a malignant and a normal pseudobulk (same tissue and stage); conditions get the suffix _mp
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

bm_donor <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$disease_categs == "HD"), ], "BM_donor", "normal")

# malignant vs normal PCs (all and matched patients) and normal PCs vs healthy donors, for the rows sel of the table
group_de <- function(sel, label) {
  bm_t <- create_mat(mat, sel, paste0("BM_", label, "_tumor"), "tumor")
  bm_n <- create_mat(mat, sel, paste0("BM_", label, "_normal"), "normal")
  detect_DE(bm_t, bm_n, OUT)
  detect_DE(bm_n, bm_donor, OUT)
  mp <- matched_tn(bm_t, bm_n)
  detect_DE(mp$t, mp$n, OUT)
}

patient_bm <- (meta$sample_type == "BM") & (meta$disease_categs != "HD")
group_de(meta[patient_bm & (meta$cytogenetics == "Other"), ], "Other")
group_de(meta[patient_bm & (meta$cytogenetics == "HRD"), ], "HRD")
group_de(meta[patient_bm & (meta$cytogenetics == "t(4;14)"), ], "t0414")
group_de(meta[patient_bm & (meta$cytogenetics == "t(11;14)"), ], "t1114")

# t(11;14) patients at each stage (SMM from disease_status, risk groups from disease_categs)
subtypes <- c("MGUS", "SMM", "LRSMM", "IRSMM", "HRSMM", "MM")
stage_cols <- c("disease_categs", "disease_status", "disease_categs", "disease_categs", "disease_categs",
                "disease_categs")
for (i in seq_along(subtypes)) {
  sel <- meta[(meta$sample_type == "BM") & (meta$cytogenetics == "t(11;14)") & (meta[, stage_cols[i]] == subtypes[i]), ]
  group_de(sel, paste0("t1114_", subtypes[i]))
}

group_de(meta[patient_bm & (meta$cytogenetics == "t(12;14)"), ], "t1214")
group_de(meta[patient_bm & (meta$cytogenetics == "t(6;14)"), ], "t0614")
group_de(meta[patient_bm & (meta$cytogenetics == "t(14;16)"), ], "t1416")
group_de(meta[patient_bm & (meta$cytogenetics == "t(14;20)"), ], "t1420")
group_de(meta[patient_bm & (meta$cytogenetics == "t(8;14)"), ], "t0814")
