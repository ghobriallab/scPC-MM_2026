# ============================================================================
# Purpose:      Pseudobulk DESeq2 comparisons by disease stage (Methods, Differential expression analysis, pseudobulk
#               level), with the functions of 02_deseq2_functions.R. In BM and in PB: malignant vs normal PCs of all
#               patients and at each stage (MGUS, SMM = all risk groups from disease_status, LRSMM, IRSMM, HRSMM, MM;
#               in BM also in the patients with both pseudobulks, condition suffix _mp); normal PCs of patients vs
#               healthy-donor PCs, overall and at each stage; malignant PCs, and normal PCs, between 12 pairs of stages.
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulks.
#               data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240329.csv: sample_type, disease_status, disease_categs
#                 (HD = healthy donor), redcap_id, assay_type.
# Outputs:      data/pseudobulk/result/GEX5_only/{tumor_vs_normal,normal_vs_donor,tumor_vs_tumor,normal_vs_normal}/
#                 <case>_n<N>_vs_<reference>_n<M>.res.csv (DESeq2 results). The BM malignant vs normal results of all
#                 stages together and of MGUS, LRSMM, IRSMM, HRSMM and MM are also written to
#                 data/pseudobulk/result/GEX5_only/stage_tumor_v_normal/{cross,<stage>}/, read by
#                 12_stage_dysregulated_genes.py.
# Dependencies: R 4.1 with DESeq2, plyr (environment.R); sources ../config.R and 02_deseq2_functions.R.
# ============================================================================
source("../config.R")      # data_path()
source("02_deseq2_functions.R")

mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)
meta <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-metadata-GEX5-240329.csv"), as.is = TRUE)

RES <- data_path("pseudobulk", "result", "GEX5_only")
TVN <- file.path(RES, "tumor_vs_normal")
NVD <- file.path(RES, "normal_vs_donor")
TVT <- file.path(RES, "tumor_vs_tumor")
NVN <- file.path(RES, "normal_vs_normal")
STAGE_DIR <- file.path(RES, "stage_tumor_v_normal")

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

# 1. all patients: malignant vs normal PCs (BM: all patients and matched patients; PB)
bm_t <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$disease_categs != "HD"), ], "BM_tumor", "tumor")
bm_n <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$disease_categs != "HD"), ], "BM_normal", "normal")
detect_DE(bm_t, bm_n, c(TVN, file.path(STAGE_DIR, "cross")))
mp <- matched_tn(bm_t, bm_n)
detect_DE(mp$t, mp$n, TVN)

pb_t <- create_mat(mat, meta[(meta$sample_type == "PB") & (meta$disease_categs != "HD"), ], "PB_tumor", "tumor")
pb_n <- create_mat(mat, meta[(meta$sample_type == "PB") & (meta$disease_categs != "HD"), ], "PB_normal", "normal")
detect_DE(pb_t, pb_n, TVN)

# 2. normal PCs of patients vs healthy-donor PCs
bm_donor <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$disease_categs == "HD"), ], "BM_donor", "normal")
detect_DE(bm_n, bm_donor, NVD)
pb_donor <- create_mat(mat, meta[(meta$sample_type == "PB") & (meta$disease_categs == "HD"), ], "PB_donor", "normal")
detect_DE(pb_n, pb_donor, NVD)

# 3. each stage: malignant vs normal PCs, normal PCs vs healthy donors (SMM from disease_status, risk groups from
#    disease_categs)
subtypes <- c("MGUS", "SMM", "LRSMM", "IRSMM", "HRSMM", "MM")
stage_cols <- c("disease_categs", "disease_status", "disease_categs", "disease_categs", "disease_categs",
                "disease_categs")
bm_sub_t <- list(); bm_sub_n <- list()
pb_sub_t <- list(); pb_sub_n <- list()
for (i in seq_along(subtypes)) {
  st <- subtypes[i]
  sel_bm <- meta[(meta$sample_type == "BM") & (meta[, stage_cols[i]] == st), ]
  bm_t <- create_mat(mat, sel_bm, paste0("BM_", st, "_tumor"), "tumor")
  bm_n <- create_mat(mat, sel_bm, paste0("BM_", st, "_normal"), "normal")
  detect_DE(bm_t, bm_n, if (st == "SMM") TVN else c(TVN, file.path(STAGE_DIR, st)))
  detect_DE(bm_n, bm_donor, NVD)
  mp <- matched_tn(bm_t, bm_n)
  detect_DE(mp$t, mp$n, TVN)
  bm_sub_t[[st]] <- bm_t
  bm_sub_n[[st]] <- bm_n

  sel_pb <- meta[(meta$sample_type == "PB") & (meta[, stage_cols[i]] == st), ]
  pb_sub_t[[st]] <- create_mat(mat, sel_pb, paste0("PB_", st, "_tumor"), "tumor")
  pb_sub_n[[st]] <- create_mat(mat, sel_pb, paste0("PB_", st, "_normal"), "normal")
  detect_DE(pb_sub_t[[st]], pb_sub_n[[st]], TVN)
  detect_DE(pb_sub_n[[st]], pb_donor, NVD)
}

# 4. pairs of stages (later stage vs earlier stage): malignant PCs and normal PCs, in BM and PB
pairs <- rbind(c("SMM", "MGUS"), c("LRSMM", "MGUS"), c("IRSMM", "MGUS"), c("HRSMM", "MGUS"),
               c("MM", "MGUS"), c("MM", "SMM"), c("IRSMM", "LRSMM"), c("HRSMM", "LRSMM"),
               c("MM", "LRSMM"), c("HRSMM", "IRSMM"), c("MM", "IRSMM"), c("MM", "HRSMM"))
for (i in seq_len(nrow(pairs))) {
  st1 <- pairs[i, 1]
  st2 <- pairs[i, 2]
  detect_DE(bm_sub_t[[st1]], bm_sub_t[[st2]], TVT)
  detect_DE(bm_sub_n[[st1]], bm_sub_n[[st2]], NVN)
  detect_DE(pb_sub_t[[st1]], pb_sub_t[[st2]], TVT)
  detect_DE(pb_sub_n[[st1]], pb_sub_n[[st2]], NVN)
}
