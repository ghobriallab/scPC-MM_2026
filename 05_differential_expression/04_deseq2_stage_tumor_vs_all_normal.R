# ============================================================================
# Purpose:      Pseudobulk DESeq2 comparisons of the malignant BM PCs of each disease stage (MGUS, SMM, MM; stages
#               from disease_status) with the normal BM PCs of all patients and healthy donors together (Methods,
#               Differential expression analysis, pseudobulk level), with the functions of 02_deseq2_functions.R.
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulks.
#               data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240329.csv: sample_type, disease_status, disease_categs,
#                 redcap_id, assay_type.
# Outputs:      data/pseudobulk/result/GEX5_only/stage_tumor_v_normal/cross/
#                 BM_<stage>_tumor_n<N>_vs_BM_ALL_normal_n<M>.res.csv (DESeq2 results), read by
#                 12_stage_dysregulated_genes.py.
# Dependencies: R 4.1 with DESeq2, plyr (environment.R); sources ../config.R and 02_deseq2_functions.R.
# ============================================================================
source("../config.R")      # data_path()
source("02_deseq2_functions.R")

mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)
meta <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-metadata-GEX5-240329.csv"), as.is = TRUE)
OUT <- data_path("pseudobulk", "result", "GEX5_only", "stage_tumor_v_normal", "cross")

bm_mgus <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$disease_status == "MGUS"), ], "BM_MGUS_tumor",
                      "tumor")
bm_smm <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$disease_status == "SMM"), ], "BM_SMM_tumor", "tumor")
bm_mm <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$disease_status == "MM"), ], "BM_MM_tumor", "tumor")

# normal BM PCs of every row of the sample table (patients and healthy donors)
bm_normal <- create_mat(mat, meta[(meta$sample_type == "BM"), ], "BM_ALL_normal", "normal")

detect_DE(bm_mgus, bm_normal, OUT)
detect_DE(bm_smm, bm_normal, OUT)
detect_DE(bm_mm, bm_normal, OUT)
