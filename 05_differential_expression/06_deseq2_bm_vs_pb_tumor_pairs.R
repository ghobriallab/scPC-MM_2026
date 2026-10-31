# ============================================================================
# Purpose:      Pseudobulk DESeq2 comparison of malignant PCs between BM and PB (circulating tumor cells) in the
#               patients sampled in both tissues at the same stage (BM_PB_set Pair), with the functions of
#               02_deseq2_functions.R (Methods, Differential expression analysis, pseudobulk level).
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulks.
#               data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240907.csv: sample_type, disease_status, disease_categs
#                 (HD = healthy donor), redcap_id, assay_type, BM_PB_set.
# Outputs:      data/pseudobulk/result/GEX5_only/BM_vs_PB_tumor/BM_tumor_pair_n<N>_vs_PB_tumor_pair_n<M>.res.csv (DESeq2
#                 results, PB as reference).
# Dependencies: R 4.1 with DESeq2, plyr (environment.R); sources ../config.R and 02_deseq2_functions.R.
# ============================================================================
source("../config.R")      # data_path()
source("02_deseq2_functions.R")

mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)
meta <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-metadata-GEX5-240907.csv"), as.is = TRUE)
OUT <- data_path("pseudobulk", "result", "GEX5_only", "BM_vs_PB_tumor")

pair <- (meta$disease_categs != "HD") & (meta$BM_PB_set == "Pair")
bm_t <- create_mat(mat, meta[(meta$sample_type == "BM") & pair, ], "BM_tumor_pair", "tumor")
pb_t <- create_mat(mat, meta[(meta$sample_type == "PB") & pair, ], "PB_tumor_pair", "tumor")
detect_DE(bm_t, pb_t, OUT)
