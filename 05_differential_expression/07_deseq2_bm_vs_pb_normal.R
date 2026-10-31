# ============================================================================
# Purpose:      Pseudobulk DESeq2 comparison of normal PCs between BM and PB, patients and healthy donors together
#               (Methods, Differential expression analysis, pseudobulk level), with the functions of
#               02_deseq2_functions.R: all pseudobulks, and the 5' (GEX5) pseudobulks.
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulks.
#               data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240329.csv: sample_type, disease_status, disease_categs,
#                 redcap_id, assay_type.
# Outputs:      data/pseudobulk/result/GEX5_only/BM_npc_vs_PB_npc/BM_normal_pc_n<N>_vs_PB_normal_pc_n<M>.res.csv and
#                 BM_npc_GEX5_n<N>_vs_PB_npc_GEX5_n<M>.res.csv (DESeq2 results, PB as reference).
# Dependencies: R 4.1 with DESeq2, plyr (environment.R); sources ../config.R and 02_deseq2_functions.R.
# ============================================================================
source("../config.R")      # data_path()
source("02_deseq2_functions.R")

mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)
meta <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-metadata-GEX5-240329.csv"), as.is = TRUE)
OUT <- data_path("pseudobulk", "result", "GEX5_only", "BM_npc_vs_PB_npc")

# all normal PCs (patients and healthy donors)
bm <- create_mat(mat, meta[(meta$sample_type == "BM"), ], "BM_normal_pc", "normal")
pb <- create_mat(mat, meta[(meta$sample_type == "PB"), ], "PB_normal_pc", "normal")
detect_DE(bm, pb, OUT)

# 5' libraries only
bm <- create_mat(mat, meta[(meta$sample_type == "BM") & (meta$assay_type == "GEX5"), ], "BM_npc_GEX5", "normal")
pb <- create_mat(mat, meta[(meta$sample_type == "PB") & (meta$assay_type == "GEX5"), ], "PB_npc_GEX5", "normal")
detect_DE(bm, pb, OUT)
