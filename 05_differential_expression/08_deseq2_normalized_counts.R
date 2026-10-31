# ============================================================================
# Purpose:      DESeq2 size-factor normalized counts and variance-stabilized expression (vst, blind = FALSE) of all
#               pseudobulks (Methods, Differential expression analysis, pseudobulk level). Counts + 1; design
#               ~ sample_type + disease_status + condition, with condition (tumor, normal), tissue (BM, PB) and disease
#               status (Donor, MGUS, SMM, MM) read from the pseudobulk names.
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulks (columns
#                 <tissue>.<stage>.<record id>.<normal | tumor.k>).
# Outputs:      data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5-DESeqCounts.csv: normalized counts.
#               data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5-DESeqVST.csv: variance-stabilized expression.
# Dependencies: R 4.1 with DESeq2 (environment.R); sources ../config.R.
# ============================================================================
source("../config.R")      # data_path()
suppressMessages(library(DESeq2))

mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)

# disease status from the pseudobulk names
donor_ <- grepl("\\.HD.", colnames(mat))
mgus_ <- grepl("MGUS.", colnames(mat))
smm_ <- grepl("SMM.", colnames(mat))
mm_ <- grepl("\\.MM.", colnames(mat))

meta <- data.frame(condition = ifelse(grepl("tumor.", colnames(mat)), "tumor", "normal"),
                   sample_type = ifelse(startsWith(colnames(mat), "BM."), "BM", "PB"),
                   disease_status = ifelse(donor_, "Donor", ifelse(mgus_, "MGUS",
                                                                   ifelse(smm_, "SMM", ifelse(mm_, "MM", ".")))))
rownames(meta) <- colnames(mat)

dds <- DESeqDataSetFromMatrix(countData = mat + 1, colData = meta, design = ~ sample_type + disease_status + condition)
dds <- estimateSizeFactors(dds)

cnt <- counts(dds, normalized = TRUE)
write.table(cnt, data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5-DESeqCounts.csv"),
            quote = FALSE, append = FALSE, sep = ",", col.names = TRUE, row.names = TRUE)

vsd <- vst(dds, blind = FALSE)
write.table(assay(vsd), data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5-DESeqVST.csv"),
            quote = FALSE, append = FALSE, sep = ",", col.names = TRUE, row.names = TRUE)
