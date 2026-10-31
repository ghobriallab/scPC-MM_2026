# ============================================================================
# Purpose:      Functions of the pseudobulk DESeq2 comparisons (Methods, Differential expression analysis, pseudobulk
#               level), sourced by 03-07. create_mat() takes the pseudobulk columns of the rows of a sample table
#               (columns <tissue>.<stage>.<record id>.<tumor | normal>...). detect_DE() keeps the genes of both groups
#               with > 3 UMIs in > 25% of the compared pseudobulks, adds a pseudocount of 1, fits DESeq2 with design
#               ~ condition (~ assay_type + condition when the pseudobulks hold two assay types), with the reference
#               group as baseline, and writes the Wald test results ordered by adjusted P to each output folder given.
# Inputs:       Pseudobulk count matrix and sample table passed by the calling script (sample table columns in the
#               order of data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240329.csv: sample_type, disease_status,
#               disease_categs, redcap_id, assay_type, ...).
# Outputs:      <case>_n<N>_vs_<reference>_n<M>.res.csv (baseMean, log2FoldChange, lfcSE, stat, pvalue, padj; N and M,
#               numbers of pseudobulks of each group) in each output folder given to detect_DE().
# Dependencies: R 4.1 with DESeq2, plyr (environment.R).
# ============================================================================
suppressMessages({ library(plyr); library(DESeq2) })

# DESeq2 on counts + 1, with the reference group as baseline
run_deseq2 <- function(df, reference, correct_assay = TRUE) {
  if (correct_assay == TRUE) {
    dds <- DESeqDataSetFromMatrix(countData = df$mat + 1, colData = df$meta, design = ~ assay_type + condition)
  } else {
    dds <- DESeqDataSetFromMatrix(countData = df$mat + 1, colData = df$meta, design = ~ condition)
  }
  if (!is.na(reference)) {
    dds$condition <- relevel(dds$condition, ref = reference)
  }
  DESeq(dds)
}

# genes measured in every group; keeps the genes with > 3 UMIs in > 25% of the pseudobulks
preprocess_data <- function(xc) {
  genes <- rownames(xc[[1]]$mat)
  for (i in 2:length(xc)) {
    genes <- intersect(genes, rownames(xc[[i]]$mat))
  }
  mat <- xc[[1]]$mat[genes, ]
  meta <- xc[[1]]$meta
  for (i in 2:length(xc)) {
    mat <- cbind(mat, xc[[i]]$mat[genes, ])
    meta <- rbind(meta, xc[[i]]$meta)
  }
  meta$assay_type <- factor(meta$assay_type)
  meta$condition <- factor(meta$condition)
  filter_index <- apply(mat, 1, function(x) sum(x > 3) / length(x)) > 0.25
  list("mat" = mat[filter_index, ], "meta" = meta)
}

# pseudobulk columns of one sample-table row: names starting with <sample_type>.<disease_categs>.<redcap_id>.<suffix>
# (row[1], row[3], row[4]); row[5] = assay_type
meta_row <- function(names, row, condition, suffix = "") {
  search_str <- paste(row[1], row[3], row[4], suffix, sep = ".")
  match <- names[startsWith(names, search_str)]
  N <- length(match)
  data.frame(name = match, assay_type = rep(row[5], N), condition = rep(condition, N))
}

# counts and sample table of the pseudobulks of the rows of a sample table, all labeled with one condition
create_mat <- function(expr, metadata, condition, suffix = "") {
  rows <- apply(metadata, 1, function(x) meta_row(colnames(expr), x, condition, suffix))
  sample <- ldply(rows, data.frame, .id = NULL)
  rownames(sample) <- sample$name
  list("mat" = expr[, sample$name], "meta" = sample)
}

# DESeq2 of treatment vs reference (lists from create_mat); the results table is written to every folder of out_dirs
detect_DE <- function(treatment, reference, out_dirs, correct_assay = TRUE) {
  df <- preprocess_data(list(treatment, reference))
  if (length(unique(df$meta[, "assay_type"])) == 1) {
    correct_assay <- FALSE
  }
  dds <- run_deseq2(df, unique(reference$meta$condition), correct_assay)
  res <- results(dds)
  res <- res[order(res$padj), ]
  tre_name <- paste0(unique(treatment$meta$condition), "_n", nrow(treatment$meta))
  ref_name <- paste0(unique(reference$meta$condition), "_n", nrow(reference$meta))
  prefix <- paste0(tre_name, "_vs_", ref_name)
  for (d in out_dirs) {
    dir.create(d, showWarnings = FALSE, recursive = TRUE)
    write.table(res, file.path(d, paste0(prefix, ".res.csv")), quote = FALSE, append = FALSE, sep = ",",
                col.names = TRUE, row.names = TRUE)
  }
  invisible(dds)
}
