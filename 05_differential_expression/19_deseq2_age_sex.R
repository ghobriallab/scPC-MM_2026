# ============================================================================
# Purpose:      Refits the pseudobulk DESeq2 comparisons of BM PCs that select the genes for signature detection (Methods,
#               Detection of single-cell transcriptional signatures, groups i-iv) with age and sex as covariates, and the
#               malignant versus normal comparisons also with patient as a covariate, on the same pseudobulk samples
#               (Results; Methods, Pseudobulk level). Designs:
#                 paper     ~ condition                    (all samples are 5' GEX, so no assay term)
#                 adjusted  ~ age + sex + condition        (age at sampling, z-scored; sex)
#               and, for malignant vs normal PCs at each stage, in the patients with both pseudobulks:
#                 paper_matched ~ condition                (the paper design on these patients)
#                 paired        ~ patient + condition      (each patient's malignant PCs compared with its own normal PCs)
#               Comparisons (BM): (i) malignant vs normal PCs per stage; (ii) malignant PCs between every pair of stages;
#               (iii) normal PCs between every pair of stages; (iv) normal PCs per stage vs healthy-donor PCs. Stages
#               MGUS, LRSMM, IRSMM, HRSMM, MM; a comparison needs >= 3 samples per group. Settings as in the pseudobulk
#               analysis of the paper: counts + 1, genes with > 3 UMIs in > 25% of the compared samples, results() Wald
#               test; DEG = Benjamini-Hochberg q < 0.01 and |log2 fold change| > 1 (Methods), immunoglobulin,
#               mitochondrial and ribosomal genes not counted. Each design is compared with its reference design
#               (DEGs retained, Jaccard index, Pearson r of log2 fold changes, sign agreement).
# Inputs:       data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed UMI counts, genes x pseudobulk samples;
#                 columns <tissue>.<stage>.<record id>.<normal | tumor.k> (k = malignant population of the patient).
#               data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240329.csv: sample_type, disease_categs, redcap_id, Sex.
#               data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id, sample_type, disease_status, Age (at
#                 sampling); age of a pseudobulk = mean over the libraries of that record, tissue and stage (record and
#                 tissue if the stage label differs; pseudobulks without age are left out of every comparison).
# Outputs:      results/differential_expression/deseq2_age_sex_concordance.tsv: per comparison and design, samples,
#                 genes tested, DEGs, DEGs shared with the reference design, % retained, Jaccard index, Pearson r of
#                 log2 fold changes, sign agreement of the reference design's DEGs, genes with q < 0.01 for age and sex.
#               results/differential_expression/deseq2_age_sex_union.tsv: union of DEGs over all comparisons per design
#                 and its overlap with the paper design.
#               results/differential_expression/deseq2_age_sex_log.txt: per-comparison results and summaries (no sample
#                 identifiers).
# Dependencies: R 4.1 with DESeq2, BiocParallel (environment.R); sources ../config.R.
# ============================================================================
source("../config.R")      # DATA_DIR, RESULTS_DIR, data_path(), result_dir(), SEED
suppressMessages({ library(DESeq2); library(BiocParallel) })
register(MulticoreParam(2))
OUT <- result_dir("differential_expression")
logf <- file(file.path(OUT, "deseq2_age_sex_log.txt"), "w")
say <- function(...) { s <- paste(...); cat(s, "\n"); cat(s, "\n", file = logf); flush(logf) }
STAGES <- c("MGUS", "LRSMM", "IRSMM", "HRSMM", "MM")
say(R.version.string, "| DESeq2", format(packageVersion("DESeq2")))

## 1. pseudobulk samples with age and sex --------------------------------------------------------------------------------
mat <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-expr-matrix-GEX5.csv"), row.names = 1, as.is = TRUE)
pmeta <- read.csv(data_path("pseudobulk", "matrix", "pseudobulk-metadata-GEX5-240329.csv"), as.is = TRUE)
lib <- read.csv(data_path("metadata", "tumor-cohort-metadata-final-240325.csv"), as.is = TRUE)
lib$disease_status[lib$disease_status == "Donor"] <- "HD"
age3 <- aggregate(Age ~ redcap_id + sample_type + disease_status, lib, mean)
age2 <- aggregate(Age ~ redcap_id + sample_type, lib, mean)
k3 <- paste(age3$redcap_id, age3$sample_type, age3$disease_status); k2 <- paste(age2$redcap_id, age2$sample_type)
pmeta$age <- age3$Age[match(paste(pmeta$redcap_id, pmeta$sample_type, pmeta$disease_categs), k3)]
miss <- is.na(pmeta$age)
pmeta$age[miss] <- age2$Age[match(paste(pmeta$redcap_id, pmeta$sample_type)[miss], k2)]
say("pseudobulk records:", nrow(pmeta), "| age from the same stage:", sum(!miss), "| from the same record and tissue:",
    sum(miss & !is.na(pmeta$age)), "| no age:", sum(is.na(pmeta$age)), "| no sex:", sum(is.na(pmeta$Sex) | pmeta$Sex == ""))

# one row per pseudobulk column (column names start with <tissue>.<stage>.<record id>.<cell type>)
cols <- colnames(mat)
cell_type <- ifelse(grepl("\\.tumor", cols), "tumor", ifelse(grepl("\\.normal", cols), "normal", NA))
prefix <- paste(pmeta$sample_type, pmeta$disease_categs, pmeta$redcap_id, sep = ".")
row_of <- sapply(seq_along(cols), function(i) {
  hit <- which(startsWith(cols[i], paste0(prefix, ".", cell_type[i])))
  if (length(hit) == 1) hit else NA })
cm <- data.frame(name = cols, cell_type = cell_type, tissue = pmeta$sample_type[row_of], stage = pmeta$disease_categs[row_of],
                 patient = pmeta$redcap_id[row_of], age = pmeta$age[row_of], sex = pmeta$Sex[row_of], stringsAsFactors = FALSE)
cm$sex[cm$sex == ""] <- NA
say("pseudobulk columns:", nrow(cm), "| matched to metadata:", sum(!is.na(row_of)))
cm <- cm[!is.na(row_of) & cm$tissue == "BM" & !is.na(cm$age) & !is.na(cm$sex), ]
cm$age_z <- as.numeric(scale(cm$age)); cm$sex <- factor(cm$sex, levels = c("F", "M"))
say("BM pseudobulks with age and sex:", nrow(cm), "|", paste(names(table(paste(cm$stage, cm$cell_type))),
                                                           table(paste(cm$stage, cm$cell_type)), sep = "=", collapse = "; "))
keep_gene <- !grepl("^IG[HKL]|^MT-|^RP[SL]", rownames(mat))   # genes not counted as DEGs (Methods)

## 2. one comparison under each design -------------------------------------------------------------------------------------
fit <- function(s, design) {
  rownames(s) <- s$name
  m <- round(as.matrix(mat[, s$name]))
  m <- m[rowMeans(m > 3) > 0.25, ]                            # genes with > 3 UMIs in > 25% of samples
  dds <- DESeqDataSetFromMatrix(countData = m + 1, colData = s, design = design)
  dds <- DESeq(dds, quiet = TRUE, parallel = TRUE)
  dds
}
deg <- function(res) {
  ok <- !is.na(res$padj) & res$padj < 0.01 & abs(res$log2FoldChange) > 1 & keep_gene[match(rownames(res), rownames(mat))]
  rownames(res)[ok]
}
n_sig <- function(dds, name) { r <- results(dds, name = name); sum(!is.na(r$padj) & r$padj < 0.01) }

rows <- list(); degs <- list()
compare <- function(label, group, s, case, ref, stage = "") {
  s$condition <- factor(ifelse(s[[group]] == case, "case", "ref"), levels = c("ref", "case"))
  if (min(table(s$condition)) < 3) return(invisible())
  d_paper <- fit(s, ~ condition); r_paper <- results(d_paper, name = "condition_case_vs_ref")
  d_adj <- fit(s, ~ age_z + sex + condition); r_adj <- results(d_adj, name = "condition_case_vs_ref")
  out <- list(paper = list(r_paper, NA, NA), adjusted = list(r_adj, n_sig(d_adj, "age_z"), n_sig(d_adj, "sex_M_vs_F")))
  if (label == "i_malignant_vs_normal") {                    # paired design in patients with both pseudobulks
    both <- intersect(s$patient[s$condition == "case"], s$patient[s$condition == "ref"])
    sp <- s[s$patient %in% both, ]; sp$patient <- factor(sp$patient)
    out$paper_matched <- list(results(fit(sp, ~ condition), name = "condition_case_vs_ref"), NA, NA)
    out$paired <- list(results(fit(sp, ~ patient + condition), name = "condition_case_vs_ref"), NA, NA)
  }
  ref_set <- list(adjusted = "paper", paper = "paper", paper_matched = "paper_matched", paired = "paper_matched")
  for (dz in names(out)) {
    r <- out[[dz]][[1]]; b <- out[[ref_set[[dz]]]][[1]]
    g <- intersect(rownames(r), rownames(b)); dr <- deg(r); db <- deg(b)
    sx <- if (dz %in% c("paper_matched", "paired")) sp else s
    rows[[length(rows) + 1]] <<- data.frame(
      comparison = label, stage = stage, case = case, ref = ref, design = dz, reference_design = ref_set[[dz]],
      n_case = sum(sx$condition == "case"), n_ref = sum(sx$condition == "ref"),
      n_patients_paired = if (dz %in% c("paper_matched", "paired")) length(unique(sp$patient)) else NA,
      genes_tested = nrow(r), degs = length(dr), degs_reference = length(db), shared = length(intersect(dr, db)),
      retained_pct = 100 * length(intersect(dr, db)) / max(length(db), 1),
      jaccard = length(intersect(dr, db)) / max(length(union(dr, db)), 1),
      r_log2fc = cor(r[g, "log2FoldChange"], b[g, "log2FoldChange"], use = "complete.obs"),
      sign_agree_pct = 100 * mean(sign(r[db, "log2FoldChange"]) == sign(b[db, "log2FoldChange"]), na.rm = TRUE),
      age_q01 = out[[dz]][[2]], sex_q01 = out[[dz]][[3]])
    degs[[paste(label, stage, case, ref, dz)]] <<- dr
    x <- rows[[length(rows)]]
    say(sprintf("%-22s %-5s %-7s vs %-6s %-13s n %d vs %d | DEGs %d (reference %d), shared %d (%.0f%%), r log2FC %.3f | age %s, sex %s",
                label, stage, case, ref, dz, x$n_case, x$n_ref, x$degs, x$degs_reference, x$shared, x$retained_pct, x$r_log2fc,
                x$age_q01, x$sex_q01))
  }
}

tum <- cm[cm$cell_type == "tumor" & cm$stage %in% STAGES, ]
nor <- cm[cm$cell_type == "normal" & cm$stage %in% STAGES, ]
don <- cm[cm$cell_type == "normal" & cm$stage == "HD", ]
for (st in STAGES) {                                                   # (i) malignant vs normal PCs, per stage
  s <- rbind(tum[tum$stage == st, ], nor[nor$stage == st, ]); s$grp <- s$cell_type
  compare("i_malignant_vs_normal", "grp", s, "tumor", "normal", st)
}
pairs <- t(combn(STAGES, 2))
for (k in seq_len(nrow(pairs))) {                                      # (ii) malignant and (iii) normal PCs, stage pairs
  compare("ii_malignant_stages", "stage", tum[tum$stage %in% pairs[k, ], ], pairs[k, 2], pairs[k, 1])
  compare("iii_normal_stages", "stage", nor[nor$stage %in% pairs[k, ], ], pairs[k, 2], pairs[k, 1])
}
for (st in STAGES) {                                                   # (iv) normal PCs vs healthy donors
  s <- rbind(nor[nor$stage == st, ], don); s$grp <- ifelse(s$stage == "HD", "HD", "patient")
  compare("iv_normal_vs_donor", "grp", s, "patient", "HD", st)
}
res <- do.call(rbind, rows)
write.table(res, file.path(OUT, "deseq2_age_sex_concordance.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

## 3. union over the comparisons (the gene pool groups i-iv) ----------------------------------------------------------------
un <- function(dz) unique(unlist(degs[grepl(paste0(" ", dz, "$"), names(degs))]))
u_paper <- un("paper"); u_adj <- un("adjusted")
union_tab <- data.frame(design = c("paper", "adjusted"), union_degs = c(length(u_paper), length(u_adj)),
                        shared_with_paper = c(length(u_paper), length(intersect(u_adj, u_paper))),
                        retained_pct = c(100, 100 * length(intersect(u_adj, u_paper)) / length(u_paper)))
write.table(union_tab, file.path(OUT, "deseq2_age_sex_union.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
say("union of DEGs over all comparisons: paper", length(u_paper), "| adjusted", length(u_adj), "| shared",
    length(intersect(u_adj, u_paper)), sprintf("(%.1f%% of the paper design's)", union_tab$retained_pct[2]))
for (lab in unique(res$comparison)) for (dz in unique(res$design[res$comparison == lab])) {
  x <- res[res$comparison == lab & res$design == dz, ]
  say(sprintf("summary %-24s %-13s: %d comparisons, DEGs retained median %.0f%% (range %.0f-%.0f), r log2FC median %.3f (min %.3f)",
              lab, dz, nrow(x), median(x$retained_pct), min(x$retained_pct), max(x$retained_pct), median(x$r_log2fc), min(x$r_log2fc)))
}
say("done")
