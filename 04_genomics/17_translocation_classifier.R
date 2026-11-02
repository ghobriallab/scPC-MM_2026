#!/usr/bin/env Rscript
# ============================================================================
# Purpose:      Translocation classifier (Methods, Classification of cytogenetic alterations). Multinomial logistic
#               regression (nnet), HRD as reference class, on the per-population median min-max scaled Zhan subtype scores
#               and the mean expression of nine genes. Trained on one BM tumor per patient with a translocation or HRD
#               from FISH or WGS: fitted once per tumor, leaving that tumor out; the final model is the median of the
#               fitted coefficients. Calls: top translocation class if its probability is > 0.5, otherwise no
#               translocation. Calls are made per malignant population and tissue, then reviewed against the expression
#               of the partner genes; tumors without a translocation are HRD if Numbat calls HRD, otherwise Other.
#               With --score-only the training step is skipped and the coefficients of
#               translocation_classifier_coefficients.tsv are used.
# Inputs:       data/cytogenetics/classifier_cells.csv: record_id, gex_id, sample_type, annotation, annotation3,
#               CD_1:Norm, CD_2:Norm, MS:Norm, MF:Norm, HP:Norm, MAF, MAFA, MAFB, ITGB7, CCND1, CCND2, CCND3, NSD2, FGFR3.
#               data/cytogenetics/fish_wgs_calls.csv: record_id, Cytogenetics_FISHorWGS, WGS, Cytogenetics.
#               data/cytogenetics/numbat_hrd.csv: record_tumor, Numbat_HRD.
#               data/metadata/tumor-cohort-metadata-final-240325.csv: gex_id, patient_id.
# Outputs:      results/genomics/translocation_classifier_input_matrix.csv: features of every population and tissue.
#               results/genomics/translocation_classifier_loo_coefficients.csv, _loo_predictions.csv: the leave-one-out fits.
#               results/genomics/translocation_classifier_final_coefficients.csv: the final model.
#               results/genomics/translocation_classifier_pb_predictions.csv: final model on PB of the reference tumors.
#               results/genomics/translocation_classifier_all_predictions.csv: every population, before and after review.
# Dependencies: R 4.1 with nnet (environment.R); config.R. Run from this folder.
# ============================================================================
source("../config.R")
suppressMessages(library(nnet))
set.seed(SEED)
OUT <- result_dir("genomics")
score_only <- "--score-only" %in% commandArgs(TRUE)
FEATURES <- c("CD_1", "CD_2", "MS", "MF", "HP", "MAF", "MAFA", "MAFB", "ITGB7", "CCND1", "CCND2", "CCND3", "NSD2", "FGFR3")
CLASSES <- c("t(11;14)", "t(4;14)", "t(14;16)", "t(14;20)", "t(6;14)", "t(8;14)")

# ---- input matrix: median scores and mean expression per population and tissue ----------------------------------------
cells <- read.csv(data_path("cytogenetics", "classifier_cells.csv"), check.names = FALSE, na.strings = "")
tumor <- cells[cells$annotation %in% "Tumor", ]
tumor$Population <- paste0(tumor$record_id, ":", tumor$annotation3, ";", tumor$sample_type)
sigs <- colnames(tumor)[grep("Norm", colnames(tumor))]
sig_df <- aggregate(tumor[, sigs], list(tumor$Population), median, na.rm = TRUE)
genes <- c("CCND1", "CCND2", "CCND3", "FGFR3", "NSD2", "MAF", "MAFA", "MAFB", "ITGB7")
gene_df <- aggregate(tumor[, genes], list(tumor$Population), mean, na.rm = TRUE)
data <- merge(sig_df, gene_df, by = "Group.1")
colnames(data) <- c("Sample", "CD_1", "CD_2", "MS", "MF", "HP", genes)
write.csv(data, file.path(OUT, "translocation_classifier_input_matrix.csv"), row.names = FALSE)

# ---- FISH and WGS calls ----------------------------------------------------------------------------------------------
data$PatientID <- gsub(":.*$", "", data$Sample)
data$PatientID_TumorID <- gsub(";.*$", "", data$Sample)
data$Tissue <- "BM"
data$Tissue[grep("PB", data$Sample)] <- "PB"
fish <- read.csv(data_path("cytogenetics", "fish_wgs_calls.csv"), na.strings = "")
fish <- fish[!duplicated(fish$record_id), ]
stopifnot(all(data$PatientID %in% fish$record_id))
data$FISHorWGS <- fish$Cytogenetics_FISHorWGS[match(data$PatientID, fish$record_id)]
# HRD with another translocation: unclassified, so that a possible IgH translocation is not used as a control
data$FISHorWGS[data$FISHorWGS %in% "HRD, Other_Translocation"] <- "Unclassified"
data$WGS <- fish$WGS[match(data$PatientID, fish$record_id)]
data$WGS[data$WGS %in% "NA"] <- NA
data$WGS_Clean <- ifelse(data$WGS %in% c("gain(1q, 3q, 7, 17q), del(3p,7p,13q)", "gain(1q, 5p, 6p, 19p), del(4p, 6q, 13, 14)",
                                         "gain(9), del(13, 14)", "gain(9), del(13, 14q, 16, 22)", "Other tx"), "Other",
                         as.character(data$WGS))
data$FISH <- fish$Cytogenetics[match(data$PatientID, fish$record_id)]
data$FISH[data$FISH %in% "NA"] <- NA
data$FISH_Clean <- ifelse(data$FISH %in% "t(11;14),HRD", "t(11;14)",
                          ifelse(data$FISH %in% "t(4;14),HRD", "t(4;14)",
                                 ifelse(data$FISH %in% "t(14;20),HRD", "t(14;20)",
                                        ifelse(data$FISH %in% c("t(11;14)_IHC", "HRD,Other_Translocation", "Negative", "Other_Translocation"),
                                               "Unclassified", as.character(data$FISH)))))

# ---- leave-one-out fits and final model -------------------------------------------------------------------------------
TRACKED <- file.path(REPO_DIR, "04_genomics", "translocation_classifier_coefficients.tsv")
tracked <- read.delim(TRACKED, check.names = FALSE)
if (score_only) {
  model_agg <- tracked
} else {
  bm <- data[grep("BM", data$Sample), ]
  train <- bm[!bm$FISHorWGS %in% "Unclassified", ]
  train <- train[-grep("Tumor2", train$Sample), ]                     # the main population of each patient
  train$FISHorWGS <- factor(as.character(train$FISHorWGS), levels = c("HRD", "t(11;14)", "t(6;14)", "t(4;14)", "t(14;16)", "t(14;20)", "t(8;14)"))
  samples <- as.character(train$PatientID_TumorID)
  coefs <- list()
  held <- list()
  for (i in seq_along(samples)) {
    tmp <- train[!train$PatientID_TumorID %in% samples[i], ]
    fit <- multinom(FISHorWGS ~ CD_1 + CD_2 + MS + MF + HP + MAF + MAFA + MAFB + ITGB7 + CCND1 + CCND2 + CCND3 + NSD2 + FGFR3,
                    data = tmp, trace = FALSE)
    tmp2 <- train[train$PatientID_TumorID %in% samples[i], ]
    tmp2$predictions <- predict(fit, newdata = tmp2)
    tmp3 <- data.frame(coef(fit), check.names = FALSE)
    tmp3$model <- paste0("m", i)
    coefs[[i]] <- tmp3
    held[[i]] <- tmp2
  }
  model_coefficients <- do.call(rbind, coefs)
  colnames(model_coefficients)[1] <- "Intercept"
  model_coefficients$Outcome <- gsub("\\).*$", ")", rownames(model_coefficients))
  write.csv(model_coefficients, file.path(OUT, "translocation_classifier_loo_coefficients.csv"), row.names = FALSE)
  write.csv(do.call(rbind, held), file.path(OUT, "translocation_classifier_loo_predictions.csv"), row.names = FALSE)
  cn <- colnames(model_coefficients)[!colnames(model_coefficients) %in% c("Outcome", "model")]
  model_agg <- aggregate(model_coefficients[, cn], list(model_coefficients$Outcome), median)
  colnames(model_agg)[1] <- "feature"
  chk <- model_agg[match(tracked$feature, model_agg$feature), colnames(tracked)[-1]]
  stopifnot(max(abs(as.matrix(chk) - as.matrix(tracked[, -1]))) < 1e-5)
}
write.csv(model_agg, file.path(OUT, "translocation_classifier_final_coefficients.csv"), row.names = FALSE)

# ---- calls: top translocation class if its probability is > 0.5 ---------------------------------------------------------
score <- function(x) {
  lin <- sapply(CLASSES, function(k) {
    b <- model_agg[model_agg$feature == k, ]
    min(as.numeric(b[, "Intercept"]) + sum(as.numeric(x[FEATURES]) * as.numeric(b[, FEATURES])), 709)
  })
  probs <- exp(lin) / (1 + sum(exp(lin)))
  top <- which.max(probs)
  if (probs[top] > 0.5) CLASSES[top] else "No Trx"
}
pb <- data[data$Tissue %in% "PB" & !data$FISHorWGS %in% "Unclassified", ]
pb <- pb[-grep("Tumor2", pb$PatientID_TumorID), ]
pb$predictions <- apply(pb[, FEATURES], 1, score)
write.csv(pb, file.path(OUT, "translocation_classifier_pb_predictions.csv"), row.names = FALSE)
data$model_agg_predictions <- apply(data[, FEATURES], 1, score)

# ---- review against the expression of the partner genes (patients of the cohort) -----------------------------------------
cohort <- read.csv(data_path("metadata", "tumor-cohort-metadata-final-240325.csv"), check.names = FALSE)
rec <- unique(cells[, c("record_id", "gex_id")])
rec$patient <- cohort$patient_id[match(rec$gex_id, cohort$gex_id)]
rec <- rec[!is.na(rec$patient) & !duplicated(rec$record_id), ]
data$patient <- rec$patient[match(data$PatientID, rec$record_id)]
data$clone <- gsub("^.*:", "", data$PatientID_TumorID)
data$predictions <- data$model_agg_predictions
data$predictions[data$patient %in% c("Pt-50", "Pt-170")] <- "t(12;14)"   # very high CCND2; the model has no CCND2 class
data$predictions[data$patient %in% "Pt-153"] <- "t(6;14)"                # high CCND3
data$predictions[data$patient %in% "Pt-136"] <- "No Trx"                 # t(8;14) predicted, no MAFA expression
data$predictions[data$patient %in% "Pt-201"] <- "t(14;16)"               # MAF far above MAFB
data$predictions[data$patient %in% "Pt-141"] <- "t(11;14)"               # BM call kept: CCND1 far above CCND3
data$predictions[data$patient %in% "Pt-111"] <- "t(6;14)"                # BM call kept: very high CCND3
data$predictions[data$patient %in% "Pt-129" & data$clone %in% "Tumor1"] <- "No Trx"   # BM call kept: no MAFB expression

# ---- final group: translocation, otherwise HRD (Numbat) or Other ----------------------------------------------------------
numbat <- read.csv(data_path("cytogenetics", "numbat_hrd.csv"))
data$Numbat_HRD <- numbat$Numbat_HRD[match(data$PatientID_TumorID, numbat$record_tumor)]
data$Numbat_HRD_Final <- ifelse(data$patient %in% "Pt-134" & data$clone %in% "Tumor", "HRD", as.character(data$Numbat_HRD))   # two near-trisomies
data$FinalClassification <- ifelse(data$predictions %in% "No Trx" & data$Numbat_HRD_Final %in% "HRD", "HRD",
                                   ifelse(data$predictions %in% "No Trx" & data$Numbat_HRD_Final %in% "No HRD", "Other",
                                          as.character(data$predictions)))
write.csv(data, file.path(OUT, "translocation_classifier_all_predictions.csv"), row.names = FALSE)
cat("populations called:", nrow(data), "\n")
