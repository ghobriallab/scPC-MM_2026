# ============================================================================
# Purpose:      Sensitivity of the CoMMpass survival associations to self-reported race (Methods). Race (D_PT_race:
#               White versus other; patients without a recorded race left out) is added to (1) the MST model of
#               04_mst_cox_risk_models.R: MST score (per SD) + age + sex + R2-ISS + IMS/IMWG 2025 high risk, and (2) the
#               signature model of Fig. 3F with age and sex (05_signature_cox_age_sex.R): five signatures +
#               proliferation index + ISS + age + sex. Each model is fit with and without race in the same patients.
#               Progression-free (ttcpfs1/censpfs1) and overall survival (ttcos/censos), IA22; hazard ratios with 95% CI
#               and two-sided Wald P.
# Inputs:       data/commpass/mm_classifier_results_250313.csv: exr_norm:MM_score (MST score; 754 patients).
#               data/commpass/MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv: auc:<signature>, auc:CoMMpassPI, D_PT_iss,
#                 ttcpfs1, censpfs1, ttcos, censos.
#               data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT.tsv: PUBLIC_ID, D_PT_age, D_PT_gender (1 male, 2 female),
#                 D_PT_iss, D_PT_race (1 White; 2, 3, 4, 6 other races).
#               data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv: PUBLIC_ID, VISIT, VISITDY, LDH_level (baseline
#                 LDH for R2-ISS, as in 04_mst_cox_risk_models.R).
#               results/commpass/imwg2025_risk.tsv (03_imwg2025_risk.py): patient, del17p, gain1q, t414, IMWG2025.
# Outputs:      results/commpass/cox_race.tsv: hazard ratio of the MST score or of each signature, and of race, with
#                 and without race (endpoint, model, adjustment, n, events, term, HR, lo, hi, P).
#               results/commpass/cox_race_log.txt: log.
# Dependencies: R 4.1 with survival (environment.R); sources ../config.R.
# ============================================================================
source("../config.R")      # DATA_DIR, RESULTS_DIR, data_path(), result_dir(), SEED
suppressMessages(library(survival))
OUT <- result_dir("commpass")
logf <- file(file.path(OUT, "cox_race_log.txt"), "w")
say <- function(...) { s <- paste(...); cat(s, "\n"); cat(s, "\n", file = logf) }
SIGS <- c("B_cell_activation", "cell_cycling_G1S", "cell_cycling_G2M", "interferon_inducible", "histone_chaperone_gene")

mst <- read.csv(data_path("commpass", "mm_classifier_results_250313.csv"), row.names = 1, check.names = FALSE)
sig <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv"), row.names = 1, check.names = FALSE)
pp <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_PER_PATIENT.tsv")); rownames(pp) <- pp$PUBLIC_ID
risk <- read.delim(file.path(RESULTS_DIR, "commpass", "imwg2025_risk.tsv")); rownames(risk) <- risk$patient
vis <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv"))
vis <- vis[vis$VISIT == 0 & vis$LDH_level %in% c("High", "Normal", "Low"), ]
vis <- vis[order(vis$PUBLIC_ID, abs(vis$VISITDY)), ]
vis <- vis[!duplicated(vis$PUBLIC_ID), ]; rownames(vis) <- vis$PUBLIC_ID

# 1. one row per patient, covariates built as in 04_mst_cox_risk_models.R (MST model) and 05_signature_cox_age_sex.R
#    (signature model) ---------------------------------------------------------------------------------------------
id <- rownames(mst)
d <- data.frame(row.names = id, pfs_t = sig[id, "ttcpfs1"], pfs_e = sig[id, "censpfs1"], os_t = sig[id, "ttcos"],
                os_e = sig[id, "censos"], MST = mst[id, "exr_norm:MM_score"], age10 = pp[id, "D_PT_age"] / 10,
                sex = factor(c("Male", "Female")[pp[id, "D_PT_gender"]], levels = c("Male", "Female")),
                iss = pp[id, "D_PT_iss"], ldh_high = vis[id, "LDH_level"] == "High",
                del17p = as.logical(risk[id, "del17p"]), t414 = as.logical(risk[id, "t414"]),
                gain1q = as.logical(risk[id, "gain1q"]),
                imwg = factor(unname(c(high_risk = "Yes", standard_risk = "No")[risk[id, "IMWG2025"]]), levels = c("No", "Yes")),
                race = factor(ifelse(is.na(pp[id, "D_PT_race"]), NA, ifelse(pp[id, "D_PT_race"] == 1, "White", "Other")),
                              levels = c("White", "Other")))
pts <- c(0, 1, 1.5)[d$iss] + d$del17p + d$ldh_high + d$t414 + 0.5 * d$gain1q
d$r2iss <- cut(pts, c(-Inf, 0.25, 1.25, 2.75, Inf), labels = c("I", "II", "III", "IV"))
d$MST_sd <- as.numeric(scale(d$MST))
d$PI <- as.numeric(scale(sig[id, "auc:CoMMpassPI"]))
d$ISS <- as.numeric(scale(sig[id, "D_PT_iss"]))
for (s in SIGS) d[[s]] <- as.numeric(scale(sig[id, paste0("auc:", s)]))
say("patients:", nrow(d), "| race White", sum(d$race == "White", na.rm = TRUE), "other", sum(d$race == "Other", na.rm = TRUE),
    "missing", sum(is.na(d$race)))

# 2. models with and without race, same patients ---------------------------------------------------------------------
models <- list(MST = list(c("MST_sd", "age10", "sex", "r2iss", "imwg"), "MST_sd"),
               signatures = list(c(SIGS, "PI", "ISS", "age10", "sex"), SIGS))
rows <- list()
for (ep in c("pfs", "os")) for (m in names(models)) {
  covs <- models[[m]][[1]]
  x <- d[complete.cases(d[, c(covs, "race", paste0(ep, c("_t", "_e")))]), ]
  for (adj in c("without race", "with race")) {
    f <- as.formula(paste0("Surv(", ep, "_t, ", ep, "_e) ~ ", paste(c(covs, if (adj == "with race") "race"), collapse = " + ")))
    s <- summary(coxph(f, data = x))
    for (term in c(models[[m]][[2]], if (adj == "with race") "raceOther")) {
      rows[[length(rows) + 1]] <- data.frame(endpoint = toupper(ep), model = m, adjustment = adj, n = s$n, events = s$nevent,
                                             term = term, HR = s$conf.int[term, "exp(coef)"], lo = s$conf.int[term, "lower .95"],
                                             hi = s$conf.int[term, "upper .95"], P = s$coefficients[term, "Pr(>|z|)"])
    }
  }
}
res <- do.call(rbind, rows)
write.table(res, file.path(OUT, "cox_race.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
for (i in seq_len(nrow(res))) with(res[i, ], say(sprintf("%-3s %-10s %-12s n=%d ev=%d %-24s HR %.2f (%.2f-%.2f) P=%.3g",
                                                       endpoint, model, adjustment, n, events, term, HR, lo, hi, P)))
say("done")
