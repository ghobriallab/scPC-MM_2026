# ============================================================================
# Purpose:      Refits the multivariable Cox model of the progression-associated signatures in CoMMpass (Fig. 3F) with
#               age and sex, and with age, sex and IMS/IMWG 2025 high-risk status (Results; Fig. 3F; Methods). Model of
#               Fig. 3F: AUCell activities of the five signatures significant by log-rank test, proliferation index and
#               ISS stage, each scaled over all patients of the table. Models: paper (Fig. 3F); age_sex (paper + age
#               per 10 years + sex); paper_imwg_subset (paper, in the patients with an IMS/IMWG 2025 status);
#               age_sex_imwg (age_sex + IMS/IMWG 2025 high risk, same patients). Progression-free survival
#               (ttcpfs1/censpfs1, the endpoint of Fig. 3F) and overall survival (ttcos/censos); hazard ratios with
#               95% CI and two-sided Wald P.
# Inputs:       data/commpass/MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv (rows = PUBLIC_ID): auc:<signature>,
#                 auc:CoMMpassPI, D_PT_iss, ttcpfs1, censpfs1, ttcos, censos.
#               data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT.tsv: PUBLIC_ID, D_PT_age, D_PT_gender (1 male, 2 female).
#               results/commpass/imwg2025_risk.tsv (03_imwg2025_risk.py): patient, IMWG2025 (high_risk, standard_risk,
#                 not_calculable, no_data).
# Outputs:      results/commpass/signature_cox_age_sex.tsv: every coefficient of every model (endpoint, model, n,
#                 events, term, HR, lo, hi, P).
#               results/commpass/signature_cox_age_sex_log.txt: log.
# Dependencies: R 4.1 with survival (environment.R); sources ../config.R.
# ============================================================================
source("../config.R")      # DATA_DIR, RESULTS_DIR, data_path(), result_dir(), SEED
suppressMessages(library(survival))
OUT <- result_dir("commpass")
logf <- file(file.path(OUT, "signature_cox_age_sex_log.txt"), "w")
say <- function(...) { s <- paste(...); cat(s, "\n"); cat(s, "\n", file = logf) }
SIGS <- c("B_cell_activation", "cell_cycling_G1S", "cell_cycling_G2M", "interferon_inducible", "histone_chaperone_gene")

sig <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv"), row.names = 1, check.names = FALSE)
pp <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_PER_PATIENT.tsv")); rownames(pp) <- pp$PUBLIC_ID
risk <- read.delim(file.path(RESULTS_DIR, "commpass", "imwg2025_risk.tsv")); rownames(risk) <- risk$patient
id <- rownames(sig)

# features scaled over all patients of the table, as in the Cox model of Fig. 3F
d <- data.frame(row.names = id, pfs_t = sig$ttcpfs1, pfs_e = sig$censpfs1, os_t = sig$ttcos, os_e = sig$censos,
                PI = as.numeric(scale(sig[["auc:CoMMpassPI"]])), ISS = as.numeric(scale(sig$D_PT_iss)),
                age10 = pp[id, "D_PT_age"] / 10,
                sex = factor(c("Male", "Female")[pp[id, "D_PT_gender"]], levels = c("Male", "Female")),
                imwg_high = factor(unname(c(high_risk = "Yes", standard_risk = "No")[risk[id, "IMWG2025"]]), levels = c("No", "Yes")))
for (s in SIGS) d[[s]] <- as.numeric(scale(sig[[paste0("auc:", s)]]))
base <- paste(c(SIGS, "PI", "ISS"), collapse = " + ")
cc <- complete.cases(d[, c(SIGS, "PI", "ISS")])
sub <- cc & !is.na(d$age10) & !is.na(d$sex) & !is.na(d$imwg_high)
say("patients:", nrow(d), "| complete for the Fig. 3F model:", sum(cc), "| also age, sex and IMS/IMWG 2025 status:", sum(sub))

models <- list(paper = list(base, cc), age_sex = list(paste(base, "+ age10 + sex"), cc),
               paper_imwg_subset = list(base, sub), age_sex_imwg = list(paste(base, "+ age10 + sex + imwg_high"), sub))
rows <- list()
for (ep in c("pfs", "os")) for (m in names(models)) {
  fit <- coxph(as.formula(paste0("Surv(", ep, "_t, ", ep, "_e) ~ ", models[[m]][[1]])), data = d[models[[m]][[2]], ])
  ci <- summary(fit)$conf.int; co <- summary(fit)$coefficients
  rows[[length(rows) + 1]] <- data.frame(endpoint = toupper(ep), model = m, n = fit$n, events = fit$nevent,
                                         term = rownames(ci), HR = ci[, "exp(coef)"], lo = ci[, "lower .95"],
                                         hi = ci[, "upper .95"], P = co[, "Pr(>|z|)"])
}
res <- do.call(rbind, rows)
write.table(res, file.path(OUT, "signature_cox_age_sex.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
for (i in seq_len(nrow(res))) with(res[i, ], say(sprintf("%-3s %-18s n=%d ev=%d %-24s HR %.2f (%.2f-%.2f) P=%.3g",
                                                       endpoint, model, n, events, term, HR, lo, hi, P)))
say("done")
