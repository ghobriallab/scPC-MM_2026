# ============================================================================
# Purpose:      Tests whether the MST score adds prognostic information to clinical and cytogenetic risk factors in
#               newly diagnosed myeloma (CoMMpass, IA22 endpoints; Results, Supplementary Fig. 15A,B and Methods). Cox
#               models of progression-free and overall survival (years = days / 365) with the MST score (per SD over the
#               754 patients) adjusted for age (per 10 years), sex, R2-ISS stage and IMS/IMWG 2025 high-risk status, in
#               the patients with complete data, against the same model without the score: likelihood-ratio test for
#               adding it; change in Harrell's C-index with a 95% CI from 1,000 bootstrap resamples of patients (both
#               models refit in each; set.seed(42)); unadjusted MST hazard ratio in all patients and in the same
#               patients; proportional hazards checked with scaled Schoenfeld residuals (cox.zph). R2-ISS stage from
#               ISS, del(17p), baseline LDH, t(4;14) and 1q gain (D'Agostino et al., J Clin Oncol 2022). Also the
#               relation of the MST score to R2-ISS (two-sided Kruskal-Wallis test), to IMS/IMWG 2025 high risk
#               (two-sided Wilcoxon rank-sum test) and to the proliferation index (Spearman's correlation), and the
#               distribution of the MST scores (range, interquartile range, groups of Fig. 5D at 0.75 and 0.80).
# Inputs:       data/commpass/mm_classifier_results_250313.csv: exr_norm:MM_score (MST score; rows = PUBLIC_ID, 754).
#               data/commpass/MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv (rows = PUBLIC_ID): ttcpfs1, censpfs1 (PFS),
#                 ttcos, censos (OS), auc:CoMMpassPI (proliferation index).
#               data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT.tsv: PUBLIC_ID, D_PT_age, D_PT_gender (1 male, 2 female),
#                 D_PT_iss.
#               data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv: PUBLIC_ID, VISIT (0 = baseline), VISITDY,
#                 LDH_level.
#               results/commpass/imwg2025_risk.tsv (03_imwg2025_risk.py): patient, del17p, gain1q, t414, IMWG2025.
# Outputs:      results/commpass/mst_cox_models.tsv: one row per endpoint and reference model: n, events, MST hazard
#                 ratios, likelihood-ratio test, C-index with and without the MST score, proportional-hazards tests.
#               results/commpass/mst_cox_coefficients.tsv: every coefficient of the models with the MST score.
#               results/commpass/mst_cox_covariates.tsv: covariate counts and age quartiles in the patients analyzed.
#               results/commpass/mst_cox_log.txt: log.
# Dependencies: R 4.1 with survival (environment.R); sources ../config.R.
# ============================================================================
source("../config.R")      # DATA_DIR, RESULTS_DIR, data_path(), result_dir(), SEED
suppressMessages(library(survival))
set.seed(42)

OUT <- result_dir("commpass")
logf <- file(file.path(OUT, "mst_cox_log.txt"), "w")
say <- function(...) { s <- paste(...); cat(s, "\n"); cat(s, "\n", file = logf) }
B <- 1000   # bootstrap resamples for the change in C-index
say(R.version.string, "| survival", format(packageVersion("survival")))

## 1. One row per CoMMpass patient with an MST score, every table joined by PUBLIC_ID ----------
mst <- read.csv(data_path("commpass", "mm_classifier_results_250313.csv"), row.names = 1, check.names = FALSE)
v <- mst[!is.na(mst[, "exr_norm:MM_score"]), "exr_norm:MM_score"]
say(sprintf("MST scores: %d patients, range %.2f-%.2f, interquartile range %.2f-%.2f; >= 0.80: %d, 0.75-0.80: %d, < 0.75: %d",
            length(v), min(v), max(v), quantile(v, 0.25), quantile(v, 0.75), sum(v >= 0.8), sum(v >= 0.75 & v < 0.8),
            sum(v < 0.75)))
sig <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv"), row.names = 1, check.names = FALSE)
pp <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_PER_PATIENT.tsv")); rownames(pp) <- pp$PUBLIC_ID
risk <- read.delim(file.path(RESULTS_DIR, "commpass", "imwg2025_risk.tsv")); rownames(risk) <- risk$patient
id <- rownames(mst)
stopifnot(all(id %in% rownames(sig)), all(id %in% rownames(pp)), all(id %in% rownames(risk)))

# baseline LDH: the VISIT 0 record closest to day 0 that has an LDH call (High = above the upper limit)
vis <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv"))
vis <- vis[vis$VISIT == 0 & vis$LDH_level %in% c("High", "Normal", "Low"), ]
vis <- vis[order(vis$PUBLIC_ID, abs(vis$VISITDY)), ]
vis <- vis[!duplicated(vis$PUBLIC_ID), ]; rownames(vis) <- vis$PUBLIC_ID

d <- data.frame(row.names = id,
  MST = mst[id, "exr_norm:MM_score"],
  age = pp[id, "D_PT_age"],     # complete; DEMOG_PATIENTAGE in the signature table misses 120 patients
  sex = factor(c("Male", "Female")[pp[id, "D_PT_gender"]], levels = c("Male", "Female")),
  iss = pp[id, "D_PT_iss"],
  ldh_high = vis[id, "LDH_level"] == "High",
  del17p = as.logical(risk[id, "del17p"]), t414 = as.logical(risk[id, "t414"]), gain1q = as.logical(risk[id, "gain1q"]),
  imwg = unname(c(high_risk = "Yes", standard_risk = "No")[risk[id, "IMWG2025"]]),   # not_calculable -> NA
  PI = sig[id, "auc:CoMMpassPI"])

# R2-ISS (D'Agostino et al., J Clin Oncol 2022): ISS II 1, ISS III 1.5, del(17p) 1, high LDH 1,
# t(4;14) 1, 1q gain 0.5; stage I = 0, II = 0.5-1, III = 1.5-2.5, IV = 3-5 points
pts <- c(0, 1, 1.5)[d$iss] + d$del17p + d$ldh_high + d$t414 + 0.5 * d$gain1q
d$r2iss <- cut(pts, c(-Inf, 0.25, 1.25, 2.75, Inf), labels = c("I", "II", "III", "IV"))
d$imwg <- factor(d$imwg, levels = c("No", "Yes"))

# continuous covariates scaled over the 754 patients (as the features of the Cox model of Fig. 3F)
d$MST_sd <- as.numeric(scale(d$MST))
d$age10 <- d$age / 10
say("patients:", nrow(d), "| MST SD:", round(sd(d$MST), 4), "| missing: R2-ISS", sum(is.na(d$r2iss)), "(ISS", sum(is.na(d$iss)),
    ", LDH", sum(is.na(d$ldh_high)), ", copy number/translocation", sum(is.na(d$del17p) | is.na(d$t414) | is.na(d$gain1q)),
    ") | IMS/IMWG 2025", sum(is.na(d$imwg)))

# how the MST score relates to the covariates
ct <- cor.test(d$MST, d$PI, method = "spearman", exact = FALSE)
say("MST vs proliferation index: Spearman rho", round(ct$estimate, 3), "P", signif(ct$p.value, 3))
say("MST median by R2-ISS:", paste(levels(d$r2iss), round(tapply(d$MST, d$r2iss, median), 3), collapse = " "),
    "| Kruskal-Wallis P", signif(kruskal.test(MST ~ r2iss, d)$p.value, 3))
say("MST median by IMS/IMWG 2025 high risk:", paste(levels(d$imwg), round(tapply(d$MST, d$imwg, median), 3), collapse = " "),
    "| Wilcoxon P", signif(wilcox.test(MST ~ imwg, d)$p.value, 3))

## 2. Cox models -----------------------------------------------------------------------------
# endpoints in years (days / 365, as in Fig. 5D)
endpoints <- list(
  "IA22 PFS" = data.frame(time = sig[id, "ttcpfs1"], event = sig[id, "censpfs1"]),
  "IA22 OS" = data.frame(time = sig[id, "ttcos"], event = sig[id, "censos"]))
refs <- list(
  "age, sex, R2-ISS, IMS/IMWG 2025" = c("age10", "sex", "r2iss", "imwg"))

cidx <- function(f) as.numeric(concordance(f)$concordance)
hr <- function(f, term) {
  s <- summary(f)
  c(hr = s$conf.int[term, "exp(coef)"], lo = s$conf.int[term, "lower .95"], hi = s$conf.int[term, "upper .95"], p = s$coefficients[term, "Pr(>|z|)"])
}

rows <- list(); coefs <- list(); fits <- list()
for (e in names(endpoints)) {
  y <- endpoints[[e]]; y$time <- y$time / 365
  uni <- coxph(Surv(time, event) ~ MST_sd, data = cbind(y, d))
  for (r in names(refs)) {
    covs <- refs[[r]]
    x <- cbind(y, d[, c(covs, "MST_sd")]); x <- x[complete.cases(x), ]   # same patients in both models
    f0 <- coxph(as.formula(paste("Surv(time, event) ~", paste(covs, collapse = " + "))), data = x)
    f1 <- coxph(as.formula(paste("Surv(time, event) ~", paste(c(covs, "MST_sd"), collapse = " + "))), data = x)
    a <- anova(f0, f1); lrt <- a[2, grep("^P", colnames(a))]   # column is P(>|Chi|) or Pr(>|Chi|) by version
    # change in C-index: refit both models in each bootstrap resample of patients
    dC <- replicate(B, {
      b <- x[sample(nrow(x), replace = TRUE), ]
      tryCatch(cidx(coxph(formula(f1), data = b)) - cidx(coxph(formula(f0), data = b)), error = function(err) NA)
    })
    zph <- cox.zph(f1)$table
    uc <- coxph(Surv(time, event) ~ MST_sd, data = x)   # unadjusted, in the same patients as the adjusted model
    hu <- hr(uni, "MST_sd"); ha <- hr(f1, "MST_sd"); hc <- hr(uc, "MST_sd")
    rows[[length(rows) + 1]] <- data.frame(endpoint = e, reference_model = r,
      n_all = uni$n, events_all = uni$nevent, uni_hr_per_sd = hu[["hr"]], uni_lo = hu[["lo"]], uni_hi = hu[["hi"]], uni_p = hu[["p"]],
      n = f1$n, events = f1$nevent, uni_same_patients_hr = hc[["hr"]], uni_same_patients_lo = hc[["lo"]],
      uni_same_patients_hi = hc[["hi"]], uni_same_patients_p = hc[["p"]], adj_hr_per_sd = ha[["hr"]], adj_lo = ha[["lo"]], adj_hi = ha[["hi"]], adj_p = ha[["p"]],
      adj_hr_per_0.1 = ha[["hr"]]^(0.1 / sd(d$MST)), lrt_p = lrt,
      c_reference = cidx(f0), c_with_mst = cidx(f1), delta_c = cidx(f1) - cidx(f0),
      delta_c_lo = quantile(dC, 0.025, na.rm = TRUE), delta_c_hi = quantile(dC, 0.975, na.rm = TRUE), boot_ok = sum(!is.na(dC)),
      ph_mst_p = zph["MST_sd", "p"], ph_global_p = zph["GLOBAL", "p"], row.names = NULL)
    s <- summary(f1)
    coefs[[length(coefs) + 1]] <- data.frame(endpoint = e, reference_model = r, term = rownames(s$conf.int),
      hr = s$conf.int[, "exp(coef)"], lo = s$conf.int[, "lower .95"], hi = s$conf.int[, "upper .95"], p = s$coefficients[, "Pr(>|z|)"], row.names = NULL)
    fits[[paste(e, r)]] <- x
    say(sprintf("%s | %s | n %d, %d events | MST HR/SD %.2f (%.2f-%.2f) P %.2g [univariable %.2f (%.2f-%.2f) P %.2g]  [same patients unadjusted %.2f (%.2f-%.2f) P %.2g] | LRT P %.2g | C %.3f -> %.3f (delta %.3f, %.3f to %.3f) | PH P %.2g",
      e, r, f1$n, f1$nevent, ha[["hr"]], ha[["lo"]], ha[["hi"]], ha[["p"]], hu[["hr"]], hu[["lo"]], hu[["hi"]], hu[["p"]], hc[["hr"]], hc[["lo"]], hc[["hi"]], hc[["p"]], lrt,
      cidx(f0), cidx(f1), cidx(f1) - cidx(f0), quantile(dC, 0.025, na.rm = TRUE), quantile(dC, 0.975, na.rm = TRUE), zph["MST_sd", "p"]))
  }
}
write.table(do.call(rbind, rows), file.path(OUT, "mst_cox_models.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(do.call(rbind, coefs), file.path(OUT, "mst_cox_coefficients.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# covariate counts among the patients of the main model
x <- fits[["IA22 PFS age, sex, R2-ISS, IMS/IMWG 2025"]]
cv <- rbind(data.frame(variable = "sex", level = levels(x$sex), n = as.vector(table(x$sex))),
  data.frame(variable = "R2-ISS", level = levels(x$r2iss), n = as.vector(table(x$r2iss))),
  data.frame(variable = "IMS/IMWG 2025 high risk", level = levels(x$imwg), n = as.vector(table(x$imwg))),
  data.frame(variable = "age, years", level = c("median", "Q1", "Q3"), n = quantile(x$age10 * 10, c(0.5, 0.25, 0.75))))
write.table(cv, file.path(OUT, "mst_cox_covariates.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
say("main model patients:", nrow(x), "| covariates:", paste(cv$variable, cv$level, cv$n, collapse = "; "))
close(logf)
