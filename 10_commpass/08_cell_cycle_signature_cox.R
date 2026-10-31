# ============================================================================
# Purpose:      Explains the opposite hazard ratios of the G1/S and G2/M cell-cycle signatures in the multivariable Cox
#               model of Fig. 3F (Results). In the patients of that model: correlation of the two signature activities
#               (AUCell, scaled over all patients of the table, as in Fig. 3F; Pearson and Spearman); Cox models of
#               progression-free survival (ttcpfs1/censpfs1, the endpoint of Fig. 3F) and overall survival (ttcos/censos)
#               with each cell-cycle signature alone (unadjusted, and with the proliferation index and ISS stage) and with
#               both together (with the proliferation index and ISS stage); hazard ratios per standard deviation with 95%
#               CI and two-sided Wald P.
# Inputs:       data/commpass/MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv (rows = PUBLIC_ID): auc:<signature>,
#                 auc:CoMMpassPI, D_PT_iss, ttcpfs1, censpfs1, ttcos, censos.
# Outputs:      results/commpass/cell_cycle_signature_cox.tsv: endpoint, model, n, events, term, HR, lo, hi, P.
#               results/commpass/cell_cycle_signature_cox_log.txt: correlations and models.
# Dependencies: R 4.1 with survival (environment.R); sources ../config.R.
# ============================================================================
source("../config.R")      # DATA_DIR, RESULTS_DIR, data_path(), result_dir(), SEED
suppressMessages(library(survival))
OUT <- result_dir("commpass")
logf <- file(file.path(OUT, "cell_cycle_signature_cox_log.txt"), "w")
say <- function(...) { s <- paste(...); cat(s, "\n"); cat(s, "\n", file = logf) }
SIGS <- c("B_cell_activation", "cell_cycling_G1S", "cell_cycling_G2M", "interferon_inducible", "histone_chaperone_gene")

sig <- read.delim(data_path("commpass", "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv"), row.names = 1, check.names = FALSE)

# features scaled over all patients of the table, as in the Cox model of Fig. 3F; patients = those of that model
d <- data.frame(row.names = rownames(sig), pfs_t = sig$ttcpfs1, pfs_e = sig$censpfs1, os_t = sig$ttcos, os_e = sig$censos,
                PI = as.numeric(scale(sig[["auc:CoMMpassPI"]])), ISS = as.numeric(scale(sig$D_PT_iss)))
for (s in SIGS) d[[s]] <- as.numeric(scale(sig[[paste0("auc:", s)]]))
d <- d[complete.cases(d[, c(SIGS, "PI", "ISS")]), ]
say("patients of the Fig. 3F model:", nrow(d))

# correlation of the two cell-cycle signatures
for (m in c("pearson", "spearman")) {
  ct <- cor.test(d$cell_cycling_G1S, d$cell_cycling_G2M, method = m, exact = FALSE)
  say(sprintf("G1/S vs G2/M activity, %s: %.2f (P = %.3g)", m, ct$estimate, ct$p.value))
}

models <- list(G1S_alone = "cell_cycling_G1S", G2M_alone = "cell_cycling_G2M",
               G1S_ISS_PI = "cell_cycling_G1S + PI + ISS", G2M_ISS_PI = "cell_cycling_G2M + PI + ISS",
               both_ISS_PI = "cell_cycling_G1S + cell_cycling_G2M + PI + ISS")
rows <- list()
for (ep in c("pfs", "os")) for (m in names(models)) {
  fit <- coxph(as.formula(paste0("Surv(", ep, "_t, ", ep, "_e) ~ ", models[[m]])), data = d)
  ci <- summary(fit)$conf.int; co <- summary(fit)$coefficients
  rows[[length(rows) + 1]] <- data.frame(endpoint = toupper(ep), model = m, n = fit$n, events = fit$nevent,
                                         term = rownames(ci), HR = ci[, "exp(coef)"], lo = ci[, "lower .95"],
                                         hi = ci[, "upper .95"], P = co[, "Pr(>|z|)"])
}
res <- do.call(rbind, rows)
write.table(res, file.path(OUT, "cell_cycle_signature_cox.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
for (i in seq_len(nrow(res))) with(res[i, ], say(sprintf("%-3s %-12s n=%d ev=%d %-18s HR %.2f (%.2f-%.2f) P=%.3g",
                                                       endpoint, model, n, events, term, HR, lo, hi, P)))
say("done")
