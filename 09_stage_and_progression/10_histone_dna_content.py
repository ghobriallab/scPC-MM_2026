#!/usr/bin/env python3
"""Histone/chaperone signature activity and tumor DNA content (Supplementary Fig. 16B-D; Results).

Purpose:      Tests whether the stage association of the histone/chaperone signature reflects the amount of DNA in the
              tumor cells. Units: the BM tumors of 02_stage_effects_genetics.py (malignant BM PCs of one patient at one
              disease stage, IMWG 2/20/20 risk groups from risk.py). DNA content: tumor ploidy estimated from whole-genome
              sequencing (WGS) of the same BM sample. Tests: Spearman correlation of ploidy with disease stage (coded 0-4,
              MGUS to MM) and with histone/chaperone activity, overall and within stage (partial correlation: Pearson
              correlation of the ranks after removing the stage means); linear regression of histone/chaperone activity
              (z-scored) on stage without and with ploidy (z-scored) in the tumors with WGS, and without and with HRD in
              all tumors, standard errors clustered by patient, two-sided Wald tests; HRD versus other tumors, two-sided
              Wilcoxon rank-sum test.

Inputs:       results/stage_and_progression/bm_tumor_units.tsv (02_stage_effects_genetics.py: patient_id,
              disease_status, mean_expr:histone_chaperone_gene, HRD); data/paper/wgs-ploidy-purity.csv (patient_id,
              disease_status as recorded at sampling, ploidy, purity of the WGS samples that are a scRNA-seq BM sample).

Outputs:      results/stage_and_progression/histone_dna_content_units.tsv: one row per BM tumor with WGS.
              results/stage_and_progression/histone_dna_content.tsv: stage coefficient per model (n, SD per stage step,
              95% CI, P).
              results/stage_and_progression/histone_dna_content_tests.tsv: correlation and rank-sum tests.
              results/stage_and_progression/histone_dna_content_log.txt: units and summaries.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir
from risk import stages

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "histone_dna_content_log.txt"), "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
H = "mean_expr:histone_chaperone_gene"
TESTS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def add_test(name, detail, n, stat, p):
    TESTS.append(dict(test=name, detail=detail, n=n, statistic=stat, P=p))
    log("  TEST %s | %s | n=%d | stat=%.3f | P=%.3g" % (name, detail, n, stat, p))


def z(s):
    return (s - s.mean()) / s.std()


def stage_fit(d, covs, label):
    """Histone/chaperone activity (z-scored within d) on stage 0-4 (+ covariates); SE clustered by patient."""
    X = sm.add_constant(d[["stage"] + covs].astype(float))
    groups = d["patient_id"].astype("category").cat.codes.values
    r = sm.OLS(z(d[H]), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    lo, hi = r.conf_int().loc["stage"]
    row = dict(model=label, n_units=len(d), n_patients=d["patient_id"].nunique(), coef_sd_per_stage=r.params["stage"],
               ci_low=lo, ci_high=hi, P=r.pvalues["stage"])
    log("  %-40s n=%d  %.3f (%.3f to %.3f)  P=%.3g" % (label, len(d), row["coef_sd_per_stage"], lo, hi, row["P"]))
    return row


def partial_spearman(d, x, y):
    """Correlation of the ranks of x and y after removing the mean rank of each disease stage."""
    D = sm.add_constant(pd.get_dummies(d["disease_status"]).iloc[:, 1:].astype(float))
    rx = sm.OLS(rankdata(d[x]), D).fit().resid
    ry = sm.OLS(rankdata(d[y]), D).fit().resid
    return pearsonr(rx, ry)


# ---- units and WGS ploidy ------------------------------------------------------------------------
units = pd.read_csv(result("stage_and_progression", "bm_tumor_units.tsv"), sep="\t")
units["stage"] = units["disease_status"].map(STAGES.index)
assert len(units) == 181 and units["stage"].notna().all()
wgs = pd.read_csv(data("paper", "wgs-ploidy-purity.csv"))
wgs["disease_status"] = stages(wgs["patient_id"], ["BM"] * len(wgs), wgs["disease_status"])     # IMWG 2/20/20 groups
m = wgs.merge(units, on=["patient_id", "disease_status"], how="inner")
log("WGS samples that are a scRNA-seq BM sample: %d (%d patients); BM tumors with at least 30 malignant PCs among them: %d"
    % (len(wgs), wgs["patient_id"].nunique(), len(m)))
log("units by stage:", m["disease_status"].value_counts().reindex(STAGES).fillna(0).astype(int).to_dict())
log("ploidy: median %.2f, range %.2f-%.2f; HRD %.2f vs other %.2f (medians)" % (
    m["ploidy"].median(), m["ploidy"].min(), m["ploidy"].max(), m.loc[m["HRD"] == 1, "ploidy"].median(),
    m.loc[m["HRD"] == 0, "ploidy"].median()))
log("ploidy by stage (median):", m.groupby("disease_status")["ploidy"].median().reindex(STAGES).round(2).to_dict())

# ---- tests -----------------------------------------------------------------------------------------
log("\ncorrelations")
r, p = spearmanr(m["ploidy"], m["stage"])
add_test("Spearman", "ploidy vs disease stage (0-4)", len(m), r, p)
r, p = spearmanr(m["ploidy"], m[H])
add_test("Spearman", "ploidy vs histone/chaperone activity", len(m), r, p)
r, p = partial_spearman(m, "ploidy", H)
add_test("Spearman (partial, within stage)", "ploidy vs histone/chaperone activity", len(m), r, p)
r, p = spearmanr(units["HRD"], units["stage"])
add_test("Spearman", "HRD vs disease stage (0-4), all tumors", len(units), r, p)
u = mannwhitneyu(units.loc[units["HRD"] == 1, H], units.loc[units["HRD"] == 0, H], alternative="two-sided")
add_test("Wilcoxon rank-sum", "histone/chaperone activity, HRD vs other tumors (all)", len(units), u.statistic, u.pvalue)

log("\nstage association of histone/chaperone activity (SD per stage step)")
m["ploidy_z"] = z(m["ploidy"])
rows = [stage_fit(m, [], "WGS tumors, unadjusted"), stage_fit(m, ["ploidy_z"], "WGS tumors, adjusted for ploidy"),
        stage_fit(units, [], "All tumors, unadjusted"), stage_fit(units, ["HRD"], "All tumors, adjusted for HRD")]
log("retained: ploidy %.0f%%, HRD %.0f%%" % (100 * rows[1]["coef_sd_per_stage"] / rows[0]["coef_sd_per_stage"],
                                           100 * rows[3]["coef_sd_per_stage"] / rows[2]["coef_sd_per_stage"]))

m[["patient_id", "disease_status", "ploidy", "purity", "HRD", H]].to_csv(
    os.path.join(OUT, "histone_dna_content_units.tsv"), sep="\t", index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "histone_dna_content.tsv"), sep="\t", index=False)
pd.DataFrame(TESTS).to_csv(os.path.join(OUT, "histone_dna_content_tests.tsv"), sep="\t", index=False)
FH.close()
