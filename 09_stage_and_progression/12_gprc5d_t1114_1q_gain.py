#!/usr/bin/env python3
"""GPRC5D upregulation in t(11;14) versus other tumors adjusted for 1q gain (Results; Methods, Differential expression
analysis, single-cell level).

Purpose:      GPRC5D expression is higher in tumors with 1q gain, which t(11;14) tumors carry less often. The tumors and
              the upregulation rule are those of 05_differential_expression/16_gprc5d_frequency.py (main malignant clone
              of a patient at one disease stage, BM; malignant vs normal PCs of the same patient, Benjamini-Hochberg
              q < 0.01 and log2 fold change > 1). 1q gain = a Numbat 1q+ driver segment in the patient's tumor, as in
              02_stage_effects_genetics.py. Logistic regression of upregulation on t(11;14) and 1q gain (odds ratios with
              95% CIs, two-sided Wald tests); two-sided Fisher's exact tests of t(11;14) versus other tumors within tumors
              without and with 1q gain.

Inputs:       results/differential_expression/BM_scRNA_tumor_DE_FDR.csv, BM_scRNA_tumor_DE_logFoldChange.csv (rows = genes,
              columns = <patient>_BM_<stage>_tumor_<clone>, clone 0 = main clone); data/paper/cytogenetics-status.csv
              (patient_id, cytogenetics); results/stage_and_progression/bm_tumor_units.tsv (patient_id, "1q+").

Outputs:      results/stage_and_progression/gprc5d_t1114_1q_gain.tsv: counts and tests (logistic regression terms,
              stratified Fisher's exact tests).
              results/stage_and_progression/gprc5d_t1114_1q_gain_log.txt: counts and model summary.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "gprc5d_t1114_1q_gain_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


# ---- 1. tumors and GPRC5D upregulation, as in 05_differential_expression/16_gprc5d_frequency.py -----------------------
fdr = pd.read_csv(result("differential_expression", "BM_scRNA_tumor_DE_FDR.csv"), index_col=0)
lfc = pd.read_csv(result("differential_expression", "BM_scRNA_tumor_DE_logFoldChange.csv"), index_col=0)
main = [c for c in fdr.columns if c.endswith("_0")]
t = pd.DataFrame({"patient": [c.split("_")[0] for c in main]}, index=main)
t["up"] = ((fdr.loc["GPRC5D", main] < 0.01) & (lfc.loc["GPRC5D", main] > 1)).astype(int)
cy = pd.read_csv(data("paper", "cytogenetics-status.csv")).drop_duplicates("patient_id").set_index("patient_id")
t["t1114"] = (t["patient"].map(cy["cytogenetics"]) == "t(11;14)").astype(int)

# ---- 2. 1q gain of the patient's tumor (Numbat driver segment, units of 02_stage_effects_genetics.py) ------------------
units = pd.read_csv(result("stage_and_progression", "bm_tumor_units.tsv"), sep="\t")
gain = units.groupby("patient_id")["1q+"].max()
assert t["patient"].isin(gain.index).all()
t["gain1q"] = t["patient"].map(gain).astype(int)
log("tumors: %d from %d patients" % (len(t), t["patient"].nunique()))
for name, g in (("t(11;14)", t[t["t1114"] == 1]), ("other", t[t["t1114"] == 0])):
    log("%-9s tumors %3d | 1q gain %3d (%.0f%%) | GPRC5D up %3d (%.0f%%)" % (
        name, len(g), g["gain1q"].sum(), 100 * g["gain1q"].mean(), g["up"].sum(), 100 * g["up"].mean()))
for k, g in t.groupby("gain1q"):
    log("1q gain %d: tumors %3d | GPRC5D up %3d (%.0f%%)" % (k, len(g), g["up"].sum(), 100 * g["up"].mean()))

# ---- 3. logistic regression and stratified Fisher's exact tests -------------------------------------------------------
rows = []
fit = sm.Logit(t["up"], sm.add_constant(t[["t1114", "gain1q"]].astype(float))).fit(disp=0)
ci = fit.conf_int()
for term in ("t1114", "gain1q"):
    rows.append(dict(test="logistic regression, GPRC5D up ~ t(11;14) + 1q gain", term=term, n=len(t),
                     odds_ratio=np.exp(fit.params[term]), ci_low=np.exp(ci.loc[term, 0]), ci_high=np.exp(ci.loc[term, 1]),
                     P=fit.pvalues[term]))
    log("%-7s OR %.2f (%.2f-%.2f), P = %.2g" % (term, rows[-1]["odds_ratio"], rows[-1]["ci_low"], rows[-1]["ci_high"],
                                                rows[-1]["P"]))
for k, g in t.groupby("gain1q"):
    a, b = g[g["t1114"] == 1], g[g["t1114"] == 0]
    odds, p = stats.fisher_exact([[a["up"].sum(), len(a) - a["up"].sum()], [b["up"].sum(), len(b) - b["up"].sum()]])
    rows.append(dict(test="two-sided Fisher's exact test, t(11;14) vs other, 1q gain = %d" % k, term="t1114",
                     n="%d vs %d" % (len(a), len(b)), odds_ratio=odds, P=p,
                     up_t1114="%d/%d" % (a["up"].sum(), len(a)), up_other="%d/%d" % (b["up"].sum(), len(b))))
    log("1q gain %d: t(11;14) up %d/%d vs other %d/%d, Fisher P = %.2g" % (
        k, a["up"].sum(), len(a), b["up"].sum(), len(b), p))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "gprc5d_t1114_1q_gain.tsv"), sep="\t", index=False)
log("done")
FH.close()
