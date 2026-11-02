#!/usr/bin/env python3
"""Progression analyses of SMM restricted to non-progressors with longer follow-up (sensitivity analysis; Results; Methods).

Purpose:      Patients who did not progress may not have been followed long enough to progress. The SMM patients of
              07_smm_progression.py are re-analyzed keeping every progressor but only the non-progressors followed for
              at least 3 years (primary cut-off, fixed beforehand) or at least 4 years after the sample (follow-up =
              year of last annotation minus year of the earliest BM sample of the unit, as in 07_smm_progression.py;
              only years are recorded; non-progressors without a recorded year are kept only in the analysis without a
              cut-off). For the mean MST score and the percentage of MM-like malignant PCs (MST score
              > 0.5; 09_mm_like_cells_statistics.py), and for the 2/20/20 risk group (coded 0-2): progressors versus
              non-progressors (two-sided Wilcoxon rank-sum test), AUROC with a 95% CI from 2,000 stratified bootstrap
              resamples (seed 42), and the likelihood-ratio test for adding the score to the risk group (logistic
              regression on IR and HR indicators). The analysis without a cut-off must reproduce
              smm_progression_results.tsv and mm_like_cells_smm.tsv.

Inputs:       results/stage_and_progression/smm_progression_patient_scores.tsv (patient_id, disease_status, mst,
              progressed); results/stage_and_progression/mm_like_cells_tumors.tsv (patient_id, disease_status,
              frac_mm_like); results/stage_and_progression/smm_progression_results.tsv and mm_like_cells_smm.tsv
              (checks); data/paper/cohort-clinical-metadata.csv (patient_id, disease_status, sample_type, sample_date,
              last_annotation_year).

Outputs:      results/stage_and_progression/smm_progression_followup.tsv: one row per cut-off and score: patients,
              progressors, non-progressors by risk group, Wilcoxon P, AUROC with 95% CI, likelihood-ratio P.
              results/stage_and_progression/smm_progression_followup_log.txt: counts and results.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir
from risk import stages

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

OUT = str(result_dir("stage_and_progression"))
logfile = open(os.path.join(OUT, "smm_progression_followup_log.txt"), "w")


def log(*a):
    print(*a)
    print(*a, file=logfile, flush=True)


SMM = ["LRSMM", "IRSMM", "HRSMM"]
RISK = {g: i for i, g in enumerate(SMM)}

# ---- 1. Patients, scores and follow-up -------------------------------------------------------------
d = pd.read_csv(result("stage_and_progression", "smm_progression_patient_scores.tsv"), sep="\t")
t = pd.read_csv(result("stage_and_progression", "mm_like_cells_tumors.tsv"), sep="\t")
d = d.merge(t[["patient_id", "disease_status", "frac_mm_like"]], on=["patient_id", "disease_status"], how="left")
d["pct"] = 100 * d["frac_mm_like"]
assert d["pct"].notna().all() and len(d) == 115
# follow-up as in 07_smm_progression.py: year of last annotation minus year of the earliest BM sample of the unit
pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
pub["sample_date"] = pd.to_datetime(pub["sample_date"], errors="coerce")
bm = pub[pub["sample_type"] == "BM"]
bm = bm.assign(disease_status=stages(bm["patient_id"], bm["sample_type"], bm["disease_status"]))
first_date = bm.groupby(["patient_id", "disease_status"])["sample_date"].min().rename("date")
d = d.join(first_date, on=["patient_id", "disease_status"])
d["followup"] = d["patient_id"].map(pub.groupby("patient_id")["last_annotation_year"].max()) - d["date"].dt.year
log("non-progressors without a recorded year of last annotation (kept only without a cut-off): %s"
    % ", ".join(d.loc[d["followup"].isna(), "patient_id"]))
assert d.loc[d["progressed"] == 1, "followup"].notna().all()
d["risk"] = d["disease_status"].map(RISK)
log("non-progressors' follow-up (years): %s" % d.loc[d["progressed"] == 0, "followup"].value_counts().sort_index().to_dict())
log("progressors' follow-up to last annotation (years): %s" % d.loc[d["progressed"] == 1, "followup"].value_counts().sort_index().to_dict())


# ---- 2. Statistics for each cut-off ------------------------------------------------------------------
def analyze(x, cut):
    rng = np.random.default_rng(42)
    y = x["progressed"].values
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    X = pd.DataFrame({"IR": (x["risk"] == 1).astype(int).values, "HR": (x["risk"] == 2).astype(int).values})
    clin = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    out = []
    for name, v, per in [("MST score", x["mst"].values, 10), ("MM-like cells (%)", x["pct"].values, 0.1),
                         ("2/20/20 risk group", x["risk"].values.astype(float), None)]:
        boot = [roc_auc_score(y[i], v[i]) for i in
                (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(2000))]
        r = dict(cutoff_years=cut, score=name, patients=len(x), progressed=int(y.sum()),
                 not_progressed=int((y == 0).sum()),
                 not_progressed_by_risk="/".join(str(int(((x["risk"] == k) & (y == 0)).sum())) for k in range(3)),
                 median_progressed=np.median(v[y == 1]), median_not_progressed=np.median(v[y == 0]),
                 wilcoxon_p=stats.mannwhitneyu(v[y == 1], v[y == 0], alternative="two-sided").pvalue,
                 auroc=roc_auc_score(y, v), auroc_lo=np.percentile(boot, 2.5), auroc_hi=np.percentile(boot, 97.5))
        if per is not None:
            full = sm.Logit(y, sm.add_constant(X.assign(s=v * per))).fit(disp=0)
            r["lrt_p_added_to_risk_group"] = stats.chi2.sf(2 * (full.llf - clin.llf), 1)
        out.append(r)
    return out


rows = []
for cut in [0, 3, 4]:
    x = d if cut == 0 else d[(d["progressed"] == 1) | (d["followup"] >= cut)].reset_index(drop=True)
    rows += analyze(x, cut)
res = pd.DataFrame(rows)

# without a cut-off, the results of 07_smm_progression.py and 09_mm_like_cells_statistics.py are reproduced
a = res[res["cutoff_years"] == 0].set_index("score")
r07 = pd.read_csv(result("stage_and_progression", "smm_progression_results.tsv"), sep="\t").iloc[0]
r09 = pd.read_csv(result("stage_and_progression", "mm_like_cells_smm.tsv"), sep="\t").set_index("group").loc["all SMM"]
assert abs(a.loc["MST score", "wilcoxon_p"] - r07["wilcoxon_p"]) < 1e-9
assert abs(a.loc["MST score", "lrt_p_added_to_risk_group"] - r07["lrt_p_adding_mst"]) < 1e-6
assert abs(a.loc["MST score", "auroc"] - r07["auroc_mst"]) < 1e-9
assert abs(a.loc["2/20/20 risk group", "auroc"] - r07["auroc_2_20_20"]) < 1e-9
assert abs(a.loc["MM-like cells (%)", "wilcoxon_p"] - r09["P_progressed_vs_not"]) < 1e-9
assert abs(a.loc["MM-like cells (%)", "lrt_p_added_to_risk_group"] - r09["LRT_P_added_to_risk_group"]) < 1e-6
assert abs(a.loc["MM-like cells (%)", "auroc"] - r09["auroc_pct"]) < 1e-9
res.to_csv(os.path.join(OUT, "smm_progression_followup.tsv"), sep="\t", index=False)
with pd.option_context("display.width", 220, "display.max_columns", 20):
    log(res.round(4).to_string(index=False))
log("done (cut-off 0 reproduces smm_progression_results.tsv and mm_like_cells_smm.tsv)")
