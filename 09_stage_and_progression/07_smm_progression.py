#!/usr/bin/env python3
"""MST scores of SMM patients who later progressed to MM versus those who did not, and comparison with the IMWG 2/20/20
risk group (Fig. 5A; Extended Data Fig. 8A,B; Results).

Purpose:      One unit per SMM patient: the mean MST score of the malignant BM PCs of the patient at one IMWG 2/20/20
              risk group (patient-level score table, risk.py; units that share a risk group are pooled, weighted by
              malignant PCs); patients sampled at two risk groups keep the earliest (date of the earliest BM sample of
              each unit). A patient progressed if any of their libraries has progression_stage "MM" (the
              progression track of Fig. 1B). Progressors and non-progressors are compared by a two-sided Wilcoxon
              rank-sum test (also within IRSMM and HRSMM); AUROC of the MST score and of the 2/20/20 risk group with
              95% CIs from 2,000 stratified bootstrap resamples (progressors and non-progressors resampled separately;
              seed 42), and the difference of the two AUROCs with a 95% CI from 2,000 paired resamples of the same kind; logistic regression of progression on the risk group (IR and HR indicators) with and without
              the MST score (per 0.1), likelihood-ratio test for adding the MST score; follow-up of non-progressors in
              years (year of last annotation minus year of sampling).

Inputs:       data/paper/classifier/patient_id_mean_probability.pkl (sample, disease_status, sample_type, annotation,
              "MST Score"); data/paper/cohort-clinical-metadata.csv (patient_id, disease_status, sample_type,
              sample_date, n_tumor_cell, progression_stage, last_annotation_year).

Outputs:      results/stage_and_progression/smm_progression_results.tsv: one row: patients, progressors, medians, Wilcoxon P, AUROCs with 95% CIs, odds ratio, likelihood-ratio P, within-group P, follow-up.
              results/stage_and_progression/smm_progression_patient_scores.tsv: one row per SMM patient: disease_status, MST score, progressed.
              results/stage_and_progression/smm_progression_log.txt: counts and results.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

SEED = 42
rng = np.random.default_rng(SEED)
OUT = str(result_dir("stage_and_progression"))
logfile = open(os.path.join(OUT, "smm_progression_log.txt"), "w")


def log(*a):
    print(*a)
    print(*a, file=logfile, flush=True)


SMM = ["LRSMM", "IRSMM", "HRSMM"]
RISK = {"LRSMM": 0, "IRSMM": 1, "HRSMM": 2}

# ---- 1. Patient-level MST scores: malignant BM PCs of SMM patients ------------------------------
mst = pd.read_pickle(data("paper", "classifier", "patient_id_mean_probability.pkl"))
mst = mst[mst["disease_status"].isin(SMM) & (mst["sample_type"] == "BM") & (mst["annotation"] == "tumor")]
mst = mst[["sample", "disease_status", "MST Score"]].rename(columns={"sample": "patient_id", "MST Score": "mst"})

# ---- 2. Sampling date and progression from the publication table ---------------------------------
pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
pub["sample_date"] = pd.to_datetime(pub["sample_date"], errors="coerce")
bm = pub[pub["sample_type"] == "BM"]
# IMWG risk groups: units of one patient that share a group are pooled (weights = malignant BM PCs of each unit)
w = bm.groupby(["patient_id", "disease_status"])["n_tumor_cell"].sum()
mst["w"] = [w[k] for k in zip(mst["patient_id"], mst["disease_status"])]
mst["disease_status"] = stages(mst["patient_id"], ["BM"] * len(mst), mst["disease_status"])
mst = mst.groupby(["patient_id", "disease_status"]).apply(lambda g: np.average(g["mst"], weights=g["w"])).rename("mst").reset_index()
bm = bm.assign(disease_status=stages(bm["patient_id"], bm["sample_type"], bm["disease_status"]))
# date of each patient x stage unit = its earliest BM sample at that stage
first_date = bm.groupby(["patient_id", "disease_status"])["sample_date"].min().rename("date")
mst = mst.join(first_date, on=["patient_id", "disease_status"])
# a patient progressed if any of their libraries has progression_stage == "MM" (as in Fig. 1B)
progressed = set(pub.loc[pub["progression_stage"] == "MM", "patient_id"])
last_year = pub.groupby("patient_id")["last_annotation_year"].max()

# Patients sampled at two SMM risk stages keep their earliest one
d = mst.sort_values(["date", "patient_id"]).drop_duplicates("patient_id").copy()
d["progressed"] = d["patient_id"].isin(progressed).astype(int)
d["risk"] = d["disease_status"].map(RISK)
y = d["progressed"].values
log("SMM units in the patient-level table: %d, patients: %d, progressed: %d" % (len(mst), len(d), y.sum()))
for s in SMM:
    log("  %s: %d/%d progressed" % (s, d.loc[d["disease_status"] == s, "progressed"].sum(), (d["disease_status"] == s).sum()))


# ---- 3. Discrimination and added value over the 2/20/20 group ------------------------------------
def auroc_ci(y, score, n=2000):
    """AUROC with a stratified bootstrap 95% CI (progressors and non-progressors resampled separately)."""
    score = np.asarray(score, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = [roc_auc_score(y[i], score[i]) for i in
            (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n))]
    return roc_auc_score(y, score), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


r = {"n_patients": len(d), "n_progressed": int(y.sum())}
r["median_mst_progressed"], r["median_mst_not_progressed"] = d.loc[y == 1, "mst"].median(), d.loc[y == 0, "mst"].median()
r["wilcoxon_p"] = stats.mannwhitneyu(d.loc[y == 1, "mst"], d.loc[y == 0, "mst"], alternative="two-sided").pvalue   # rank-sum test
r["auroc_mst"], r["auroc_mst_lo"], r["auroc_mst_hi"] = auroc_ci(y, d["mst"])
r["auroc_2_20_20"], r["auroc_2_20_20_lo"], r["auroc_2_20_20_hi"] = auroc_ci(y, d["risk"])
# difference of the two AUROCs on the same resamples (paired), MST minus 2/20/20
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
m_, k_ = d["mst"].values, d["risk"].values.astype(float)
diff = [roc_auc_score(y[i], m_[i]) - roc_auc_score(y[i], k_[i]) for i in
        (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(2000))]
r["auroc_difference"] = r["auroc_mst"] - r["auroc_2_20_20"]
r["auroc_difference_lo"], r["auroc_difference_hi"] = np.percentile(diff, [2.5, 97.5])

# Logistic regression: progression ~ 2/20/20 group (+ MST per 0.1); likelihood-ratio test for adding MST
X = pd.DataFrame({"mst": d["mst"].values * 10, "IR": (d["risk"] == 1).astype(int).values, "HR": (d["risk"] == 2).astype(int).values})
full = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
clin = sm.Logit(y, sm.add_constant(X[["IR", "HR"]])).fit(disp=0)
r["or_mst_per_0.1"] = np.exp(full.params["mst"])
r["or_lo"], r["or_hi"] = np.exp(full.conf_int().loc["mst"])
r["lrt_p_adding_mst"] = stats.chi2.sf(2 * (full.llf - clin.llf), 1)

# Within each 2/20/20 group: progressors vs non-progressors (LRSMM has two progressors, not tested)
for g in ["IRSMM", "HRSMM"]:
    sub = d[d["disease_status"] == g]
    r["wilcoxon_p_within_" + g] = stats.mannwhitneyu(sub.loc[sub["progressed"] == 1, "mst"], sub.loc[sub["progressed"] == 0, "mst"],
                                                     alternative="two-sided").pvalue

# Follow-up after sampling, in years (approximate: only the year of last annotation is recorded)
fu = d["patient_id"].map(last_year) - d["date"].dt.year
r["followup_not_progressed_median"], r["followup_not_progressed_min"] = fu[y == 0].median(), fu[y == 0].min()
pd.DataFrame([r]).to_csv(os.path.join(OUT, "smm_progression_results.tsv"), sep="\t", index=False)
for k, v in r.items():
    log("%-28s %s" % (k, round(v, 3) if isinstance(v, float) else v))
d[["patient_id", "disease_status", "mst", "progressed"]].to_csv(os.path.join(OUT, "smm_progression_patient_scores.tsv"), sep="\t", index=False)
log("done")
