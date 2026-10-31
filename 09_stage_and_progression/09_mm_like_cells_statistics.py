#!/usr/bin/env python3
"""MM-like cells in SMM tumors by IMWG 2/20/20 risk group and progression to MM, and the same statistics for the mean
MST score (Fig. 5A; Results).

Purpose:      The SMM patients of 07_smm_progression.py (earliest BM tumor of each patient) with the percentage of
              MM-like malignant PCs (MST score > 0.5) of the same tumor. Per risk group and for all SMM patients:
              patients, median and IQR of the percentage, tumors with 10-90% MM-like cells, progressors versus others
              (two-sided Wilcoxon rank-sum test). All SMM patients: Spearman's correlation with the risk group (coded
              0-2); logistic regression of progression on the risk group (IR and HR indicators) with and without the
              percentage (per 10 points), likelihood-ratio test; median and range of malignant BM PCs per tumor; the
              same Spearman, rank-sum and likelihood-ratio statistics for the mean MST score (per 0.1), checked against
              smm_progression_results.tsv. Discrimination of progressors by the percentage, as for the mean MST score in
              07_smm_progression.py: AUROC with a 95% CI from 2,000 stratified bootstrap resamples (progressors and
              non-progressors resampled separately; seed 42) and its difference from the AUROC of the 2/20/20 risk
              group with a 95% CI from 2,000 paired resamples of the same kind; per risk group, the lowest percentage
              among progressors.

Inputs:       results/stage_and_progression/smm_progression_patient_scores.tsv (patient_id, disease_status, mst,
              progressed); results/stage_and_progression/mm_like_cells_tumors.tsv (patient_id, disease_status, cells,
              frac_mm_like); results/stage_and_progression/smm_progression_results.tsv (wilcoxon_p, lrt_p_adding_mst;
              check of the MST statistics).

Outputs:      results/stage_and_progression/mm_like_cells_smm.tsv: per risk group and all SMM: patients, median and IQR of the percentage of MM-like cells, tumors with 10-90% MM-like cells, progressors vs others; all SMM: Spearman's rho and P with the risk group, likelihood-ratio P, malignant BM PCs per tumor, AUROC of the percentage with its 95% CI and its difference from the 2/20/20 risk group, and the same statistics for the mean MST score (mst_* columns).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn, statsmodels); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import result, result_dir

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

OUT = result_dir("stage_and_progression")
SMM = ["LRSMM", "IRSMM", "HRSMM"]

# SMM patients of 07_smm_progression.py with the fraction of MM-like cells of the same tumor
d = pd.read_csv(result("stage_and_progression", "smm_progression_patient_scores.tsv"), sep="\t")
t = pd.read_csv(result("stage_and_progression", "mm_like_cells_tumors.tsv"), sep="\t")
d = d.merge(t[["patient_id", "disease_status", "cells", "frac_mm_like"]], on=["patient_id", "disease_status"], how="left")
d["pct"] = 100 * d["frac_mm_like"]
assert d["pct"].notna().all() and len(d) == 115

rows = []
for g in SMM:
    x = d[d["disease_status"] == g]
    p = (stats.mannwhitneyu(x.loc[x["progressed"] == 1, "pct"], x.loc[x["progressed"] == 0, "pct"], alternative="two-sided").pvalue
         if x["progressed"].sum() >= 2 else np.nan)
    rows.append(dict(group=g, patients=len(x), progressed=int(x["progressed"].sum()), median_pct=x["pct"].median(),
                     q25=x["pct"].quantile(0.25), q75=x["pct"].quantile(0.75),
                     mixed_10_90=int(((x["pct"] >= 10) & (x["pct"] <= 90)).sum()), P_progressed_vs_not=p,
                     min_pct_progressed=x.loc[x["progressed"] == 1, "pct"].min()))
# all SMM patients: trend with the risk group (Spearman, risk coded 0-2); progressors vs others; added value over the
# risk group (logistic regression on IR and HR indicators, + percentage per 10 points; likelihood-ratio test, as in
# 07_smm_progression.py)
x = d
y = x["progressed"].values
rho, p_rho = stats.spearmanr(x["pct"], x["disease_status"].map({g: i for i, g in enumerate(SMM)}))
X = pd.DataFrame({"IR": (x["disease_status"] == "IRSMM").astype(int), "HR": (x["disease_status"] == "HRSMM").astype(int)})
clin = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
full = sm.Logit(y, sm.add_constant(X.assign(pct10=x["pct"] / 10))).fit(disp=0)
# the same three statistics for the mean MST score of 07_smm_progression.py (Fig. 5A, top)
rho_mst, p_rho_mst = stats.spearmanr(x["mst"], x["disease_status"].map({g: i for i, g in enumerate(SMM)}))
full_mst = sm.Logit(y, sm.add_constant(X.assign(mst10=x["mst"] * 10))).fit(disp=0)
# AUROC of the percentage of MM-like cells and its paired difference from the 2/20/20 risk group (coded 0-2), with the
# stratified bootstrap of 07_smm_progression.py (progressors and non-progressors resampled separately)
rng = np.random.default_rng(42)
pct, risk = x["pct"].values, x["disease_status"].map({g: i for i, g in enumerate(SMM)}).values.astype(float)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]


def resamples(n=2000):
    return (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n))


auc_pct, auc_risk = roc_auc_score(y, pct), roc_auc_score(y, risk)
boot = [roc_auc_score(y[i], pct[i]) for i in resamples()]
diff = [roc_auc_score(y[i], pct[i]) - roc_auc_score(y[i], risk[i]) for i in resamples()]
rows.append(dict(group="all SMM", patients=len(x), progressed=int(y.sum()), median_pct=x["pct"].median(),
                 q25=x["pct"].quantile(0.25), q75=x["pct"].quantile(0.75),
                 mixed_10_90=int(((x["pct"] >= 10) & (x["pct"] <= 90)).sum()),
                 median_pct_progressed=x.loc[y == 1, "pct"].median(), median_pct_not_progressed=x.loc[y == 0, "pct"].median(),
                 P_progressed_vs_not=stats.mannwhitneyu(x.loc[y == 1, "pct"], x.loc[y == 0, "pct"], alternative="two-sided").pvalue,
                 spearman_rho_risk_group=rho, spearman_P_risk_group=p_rho,
                 LRT_P_added_to_risk_group=stats.chi2.sf(2 * (full.llf - clin.llf), 1),
                 median_malignant_PCs=x["cells"].median(), min_malignant_PCs=x["cells"].min(), max_malignant_PCs=x["cells"].max(),
                 mst_spearman_rho_risk_group=rho_mst, mst_spearman_P_risk_group=p_rho_mst,
                 mst_P_progressed_vs_not=stats.mannwhitneyu(x.loc[y == 1, "mst"], x.loc[y == 0, "mst"],
                                                            alternative="two-sided").pvalue,
                 mst_LRT_P_added_to_risk_group=stats.chi2.sf(2 * (full_mst.llf - clin.llf), 1),
                 auroc_pct=auc_pct, auroc_pct_lo=np.percentile(boot, 2.5), auroc_pct_hi=np.percentile(boot, 97.5),
                 auroc_2_20_20=auc_risk, auroc_pct_minus_2_20_20=auc_pct - auc_risk,
                 auroc_difference_lo=np.percentile(diff, 2.5), auroc_difference_hi=np.percentile(diff, 97.5)))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "mm_like_cells_smm.tsv"), sep="\t", index=False)
print(S.round(4).to_string(index=False))

# the MST statistics must equal those of 07_smm_progression.py
a = S[S["group"] == "all SMM"].iloc[0]
r1 = pd.read_csv(result("stage_and_progression", "smm_progression_results.tsv"), sep="\t").iloc[0]
assert abs(a["mst_P_progressed_vs_not"] - r1["wilcoxon_p"]) < 1e-9
assert abs(a["mst_LRT_P_added_to_risk_group"] - r1["lrt_p_adding_mst"]) < 1e-6
assert abs(a["auroc_2_20_20"] - r1["auroc_2_20_20"]) < 1e-9
