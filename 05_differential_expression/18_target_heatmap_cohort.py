#!/usr/bin/env python3
"""Tumors included in and left out of the within-patient comparisons of malignant and normal PCs (Fig. 2A), and whether
the included tumors represent all tumors (Methods, Cohort structure and repeat samples).

Purpose:      Within-patient comparisons of malignant and normal PCs require at least 30 PCs of each type. Tumors =
              patient x disease stage units with malignant BM PCs. Included = the tumors whose main clone has a
              within-patient differential expression result in Fig. 2A (columns <patient>_BM_<stage>_tumor_0 of the DE
              table). Included and excluded tumors are compared by disease stage and cytogenetic group (chi-square
              statistic of the stage or group x inclusion table, P from 20,000 permutations of inclusion,
              default_rng(42)), age (two-sided Wilcoxon rank-sum test) and sex (Fisher's exact test), and by the numbers of
              malignant and normal PCs (reason for exclusion).

Inputs:       results/differential_expression/BM_scRNA_tumor_DE_FDR.csv: columns = within-patient DE results (Fig. 2A).
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, disease_status, sample_type, annotation (cell
              counts).
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id.
              data/paper/cytogenetics-status.csv: tumor_id, patient_id, cytogenetics (main tumor).
              data/paper/cohort-clinical-metadata.csv: patient_id, Age, Sex.

Outputs:      results/differential_expression/target_heatmap_tumors.tsv: one row per tumor (numbered, no patient label):
              stage, cell numbers, inclusion, reason, cytogenetic group, age, sex.
              results/differential_expression/target_heatmap_inclusion.tsv: tumors and share included, by stage and by
              cytogenetic group.
              results/differential_expression/target_heatmap_cohort_log.txt: counts and tests printed by the script.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir
from risk import stages

import numpy as np
import pandas as pd
from scipy import stats

OUT = result_dir("differential_expression")
FH = open(OUT / "target_heatmap_cohort_log.txt", "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
GROUP = {"t(11;14)": "CCND1", "t(4;14)": "NSD2", "t(14;16)": "MAF", "t(14;20)": "MAFB", "t(12;14)": "CCND2/CCND3/MAFA",
         "t(6;14)": "CCND2/CCND3/MAFA", "t(8;14)": "CCND2/CCND3/MAFA", "HRD": "HRD", "Other": "None"}
RNG = np.random.default_rng(42)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


def perm_test(labels, x, n=20000):
    """P value of the chi-square statistic of the labels x inclusion table, from permutations of inclusion"""
    obs = stats.chi2_contingency(pd.crosstab(labels, x))[0]
    return (1 + sum(stats.chi2_contingency(pd.crosstab(labels, RNG.permutation(x)))[0] >= obs for _ in range(n))) / (n + 1)


# ---- tumors and their cell counts ---------------------------------------------------------------------------------------
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells.loc[cells["sample_type"].astype(str) == "BM", ["redcap_id", "disease_status", "annotation"]].astype(str)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), dtype=str)
cells["patient_id"] = cells["redcap_id"].map(meta[["redcap_id", "patient_id"]].dropna().drop_duplicates()
                                              .set_index("redcap_id")["patient_id"])
cells["disease_status"] = stages(cells["patient_id"], ["BM"] * len(cells), cells["disease_status"])   # IMWG risk groups
n = cells.groupby(["patient_id", "disease_status", "annotation"]).size().unstack(fill_value=0)
t = n[n.get("tumor", 0) > 0].reset_index().rename(columns={"tumor": "malignant_PCs", "normal": "normal_PCs"})
t = t[t["disease_status"].isin(STAGES)]
fdr = pd.read_csv(result("differential_expression", "BM_scRNA_tumor_DE_FDR.csv"), index_col=0, nrows=1)
fig2a = {(c.split("_")[0], c.split("_")[2]) for c in fdr.columns if c.endswith("_0")}
t["in_fig2a"] = [(p, s) in fig2a for p, s in zip(t["patient_id"], t["disease_status"])]
assert t["in_fig2a"].sum() == len(fig2a), "every Fig. 2A tumor must be among the tumors"
cy = pd.read_csv(data("paper", "cytogenetics-status.csv"))
cy = cy[cy["tumor_id"].str.endswith(":tumor:0")].set_index("patient_id")
t["group"] = t["patient_id"].map(cy["cytogenetics"]).map(GROUP).fillna("No call")
clin = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
t["age"] = t["patient_id"].map(clin.groupby("patient_id")["Age"].min())
t["sex"] = t["patient_id"].map(clin.groupby("patient_id")["Sex"].first())
t["malignant_fraction"] = t["malignant_PCs"] / (t["malignant_PCs"] + t["normal_PCs"])
t["reason"] = np.select([t["in_fig2a"], t["malignant_PCs"] < 30, t["normal_PCs"] < 30],
                        ["included", "< 30 malignant PCs", "< 30 normal PCs"], "other")
t.drop(columns=["patient_id"]).assign(tumor=range(1, len(t) + 1)).to_csv(OUT / "target_heatmap_tumors.tsv", sep="\t", index=False)
log("tumors with malignant BM PCs: %d (%d patients); in Fig. 2A: %d (%d patients)" % (
    len(t), t["patient_id"].nunique(), t["in_fig2a"].sum(), t.loc[t["in_fig2a"], "patient_id"].nunique()))
log("reasons:", t["reason"].value_counts().to_dict())
log("excluded with < 30 normal PCs: median malignant fraction %.3f (included %.3f)" % (
    t.loc[t["reason"] == "< 30 normal PCs", "malignant_fraction"].median(), t.loc[t["in_fig2a"], "malignant_fraction"].median()))
log("tumors with >= 30 malignant PCs: %d; of these in Fig. 2A: %d" % ((t["malignant_PCs"] >= 30).sum(),
                                                                     t.loc[t["malignant_PCs"] >= 30, "in_fig2a"].sum()))

# ---- included vs excluded -----------------------------------------------------------------------------------------------
rows = []
for var, cats in [("disease_status", STAGES), ("group", ["CCND1", "NSD2", "MAF", "MAFB", "CCND2/CCND3/MAFA", "HRD", "None",
                                                         "No call"])]:
    for c in cats:
        x = t[t[var] == c]
        rows.append({"variable": var, "category": c, "tumors": len(x), "in_fig2a": int(x["in_fig2a"].sum()),
                     "pct_in_fig2a": 100 * x["in_fig2a"].mean() if len(x) else np.nan,
                     "share_all_pct": 100 * len(x) / len(t), "share_fig2a_pct": 100 * x["in_fig2a"].sum() / t["in_fig2a"].sum()})
comp = pd.DataFrame(rows)
p_stage = perm_test(t["disease_status"], t["in_fig2a"].values)
p_group = perm_test(t["group"], t["in_fig2a"].values)
p_age = stats.mannwhitneyu(t.loc[t["in_fig2a"], "age"], t.loc[~t["in_fig2a"], "age"]).pvalue
p_sex = stats.fisher_exact(pd.crosstab(t["in_fig2a"], t["sex"]))[1]
comp.to_csv(OUT / "target_heatmap_inclusion.tsv", sep="\t", index=False)
log(comp.round(1).to_string(index=False))
log("included vs excluded: stage P = %.3g; cytogenetic group P = %.3g (permutation chi-square); age P = %.3g (rank-sum); "
    "sex P = %.3g (Fisher)" % (p_stage, p_group, p_age, p_sex))
log("largest difference in share between all tumors and Fig. 2A tumors: stage %.1f, group %.1f percentage points" % (
    (comp.loc[comp.variable == "disease_status", "share_fig2a_pct"] - comp.loc[comp.variable == "disease_status", "share_all_pct"]).abs().max(),
    (comp.loc[comp.variable == "group", "share_fig2a_pct"] - comp.loc[comp.variable == "group", "share_all_pct"]).abs().max()))
FH.close()
