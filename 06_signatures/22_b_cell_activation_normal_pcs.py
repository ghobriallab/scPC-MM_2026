#!/usr/bin/env python3
"""B cell activation signature in malignant and normal BM PCs: same-patient comparison, change across disease stages in
malignant PCs, and healthy donors versus patients in normal PCs (Results; Discussion).

Purpose:      Activity = mean log-normalized expression of the signature genes, averaged over the PCs of a unit (patient x
              disease stage x cell type, BM; units with >= 30 PCs; SMM risk groups of the IMWG 2/20/20 model, risk.py,
              units that share a group pooled). 1. Malignant versus normal PCs of the same patient
              and stage (>= 30 PCs of each type): number of patients with higher activity in normal PCs, two-sided
              Wilcoxon signed-rank test. 2. Malignant PCs across disease stages (the BM tumors with >= 30 malignant
              PCs): median activity per stage, two-sided Wilcoxon rank-sum tests between selected stages, Spearman
              correlation with stage (MGUS 0 ... MM 4). 3. Normal PCs of healthy donors versus patients (two-sided
              Wilcoxon rank-sum test).

Inputs:       data/paper/signatures/per-sample-mean-expr-score.pkl: mean_expr:B_cell_activation, patient_id, sample_type,
              annotation, disease_status.
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, disease_status, sample_type, annotation
              (cells per unit).
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id.

Outputs:      results/signatures/b_cell_activation_by_stage.tsv: per stage: tumors, median activity, paired units, P values.
              results/signatures/b_cell_activation_log.txt: paired, stage and donor comparisons.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import pool_units, stages

import pandas as pd
from scipy import stats

OUT = result_dir("signatures")
FH = open(os.path.join(OUT, "b_cell_activation_log.txt"), "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
COL = "mean_expr:B_cell_activation"
MIN_CELLS = 30


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


m = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
m = m[m["sample_type"] == "BM"]
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells.loc[cells["sample_type"].astype(str) == "BM", ["redcap_id", "disease_status", "annotation"]].astype(str)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), dtype=str)
cells["patient_id"] = cells["redcap_id"].map(meta[["redcap_id", "patient_id"]].dropna().drop_duplicates()
                                              .set_index("redcap_id")["patient_id"])
n = cells.groupby(["patient_id", "disease_status", "annotation"]).size().rename("n_cells").reset_index()
m = m.merge(n, on=["patient_id", "disease_status", "annotation"], how="left")
m["disease_status"] = stages(m["patient_id"], m["sample_type"], m["disease_status"])      # IMWG risk groups, then pooling
m = pool_units(m[m["n_cells"].notna()], ["patient_id", "disease_status", "annotation"], "n_cells",
               [c for c in m.columns if c.startswith("mean_expr:")])
m = m[m["n_cells"] >= MIN_CELLS]

# 1. malignant vs normal PCs of the same patient and stage
pv = m[m["disease_status"].isin(STAGES)].pivot_table(index=["patient_id", "disease_status"], columns="annotation",
                                                      values=COL).dropna()
log("paired units (>= %d malignant and normal BM PCs): %d; normal higher in %d (%.0f%%); signed-rank P = %.2g" % (
    MIN_CELLS, len(pv), (pv["normal"] > pv["tumor"]).sum(), 100 * (pv["normal"] > pv["tumor"]).mean(),
    stats.wilcoxon(pv["tumor"], pv["normal"]).pvalue))

# 2. malignant PCs across stages
t = m[(m["annotation"] == "tumor") & m["disease_status"].isin(STAGES)]
rows = []
for s in STAGES:
    x = t.loc[t["disease_status"] == s, COL]
    y = pv.xs(s, level="disease_status") if s in pv.index.get_level_values(1) else None
    rows.append({"stage": s, "tumors": len(x), "median_malignant": x.median(),
                 "paired_units": len(y), "normal_higher": int((y["normal"] > y["tumor"]).sum())})
tab = pd.DataFrame(rows)
log("malignant BM tumors: %d\n%s" % (len(t), tab.round(4).to_string(index=False)))
for a, b in [("MGUS", "MM"), ("LRSMM", "HRSMM"), ("HRSMM", "MM"), ("MGUS", "LRSMM")]:
    tab.loc[tab["stage"] == b, "P_vs_" + a] = stats.mannwhitneyu(t.loc[t["disease_status"] == a, COL],
                                                                  t.loc[t["disease_status"] == b, COL]).pvalue
    log("  %s vs %s: rank-sum P = %.2g" % (a, b, tab.loc[tab["stage"] == b, "P_vs_" + a].iloc[0]))
r, pr = stats.spearmanr(t["disease_status"].map(STAGES.index), t[COL])
log("Spearman rho with stage (MGUS 0 ... MM 4): %.2f, P = %.2g" % (r, pr))
tab.to_csv(os.path.join(OUT, "b_cell_activation_by_stage.tsv"), sep="\t", index=False)

# 3. normal PCs: donors vs patients
nd = m[(m["annotation"] == "normal")]
d, pt = nd.loc[nd["disease_status"] == "Donor", COL], nd.loc[nd["disease_status"].isin(STAGES), COL]
log("normal BM PCs: donors median %.4f (n = %d), patients %.4f (n = %d); rank-sum P = %.2g" % (
    d.median(), len(d), pt.median(), len(pt), stats.mannwhitneyu(d, pt).pvalue))
FH.close()
