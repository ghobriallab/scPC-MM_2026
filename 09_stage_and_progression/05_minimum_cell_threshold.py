#!/usr/bin/env python3
"""Stage correlations of the MST score and of the five progression-associated signatures after removing units with few
malignant PCs (Methods, Cohort structure and repeat samples).

Purpose:      The stage comparisons use the mean of all PCs of each unit (individual x disease stage x tissue x PC
              status). Here, for the malignant BM PCs of precursor and MM stages, the number of cells per unit is
              counted from the cell table, and Spearman's correlation of each score with disease stage (MGUS 0, LRSMM 1,
              IRSMM 2, HRSMM 3, MM 4) is computed with no minimum and after removing units with fewer than 250 or 500
              cells. The numbers of individuals with fewer than 250 and 500 PCs are also reported.

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (one row per PC: redcap_id, disease_status,
              sample_type, annotation; used only to count cells per unit);
              data/metadata/tumor-cohort-metadata-final-240325.csv (redcap_id, patient_id);
              data/paper/classifier/patient_id_mean_probability.pkl (unit MST scores: "MST Score", sample,
              disease_status, sample_type, annotation);
              data/paper/signatures/per-sample-mean-expr-score.pkl (unit signature activities: mean_expr:<signature>,
              patient_id, disease_status, sample_type, annotation).

Outputs:      results/stage_and_progression/minimum_cell_threshold.tsv: per score and minimum (none, 250, 500): units, Spearman's rho with stage, P.
              results/stage_and_progression/minimum_cell_threshold_log.txt: individuals with fewer than 250 and 500 PCs; units per score.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import numpy as np
import pandas as pd
from scipy import stats

OUT = result_dir("stage_and_progression")
STAGE = {"MGUS": 0, "LRSMM": 1, "IRSMM": 2, "HRSMM": 3, "MM": 4}
# the five signatures the paper calls progression-associated (Results, CoMMpass survival paragraph)
SIGS = {"cell_cycling_G1S": "Cell cycling G1/S", "cell_cycling_G2M": "Cell cycling G2/M",
        "interferon_inducible": "Interferon inducible", "histone_chaperone_gene": "Histone/chaperone",
        "B_cell_activation": "B cell activation"}

# cells per unit, with the labels the cell table carries (the labels of the unit tables)
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
to_pt = meta[["redcap_id", "patient_id"]].dropna().astype(str).drop_duplicates().set_index("redcap_id")["patient_id"]
cells = cells[["redcap_id", "disease_status", "sample_type", "annotation"]].astype(str)
cells["patient_id"] = cells["redcap_id"].map(to_pt)
assert cells["patient_id"].notna().all()
n_unit = cells.groupby(["patient_id", "disease_status", "sample_type", "annotation"]).size().rename("cells")
per_ind = cells.groupby("patient_id").size()
log = ["individuals: %d; with fewer than 250 PCs: %d; fewer than 500 PCs: %d; minimum %d PCs"
       % (len(per_ind), (per_ind < 250).sum(), (per_ind < 500).sum(), per_ind.min())]

# unit scores: malignant PCs of BM samples, precursor and MM stages
mst = pd.read_pickle(data("paper", "classifier", "patient_id_mean_probability.pkl"))
mst = mst.rename(columns={"sample": "patient_id"})
sig = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
scores = {"MST score": (mst, "MST Score")}
scores.update({SIGS[s]: (sig, "mean_expr:" + s) for s in SIGS})

rows = []
for name, (df, col) in scores.items():
    u = df[(df["annotation"] == "tumor") & (df["sample_type"] == "BM") & df["disease_status"].isin(list(STAGE))].copy()
    u = u[u[col].notna()]                                    # units without a score in the unit table
    u = u.join(n_unit, on=["patient_id", "disease_status", "sample_type", "annotation"])
    # a unit whose label had no PCs in the cell table would count as 0 cells (kept without a minimum, removed with one); none
    # occurs once units without a score are dropped
    n0, missing = len(u), u["cells"].isna().sum()
    u["cells"] = u["cells"].fillna(0)
    # IMWG risk groups; units of one patient that share a group are pooled (cell-count weighted means)
    u["disease_status"] = stages(u["patient_id"], u["sample_type"], u["disease_status"])
    u = u.groupby(["patient_id", "disease_status"]).apply(lambda g: pd.Series(
        {col: np.average(g[col], weights=g["cells"]) if g["cells"].sum() > 0 else g[col].mean(),
         "cells": g["cells"].sum()})).reset_index()
    log.append("%s: %d units (%d before pooling units that share an IMWG group), %d without cells under their label"
               % (name, len(u), n0, missing))
    for m in (0, 250, 500):
        v = u[u["cells"] >= m]
        rho, p = stats.spearmanr(v["disease_status"].map(STAGE), v[col])
        rows.append(dict(score=name, min_cells=m, units=len(v), spearman_rho_with_stage=rho, P=p))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "minimum_cell_threshold.tsv"), sep="\t", index=False)
log.append(R.round(4).to_string(index=False))
open(os.path.join(OUT, "minimum_cell_threshold_log.txt"), "w").write("\n".join(log) + "\n")
print("\n".join(log))
