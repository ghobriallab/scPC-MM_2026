#!/usr/bin/env python3
"""Activity of each of the 16 signatures in SMM patients who later progressed to MM versus those who did not (Results;
Methods).

Purpose:      The SMM patients of 07_smm_progression.py: one unit per patient, the malignant BM PCs of the patient at
              one IMWG 2/20/20 risk group (risk.py; units that share a risk group are pooled, weighted by malignant BM
              PCs), the earliest risk group kept for patients sampled at two (date of the earliest BM sample of each
              unit; units without malignant BM PCs in the publication table are left out, as they have no MST
              score). Progression as in 07_smm_progression.py (any library with progression_stage "MM"). For each
              signature (mean activity of the malignant BM PCs of the unit): median in progressors and non-progressors,
              two-sided Wilcoxon rank-sum test, Benjamini-Hochberg-adjusted over the 16 signatures (reported at q < 0.1), and AUROC (higher
              activity in progressors > 0.5) with a 95% CI from 2,000 stratified bootstrap resamples (progressors and
              non-progressors resampled separately; seed 42). The patients, progressors and MST scores are checked
              against smm_progression_patient_scores.tsv.

Inputs:       data/paper/signatures/per-sample-mean-expr-score.pkl (patient_id, disease_status, sample_type, annotation,
              mean_expr:<signature>); data/paper/cohort-clinical-metadata.csv (patient_id, disease_status, sample_type,
              sample_date, n_tumor_cell, progression_stage); results/stage_and_progression/
              smm_progression_patient_scores.tsv (07_smm_progression.py).

Outputs:      results/stage_and_progression/signature_smm_progression.tsv: one row per signature: patients, progressors,
              medians, Wilcoxon P, BH q, AUROC with 95% CI.
              results/stage_and_progression/signature_smm_progression_log.txt: counts and results.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir
from risk import stages

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

SEED = 42
rng = np.random.default_rng(SEED)
OUT = str(result_dir("stage_and_progression"))
logfile = open(os.path.join(OUT, "signature_smm_progression_log.txt"), "w")


def log(*a):
    print(*a)
    print(*a, file=logfile, flush=True)


SMM = ["LRSMM", "IRSMM", "HRSMM"]
NAMES = {"cell_cycling_G1S": "Cell cycling G1/S", "cell_cycling_G2M": "Cell cycling G2/M",
         "interferon_inducible": "Interferon inducible", "histone_chaperone_gene": "Histone/chaperone",
         "B_cell_activation": "B cell activation", "FOS-JUN_stress_response": "FOS-JUN stress response",
         "HRD_signature": "HRD signature", "MAF*_signature": "MAF* signature", "MCL1-BCL2_signature": "MCL1-BCL2 signature",
         "MHC-II_immune_checkpoint": "MHC-II immune checkpoint", "MHC-I_gene_module": "MHC-I gene module",
         "circulatory_capacity": "Circulatory capacity", "extracellular_matrix": "Extracellular matrix",
         "myeloma_oncogene": "Myeloma oncogene", "secretory_granule": "Secretory granule",
         "vacuole_ER_signaling": "Vacuole/ER signaling"}
PROGRESSION_ASSOCIATED = ["Cell cycling G1/S", "Cell cycling G2/M", "Interferon inducible", "Histone/chaperone",
                          "B cell activation"]

# ---- 1. Signature activity of the malignant BM PCs of SMM patients -----------------------------------
sig = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
cols = ["mean_expr:" + k for k in NAMES]
sig = sig[sig["disease_status"].isin(SMM) & (sig["sample_type"] == "BM") & (sig["annotation"] == "tumor")]
sig = sig[["patient_id", "disease_status"] + cols]

# ---- 2. IMWG risk groups, pooling, earliest unit and progression, as in 07_smm_progression.py ------------
pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
pub["sample_date"] = pd.to_datetime(pub["sample_date"], errors="coerce")
bm = pub[pub["sample_type"] == "BM"]
w = bm.groupby(["patient_id", "disease_status"])["n_tumor_cell"].sum()
sig["w"] = [w.get(k, 0) for k in zip(sig["patient_id"], sig["disease_status"])]
sig = sig[sig["w"] > 0]       # units without malignant BM PCs in the publication table (not in 07_smm_progression.py)
sig["disease_status"] = stages(sig["patient_id"], ["BM"] * len(sig), sig["disease_status"])
sig = sig.groupby(["patient_id", "disease_status"]).apply(
    lambda g: pd.Series({c: np.average(g[c], weights=g["w"]) for c in cols})).reset_index()
bm = bm.assign(disease_status=stages(bm["patient_id"], bm["sample_type"], bm["disease_status"]))
first_date = bm.groupby(["patient_id", "disease_status"])["sample_date"].min().rename("date")
sig = sig.join(first_date, on=["patient_id", "disease_status"])
d = sig.sort_values(["date", "patient_id"]).drop_duplicates("patient_id").copy()
progressed = set(pub.loc[pub["progression_stage"] == "MM", "patient_id"])
d["progressed"] = d["patient_id"].isin(progressed).astype(int)

# same patients, risk groups and progressors as 07_smm_progression.py
ref = pd.read_csv(result("stage_and_progression", "smm_progression_patient_scores.tsv"), sep="\t")
chk = ref.merge(d[["patient_id", "disease_status", "progressed"]], on="patient_id", how="outer", suffixes=("_ref", ""))
assert len(chk) == len(ref) == len(d), (len(ref), len(d))
assert (chk["disease_status_ref"] == chk["disease_status"]).all() and (chk["progressed_ref"] == chk["progressed"]).all()
y = d["progressed"].values
log("SMM patients: %d, progressed: %d (same patients, risk groups and progressors as smm_progression_patient_scores.tsv)"
    % (len(d), y.sum()))


# ---- 3. Progressors versus non-progressors, per signature ------------------------------------------
def auroc_ci(y, score, n=2000):
    """AUROC with a stratified bootstrap 95% CI (progressors and non-progressors resampled separately)."""
    score = np.asarray(score, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = [roc_auc_score(y[i], score[i]) for i in
            (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n))]
    return roc_auc_score(y, score), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


rows = []
for c in cols:
    v = d[c].values
    a, lo, hi = auroc_ci(y, v)
    rows.append({"signature": NAMES[c.split(":", 1)[1]], "patients": len(d), "progressed": int(y.sum()),
                 "median_progressed": np.median(v[y == 1]), "median_not_progressed": np.median(v[y == 0]),
                 "wilcoxon_p": stats.mannwhitneyu(v[y == 1], v[y == 0], alternative="two-sided").pvalue,
                 "auroc": a, "auroc_lo": lo, "auroc_hi": hi})
res = pd.DataFrame(rows)
res["q"] = multipletests(res["wilcoxon_p"], method="fdr_bh")[1]
res["progression_associated"] = res["signature"].isin(PROGRESSION_ASSOCIATED)
res = res.sort_values("wilcoxon_p")
res.to_csv(os.path.join(OUT, "signature_smm_progression.tsv"), sep="\t", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    log(res.round(4).to_string(index=False))
log("signatures with P < 0.05: %d; with q < 0.1: %s" % ((res["wilcoxon_p"] < 0.05).sum(),
    ", ".join(res.loc[res["q"] < 0.1, "signature"]) or "none"))
log("done")
