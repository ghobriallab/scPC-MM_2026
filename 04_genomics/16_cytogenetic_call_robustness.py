#!/usr/bin/env python3
"""Scale and support of the scRNA-seq cytogenetic calls: population sizes, partner-gene detection and HRD calls against FISH and
WGS (Methods, Classification of cytogenetic alterations).

Purpose:      Calls are made per malignant population (one per patient, two in the four patients with two malignant
              populations), never per cell.
              1. Size of the malignant populations that received a call.
              2. Detection of the translocation partner genes in single cells: percentage of malignant PCs with at least
                 one UMI of the partner gene, per tumor, against healthy-donor PCs.
              3. HRD calls (Numbat, tumors without a translocation): agreement with FISH and WGS calls of HRD (Numbat uses
                 donor PCs as its normal reference, so it cannot be tested on donor cells).

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, gex_id, sample_type, annotation, tumor_id,
              CCND1, CCND2, NSD2, FGFR3, MAF, MAFB.
              data/paper/cytogenetics-status.csv: tumor_id (index, <Pt-###>:tumor:<k>), patient_id, cytogenetics, FISH,
              WGS, scRNA.
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id, disease_status.

Outputs:      results/genomics/cytogenetic_population_sizes.tsv: malignant PCs per called population, by reference.
              results/genomics/partner_gene_detection.tsv: partner-gene detection in malignant PCs and healthy-donor PCs.
              results/genomics/hrd_concordance.tsv: scRNA-seq HRD calls against FISH and WGS.
              results/genomics/cytogenetic_call_robustness_log.txt: tables printed by the script (no internal IDs).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import numpy as np
import pandas as pd

OUT = result_dir("genomics")
FH = open(OUT / "cytogenetic_call_robustness_log.txt", "w")
GENES = ["CCND1", "CCND2", "NSD2", "FGFR3", "MAF", "MAFB"]
CLASSES = ["t(11;14)", "t(4;14)", "t(14;16)", "t(14;20)", "t(6;14)", "t(8;14)"]
PARTNER = {"t(11;14)": "CCND1", "t(4;14)": "NSD2", "t(14;16)": "MAF", "t(14;20)": "MAFB", "t(12;14)": "CCND2"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


# ---- cells (as in 15_two_clone_tumors.py) ---------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
red2pt = meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"]
donors = set(meta.loc[meta["disease_status"] == "Donor", "patient_id"])
cols = ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id"] + GENES
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[cols]
for c in ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id"]:
    ann[c] = ann[c].astype(str)
ann["patient_id"] = ann["redcap_id"].map(red2pt)
assert ann["patient_id"].notna().all()
ann["tumor_id"] = np.where(ann["annotation"] == "tumor", ann["patient_id"] + ":tumor:" + ann["tumor_id"].str.rsplit(":", n=1).str[-1], "nan")
don = ann[ann["patient_id"].isin(donors)]
log("healthy donors: %d individuals, %d PCs (annotation of donor PCs: %s)"
    % (don["patient_id"].nunique(), len(don), don["annotation"].value_counts().to_dict()))
assert don["patient_id"].nunique() == 29 and (don["annotation"] == "normal").all()

cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"), index_col=0)
tum = ann[ann["annotation"] == "tumor"]
assert tum["tumor_id"].isin(cyto.index).all()
pops = {tid: q for tid, q in tum.groupby("tumor_id")}

# ---- 1. size of the populations that received a call ----------------------------------------------
valid = CLASSES + ["HRD"]
cyto["n_cells"] = [len(pops[t]) for t in cyto.index]
cyto["reference"] = np.where(cyto["FISH"].isin(valid) | cyto["WGS"].isin(valid), "FISH or WGS", "scRNA-seq only")
S = cyto.groupby("reference")["n_cells"].describe(percentiles=[0.1, 0.5]).reset_index()
S = pd.concat([S, cyto["n_cells"].describe(percentiles=[0.1, 0.5]).to_frame().T.assign(reference="all")])
S.to_csv(OUT / "cytogenetic_population_sizes.tsv", sep="\t", index=False)
log("\n1. malignant PCs per called population (%d populations, %d patients)" % (len(cyto), cyto["patient_id"].nunique()))
log(S.round(1).to_string(index=False))
log("populations with < 100 / < 50 malignant PCs: %d / %d" % ((cyto["n_cells"] < 100).sum(), (cyto["n_cells"] < 50).sum()))
log("calls of populations with < 100 malignant PCs: %s (reference: %s)" % (
    cyto.loc[cyto["n_cells"] < 100, "scRNA"].value_counts().to_dict(), cyto.loc[cyto["n_cells"] < 100, "reference"].value_counts().to_dict()))

# ---- 2. partner-gene detection in single cells -------------------------------------------------------
rows = []
for t, g in PARTNER.items():
    with_t = [tid for tid in cyto.index if cyto.loc[tid, "cytogenetics"] == t]
    other = [tid for tid in cyto.index if cyto.loc[tid, "cytogenetics"] != t]
    det = pd.Series({tid: 100 * (pops[tid][g] > 0).mean() for tid in cyto.index})
    rows.append(dict(translocation=t, gene=g, n_tumors=len(with_t),
                     pct_cells_detected_median=det[with_t].median(), pct_cells_detected_min=det[with_t].min(),
                     other_tumors_pct_median=det[other].median(),
                     donor_pct_cells=100 * (don[g] > 0).mean(),
                     donor_pct_cells_max_individual=don.groupby("patient_id")[g].apply(lambda v: 100 * (v > 0).mean()).max()))
    low = [tid for tid in with_t if det[tid] < 25]
    if low:
        log("  %s tumors with %s detected in < 25%% of malignant PCs: %d (FISH %s; WGS %s)" % (
            t, g, len(low), cyto.loc[low, "FISH"].tolist(), cyto.loc[low, "WGS"].tolist()))
D = pd.DataFrame(rows)
D.to_csv(OUT / "partner_gene_detection.tsv", sep="\t", index=False)
log("\n2. partner gene detected (>= 1 UMI) in malignant PCs of tumors with the translocation vs healthy-donor PCs")
log(D.round(2).to_string(index=False))

# ---- 3. HRD calls against FISH and WGS ------------------------------------------------------------------
m0 = cyto[cyto.index.str.endswith("tumor:0")].copy()
m0["scRNA"] = m0["scRNA"].replace({"no_translocation_detected": "none"})
rows = []
for assay in ["FISH", "WGS"]:
    h = m0[m0[assay] == "HRD"]
    rows.append(dict(assay=assay, reference="HRD", n=len(h), scRNA_HRD=int((h["scRNA"] == "HRD").sum()),
                     scRNA_calls=h["scRNA"].value_counts().to_dict()))
    o = m0[m0[assay].isin(["Other"])]
    if len(o):
        rows.append(dict(assay=assay, reference="Other (no translocation or HRD)", n=len(o),
                         scRNA_HRD=int((o["scRNA"] == "HRD").sum()), scRNA_calls=o["scRNA"].value_counts().to_dict()))
H = pd.DataFrame(rows)
H.to_csv(OUT / "hrd_concordance.tsv", sep="\t", index=False)
log("\n3. scRNA-seq HRD calls (Numbat) in the main population of patients with an HRD or 'Other' call by FISH or WGS")
log(H.to_string(index=False))
log("\ndone")
