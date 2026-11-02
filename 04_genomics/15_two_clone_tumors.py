#!/usr/bin/env python3
"""Patients with two malignant PC populations: BCR, constant genes and partner genes (Results; Methods, Classification of
cytogenetic alterations).

Purpose:      The four patients with two cytogenetically distinct malignant populations (tumor populations :0 and :1 of the
              cytogenetic table). Malignant PCs are defined by expanded BCR clonotypes (Methods), so each population is
              compared on its BCR, measured independently of the expression-based cytogenetic call: modal heavy-chain CDR3
              (amino acid) and the share of cells carrying it or the other population's; whether the modal CDR3 occurs in any
              other individual of the cohort (cross-sample contamination); heavy- and light-chain constant genes; libraries
              and tissues; median UMIs; the clinical immunofixation entry; and the expression of the translocation partner
              genes in each population, the patient's normal PCs and all healthy-donor PCs.

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, gex_id, sample_type, annotation, tumor_id,
              cytogenetics, heavy_cdr3_aa, heavy_c_gene, light_c_gene, total_counts, CCND1, CCND2, NSD2, FGFR3, MAF, MAFB.
              data/paper/cytogenetics-status.csv: tumor_id (index, <Pt-###>:tumor:<k>), patient_id, cytogenetics, FISH,
              WGS, regression_pred.
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id, disease_status (Donor), IFX
              (clinical serum immunofixation).

Outputs:      results/genomics/two_clone_tumors.tsv: one row per population of the four patients (no sequences or
              internal IDs).
              results/genomics/two_clone_partner_genes.tsv: partner-gene expression by population, normal PCs and donor
              PCs.
              results/genomics/two_clone_tumors_log.txt: tables printed by the script.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import numpy as np
import pandas as pd

OUT = result_dir("genomics")
FH = open(OUT / "two_clone_tumors_log.txt", "w")
GENES = ["CCND1", "CCND2", "NSD2", "FGFR3", "MAF", "MAFB"]
PARTNER = {"t(11;14)": "CCND1", "t(14;16)": "MAF", "t(14;20)": "MAFB", "t(4;14)": "NSD2"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


# ---- cells --------------------------------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
red2pt = meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"]
donors = set(meta.loc[meta["disease_status"] == "Donor", "patient_id"])
cols = ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id", "cytogenetics", "heavy_cdr3_aa",
        "heavy_c_gene", "light_c_gene", "total_counts"] + GENES
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[cols]
for c in ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id", "cytogenetics", "heavy_cdr3_aa",
          "heavy_c_gene", "light_c_gene"]:
    ann[c] = ann[c].astype(str)
ann["patient_id"] = ann["redcap_id"].map(red2pt)
assert ann["patient_id"].notna().all()
# population IDs: the cell table names populations <record ID>:tumor:<k>, the cytogenetic table <Pt-###>:tumor:<k>
ann["tumor_id"] = np.where(ann["annotation"] == "tumor", ann["patient_id"] + ":tumor:" + ann["tumor_id"].str.rsplit(":", n=1).str[-1], "nan")
log("PCs: %d | individuals: %d" % (len(ann), ann["patient_id"].nunique()))

cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"), index_col=0)
tum = ann[ann["annotation"] == "tumor"]
log("malignant PCs whose population is in the cytogenetic table: %.4f" % tum["tumor_id"].isin(cyto.index).mean())
assert tum["tumor_id"].isin(cyto.index).all()

# ---- Part A: the four two-population patients ---------------------------------------------------
two = cyto.groupby("patient_id").size()
two = sorted(two[two > 1].index)
log("\nPART A: patients with two malignant populations:", two)
rows, expr = [], []
for p in two:
    d = ann[ann["patient_id"] == p]
    t = d[d["annotation"] == "tumor"]
    pops = sorted(t["tumor_id"].unique(), key=lambda x: x.split(":")[-1])
    modal = {}
    for tid in pops:
        q = t[t["tumor_id"] == tid]
        h = q.loc[q["heavy_cdr3_aa"] != "nan", "heavy_cdr3_aa"]
        modal[tid] = h.value_counts().index[0]
    for tid in pops:
        q = t[t["tumor_id"] == tid]
        other = [o for o in pops if o != tid][0]
        h = q.loc[q["heavy_cdr3_aa"] != "nan", "heavy_cdr3_aa"]
        elsewhere = ann.loc[(ann["heavy_cdr3_aa"] == modal[tid]) & (ann["patient_id"] != p), "patient_id"].nunique()
        rows.append(dict(
            patient=p, population=tid.split(":")[-1], cytogenetics=cyto.loc[tid, "cytogenetics"], n_cells=len(q),
            pct_of_malignant=100 * len(q) / len(t),
            pct_cells_modal_heavy_cdr3=100 * (h == modal[tid]).mean(),
            pct_cells_with_other_population_cdr3=100 * (h == modal[other]).mean(),
            other_individuals_with_modal_cdr3=elsewhere,
            heavy_constant=q["heavy_c_gene"].mode().iloc[0], light_constant=q["light_c_gene"].mode().iloc[0],
            pct_light_mode=100 * (q["light_c_gene"] == q["light_c_gene"].mode().iloc[0]).mean(),
            n_libraries=q["gex_id"].nunique(), tissues="/".join(sorted(q["sample_type"].unique())),
            median_umi=q["total_counts"].median()))
    rows.append(dict(patient=p, population="normal PCs", n_cells=int((d["annotation"] == "normal").sum()),
                     median_umi=d.loc[d["annotation"] == "normal", "total_counts"].median()))
    rows.append(dict(patient=p, population="clinical immunofixation",
                     clinical_ifx=" | ".join(sorted(meta.loc[meta["patient_id"] == p, "IFX"].dropna().astype(str).unique()))))
    # partner genes: each population, the patient's normal PCs and all healthy-donor PCs
    groups = {("population " + tid.split(":")[-1] + " " + cyto.loc[tid, "cytogenetics"]): t[t["tumor_id"] == tid] for tid in pops}
    groups["normal PCs (patient)"] = d[d["annotation"] == "normal"]
    for g in sorted(set(PARTNER[c] for c in cyto.loc[pops, "cytogenetics"] if c in PARTNER)):
        for lab, q in list(groups.items()) + [("healthy-donor PCs", None)]:
            v = ann.loc[ann["patient_id"].isin(donors), g] if q is None else q[g]
            expr.append(dict(patient=p, gene=g, group=lab, n_cells=len(v), mean_expr=v.mean(), pct_expressing=100 * (v > 0).mean()))
A = pd.DataFrame(rows)
A.to_csv(OUT / "two_clone_tumors.tsv", sep="\t", index=False)
E = pd.DataFrame(expr)
E.to_csv(OUT / "two_clone_partner_genes.tsv", sep="\t", index=False)
log(A.round(2).to_string())
log(E.round(3).to_string())
log("\ndone")
