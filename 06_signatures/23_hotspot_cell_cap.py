#!/usr/bin/env python3
"""Composition of the PCs used for signature discovery (at most 1,000 PCs per individual, tissue and cell type) compared
with all PCs, by disease stage, cytogenetic group, hyperdiploidy and age (Methods, detection of single-cell
transcriptional signatures).

Purpose:      The cap keeps min(n, 1,000) cells of each individual x tissue x cell type (malignant or normal), so the
              composition of the capped set follows from the cell counts alone (which cells are drawn does not change
              it); an individual sampled at two stages keeps cells of both stages in proportion to their numbers. For
              each variable, the share of PCs in each category is compared between all PCs and the capped set, with the
              share of individuals (or tumors) as the reference that the cap moves the cells towards. Age: median age
              at the first sample weighted by PCs (all and capped) and over individuals.

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: one row per PC; redcap_id, disease_status, sample_type,
              annotation (tumor/normal).
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id.
              data/paper/cohort-clinical-metadata.csv: patient_id, Age (age at the first sample).
              data/paper/cytogenetics-status.csv: patient_id, tumor_id, cytogenetics (group of the patient's main tumor,
              tumor 0), hrd_status.

Outputs:      results/signatures/hotspot_cell_cap_composition.tsv: per variable and category: cells and % (all, capped), units.
              results/signatures/hotspot_cell_cap_log.txt: cell totals, composition, age and largest individual share.

Dependencies: Python 3.8 with requirements.txt (pandas, numpy); config.py, risk.py (IMWG 2/20/20 risk groups).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import numpy as np
import pandas as pd

OUT = result_dir("signatures")
FH = open(os.path.join(OUT, "hotspot_cell_cap_log.txt"), "w")
CAP = 1000
STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
GROUP = {"t(11;14)": "CCND1 [t(11;14)]", "t(4;14)": "NSD2 [t(4;14)]", "t(14;16)": "MAF [t(14;16)]",
         "t(14;20)": "MAFB [t(14;20)]", "t(12;14)": "CCND2, CCND3 or MAFA", "t(6;14)": "CCND2, CCND3 or MAFA",
         "t(8;14)": "CCND2, CCND3 or MAFA", "HRD": "HRD", "Other": "No translocation or HRD"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


# ---- cells per individual x stage x tissue x cell type, and the capped counts -----------------------------------------
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells[["redcap_id", "disease_status", "sample_type", "annotation"]].astype(str)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), dtype=str)
to_pt = meta[["redcap_id", "patient_id"]].dropna().drop_duplicates().set_index("redcap_id")["patient_id"]
cells["patient_id"] = cells["redcap_id"].map(to_pt)
assert cells["patient_id"].notna().all()
cells["disease_status"] = stages(cells["patient_id"], cells["sample_type"], cells["disease_status"])   # IMWG
n = cells.groupby(["patient_id", "sample_type", "annotation", "disease_status"]).size().rename("all").reset_index()
# the cap applies per individual x tissue x cell type; an individual sampled at two stages keeps cells of both stages in
# proportion to their numbers
tot = n.groupby(["patient_id", "sample_type", "annotation"])["all"].transform("sum")
n["capped"] = n["all"] * np.minimum(tot, CAP) / tot
log("PCs: all %d, capped %d (%.1f%%); individuals %d" % (n["all"].sum(), round(n["capped"].sum()), 100 * n["capped"].sum() / n["all"].sum(),
                                                       n["patient_id"].nunique()))

clin = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
age = clin.groupby("patient_id")["Age"].min()
cyt = pd.read_csv(data("paper", "cytogenetics-status.csv"))
cyt = cyt[cyt["tumor_id"].str.endswith(":tumor:0")].set_index("patient_id")
n["stage"] = n["disease_status"]
n["group"] = n["patient_id"].map(cyt["cytogenetics"]).map(GROUP)
n["hrd"] = n["patient_id"].map(cyt["hrd_status"]).map({"Y": "Hyperdiploid", "N": "Not hyperdiploid"})
n["age_group"] = pd.cut(n["patient_id"].map(age), [0, 55, 65, 75, 120], right=False,
                        labels=["< 55 years", "55-64 years", "65-74 years", ">= 75 years"]).astype(str)


def share(d, var, cats, unit):
    """% of all and capped PCs per category; units = individuals (or tumors) with cells in that category"""
    rows = []
    for c in cats:
        x = d[d[var] == c]
        rows.append({"variable": var, "category": c, "cells_all": int(x["all"].sum()), "cells_capped": int(round(x["capped"].sum())),
                     "units": x[unit].nunique()})
    t = pd.DataFrame(rows)
    t["pct_all"] = 100 * t["cells_all"] / t["cells_all"].sum()
    t["pct_capped"] = 100 * t["cells_capped"] / t["cells_capped"].sum()
    t["pct_units"] = 100 * t["units"] / t["units"].sum()
    return t


n["unit"] = n["patient_id"] + ":" + n["disease_status"]
tabs = [share(n, "stage", STAGES, "unit").assign(cells="all PCs, BM and PB"),
        share(n[(n["annotation"] == "tumor") & n["group"].notna()], "group", list(dict.fromkeys(GROUP.values())), "patient_id")
        .assign(cells="malignant PCs, BM and PB"),
        share(n[(n["annotation"] == "tumor") & n["hrd"].notna()], "hrd", ["Hyperdiploid", "Not hyperdiploid"], "patient_id")
        .assign(cells="malignant PCs, BM and PB"),
        share(n[n["age_group"] != "nan"], "age_group", ["< 55 years", "55-64 years", "65-74 years", ">= 75 years"], "patient_id")
        .assign(cells="all PCs, BM and PB")]
res = pd.concat(tabs)
res.to_csv(os.path.join(OUT, "hotspot_cell_cap_composition.tsv"), sep="\t", index=False)
pd.set_option("display.width", 200)
log(res.round(1).to_string(index=False))
for t in tabs:
    log("%-10s largest change in share (percentage points): %.1f" % (t["variable"].iloc[0], (t["pct_capped"] - t["pct_all"]).abs().max()))
# age, cell-weighted medians
a = n.dropna(subset=["age_group"]).copy()
a["age"] = a["patient_id"].map(age)
a = a.dropna(subset=["age"])


def wmedian(v, w):
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), w.sum() / 2)]


log("age at first sample, median weighted by PCs: all %.1f, capped %.1f; median over individuals %.1f" % (
    wmedian(a["age"], a["all"]), wmedian(a["age"], a["capped"]), age.loc[a["patient_id"].unique()].median()))
ind = n.groupby("patient_id")[["all", "capped"]].sum()
log("largest share of one individual: all PCs %.1f%%, capped %.1f%%" % (100 * ind["all"].max() / ind["all"].sum(),
                                                                       100 * ind["capped"].max() / ind["capped"].sum()))
FH.close()
