#!/usr/bin/env python3
"""Stage comparisons of tumor fraction, signature activity and malignancy and MST scores per patient (Figs. 3C and
4D,E; Extended Data Fig. 1B and Supplementary Figs. 3 and 4).

Purpose:      Units: the mean signature activity, malignancy score and MST score of the malignant or of the normal PCs
              of one patient at one disease stage in one tissue (BM or PB). SMM risk groups follow the IMWG 2/20/20
              model (risk.py); units that then share patient, stage, tissue and PC status are pooled (means weighted
              by cell numbers). Units without a value are left out. In each block (BM
              normal, BM malignant, PB normal, PB malignant), every pair of disease stages is compared by two-sided
              Wilcoxon rank-sum tests, Benjamini-Hochberg-adjusted within the block; malignant and normal PCs are
              compared at each disease stage within each tissue (two-sided Wilcoxon rank-sum tests, Benjamini-Hochberg-
              adjusted across stages). Tumor fraction: percentage of malignant PCs per patient and disease stage in BM
              and in PB samples, every pair of stages compared by two-sided Wilcoxon rank-sum tests,
              Benjamini-Hochberg-adjusted within each tissue.

Inputs:       data/paper/signatures/per-sample-mean-expr-score.pkl (patient_id, disease_status, sample_type,
              annotation, mean_expr:<signature>); data/paper/classifier/patient_id_mean_probability.pkl (sample,
              disease_status, sample_type, annotation, "Malignancy Score", "MST Score");
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (per cell: redcap_id, sample_type, annotation,
              disease_status; cells per unit and tumor fraction); data/metadata/tumor-cohort-metadata-final-240325.csv
              (redcap_id, patient_id; IMWG risk groups through risk.py).

Outputs:      results/stage_and_progression/stage_comparisons.tsv: one row per test: panel ("tumor fraction",
              "signature <name>", "score Malignancy Score", "score MST Score"), test (block), comparison, p, fdr.
              results/stage_and_progression/stage_comparisons_log.txt: pooled units, numbers of units and tests.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest as multi

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "stage_comparisons_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
GROUPS = ["Donor:BM:normal", "MGUS:BM:normal", "LRSMM:BM:normal", "IRSMM:BM:normal", "HRSMM:BM:normal", "MM:BM:normal",
          "MGUS:BM:tumor", "LRSMM:BM:tumor", "IRSMM:BM:tumor", "HRSMM:BM:tumor", "MM:BM:tumor",
          "Donor:PB:normal", "MGUS:PB:normal", "LRSMM:PB:normal", "IRSMM:PB:normal", "HRSMM:PB:normal", "MM:PB:normal",
          "MGUS:PB:tumor", "LRSMM:PB:tumor", "IRSMM:PB:tumor", "HRSMM:PB:tumor", "MM:PB:tumor"]

# ---- inputs ------------------------------------------------------------------------------------------
sig = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
prob = pd.read_pickle(data("paper", "classifier", "patient_id_mean_probability.pkl"))
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells[["redcap_id", "sample_type", "annotation", "disease_status"]].astype(str)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
cells["patient_id"] = cells["redcap_id"].map(meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"])
ncell = cells.groupby(["disease_status", "sample_type", "patient_id", "annotation"]).size()   # cells per unit (cell-table groups)
sig_cols = [c for c in sig.columns if c.startswith("mean_expr:") and not c.split(":")[1].endswith(("_UP", "_DN"))]


# ---- per-unit tables: IMWG risk groups, and pooling of units that become one (cell-count weighted means)
def clinical_groups(df, patient_col):
    d = df.copy()
    d["n"] = [ncell.get((r["disease_status"], r["sample_type"], r[patient_col], r["annotation"]), np.nan) for _, r in d.iterrows()]
    d["disease_status"] = stages(d[patient_col], d["sample_type"], d["disease_status"])
    num = [c for c in d.columns if c.startswith(("mean_expr:", "auc:")) or c in ("Malignancy Score", "MST Score")]
    out = []
    for key, g in d.groupby(["disease_status", "sample_type", patient_col, "annotation"], sort=False):
        r = g.iloc[0].copy()
        if len(g) > 1:
            w = g["n"].astype(float)
            for c in num:
                r[c] = np.average(g[c].astype(float), weights=w)
            r["n"] = w.sum()
            log("pooled %d units into %s" % (len(g), ":".join(key)))
        out.append(r)
    d = pd.DataFrame(out)
    d["group"] = d["disease_status"] + ":" + d["sample_type"] + ":" + d["annotation"]
    return d


def finish(d):
    d = d[d["group"].isin(GROUPS)].copy()
    d["group"] = pd.Categorical(d["group"], categories=GROUPS)
    d["disease_status"] = pd.Categorical(d["disease_status"], categories=STAGES)
    return d


sig_units = finish(clinical_groups(sig, "patient_id"))
prob_units = finish(clinical_groups(prob.rename(columns={"sample": "patient_id"}), "patient_id"))
for nm, b in [("signature units", sig_units), ("score units", prob_units)]:
    log("%s: %d; PB malignant units by stage %s" % (nm, len(b),
        b[b["group"].astype(str).str.endswith("PB:tumor")]["disease_status"].value_counts().reindex(STAGES[1:]).tolist()))


# ---- tests ---------------------------------------------------------------------------------------------
def cross_wilcox(df, value_col):
    """all pairs of disease stages, two-sided rank-sum, BH within the block"""
    cats = list(df["disease_status"].astype(str).unique())
    res = {"comparison": [], "p": []}
    for x, y in combinations(cats, 2):
        res["comparison"].append("%s,%s" % (x, y))
        res["p"].append(scipy.stats.ranksums(df.loc[df["disease_status"] == x, value_col], df.loc[df["disease_status"] == y, value_col])[1])
    res = pd.DataFrame(res)
    res["fdr"] = multi.fdrcorrection(res["p"], alpha=0.05, method="indep", is_sorted=False)[1] if len(res) else []
    return res


def tvn_wilcox(normal, tumor, value_col):
    """tumor vs normal PCs at each disease stage, two-sided rank-sum, BH across stages"""
    cats = sorted(set(normal["disease_status"].astype(str)) & set(tumor["disease_status"].astype(str)), key=STAGES.index)
    p = [scipy.stats.ranksums(normal.loc[normal["disease_status"] == c, value_col], tumor.loc[tumor["disease_status"] == c, value_col])[1]
         for c in cats]
    return pd.DataFrame({"comparison": cats, "p": p, "fdr": multi.fdrcorrection(p, alpha=0.05, method="indep", is_sorted=False)[1]})


def panel_tests(df, value_col):
    df = df[df[value_col].notna()]                    # units without a value are left out
    sub = lambda t, a: df[(df["sample_type"] == t) & (df["annotation"] == a)]
    return {"BM:tumor": cross_wilcox(sub("BM", "tumor"), value_col), "BM:normal": cross_wilcox(sub("BM", "normal"), value_col),
            "PB:tumor": cross_wilcox(sub("PB", "tumor"), value_col), "PB:normal": cross_wilcox(sub("PB", "normal"), value_col),
            "BM:TvN": tvn_wilcox(sub("BM", "normal"), sub("BM", "tumor"), value_col),
            "PB:TvN": tvn_wilcox(sub("PB", "normal"), sub("PB", "tumor"), value_col)}


rows = []


def record(panel, tests):
    for key in tests:
        t = tests[key].set_index("comparison")
        for comp in sorted(t.index):
            rows.append({"panel": panel, "test": key, "comparison": comp, "p": t.loc[comp, "p"], "fdr": t.loc[comp, "fdr"]})


# ---- signature activity (Fig. 3C, Supplementary Figs. 3 and 4) ---------------------------------------------------------------
for col in sig_cols:
    record("signature " + col.split(":")[1], panel_tests(sig_units, col))

# ---- malignancy and MST scores (Fig. 4D,E) -------------------------------------------------------------------------------
for col in ("Malignancy Score", "MST Score"):
    record("score " + col, panel_tests(prob_units, col))

# ---- tumor fraction per patient and stage, BM and PB (Extended Data Fig. 1B) --------------------------------------------
cells["stage_clinical"] = stages(cells["patient_id"], cells["sample_type"], cells["disease_status"])


def tumor_fraction(stage_col, tissue):
    x = cells[(cells[stage_col] != "Donor") & (cells["sample_type"] == tissue)]
    f = x.groupby(["patient_id", stage_col]).agg(tumor=("annotation", lambda s: (s == "tumor").sum()), N=("annotation", "size")).reset_index()
    f["%"] = f["tumor"] / f["N"] * 100
    return f.rename(columns={stage_col: "disease_status"})


def ranksums_test(df):
    pairs = list(combinations(df["disease_status"].unique(), 2))
    p = [scipy.stats.ranksums(df.loc[df["disease_status"] == x, "%"], df.loc[df["disease_status"] == y, "%"])[1] for x, y in pairs]
    return pd.DataFrame({"pair": [tuple(sorted(pr, key=STAGES.index)) for pr in pairs], "p": p,
                         "fdr": multi.fdrcorrection(p, alpha=0.05, method="indep", is_sorted=False)[1]})


for tissue in ("BM", "PB"):
    f = tumor_fraction("stage_clinical", tissue)
    q = ranksums_test(f)
    for pr, p, fdr in zip(q["pair"], q["p"], q["fdr"]):
        rows.append({"panel": "tumor fraction", "test": tissue, "comparison": ",".join(pr), "p": p, "fdr": fdr})
    log("tumor fraction, %s samples: %d patient units" % (tissue, len(f)))

# ---- all tests -----------------------------------------------------------------------------------------------------------
res = pd.DataFrame(rows, columns=["panel", "test", "comparison", "p", "fdr"])
res.to_csv(os.path.join(OUT, "stage_comparisons.tsv"), sep="\t", index=False)
log("panels: %d; tests: %d; q < 0.05: %d" % (res["panel"].nunique(), len(res), (res["fdr"] < 0.05).sum()))
FH.close()
