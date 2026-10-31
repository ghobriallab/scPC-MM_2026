#!/usr/bin/env python3
"""Changes in signature activity and MST score of malignant PCs between the first and last samples of individuals
sampled at several visits, within all malignant PCs and within the largest shared CNV clone (Results, Supplementary
Fig. 14C paragraph; Methods, Longitudinal samples).

Purpose:      Individuals sampled at more than one time point (cohort metadata "timepoint"). For each individual,
              tissue and time point, malignant PCs are compared between the first and the last time point with at
              least 30 malignant PCs (two-sided Mann-Whitney test on cells): signature activity (mean log-normalized
              expression of the signature genes) and the MST score. Changes are expressed in units of the
              between-tumor standard deviation of the score (SD of the tumor means over the BM tumors of the cohort),
              so that a change of 1 equals the typical difference between two patients. Chromosomal changes: the
              Numbat clone of each malignant PC (Numbat run per patient on all its samples pooled) gives, per time
              point, the percentage of cells in each clone and with each driver arm CNV; the change in each score is
              also computed within the patient's largest clone with at least 30 cells at both time points. Signature
              scores are computed from the log-normalized matrix for the cells of these individuals only, and checked
              against the saved per-stage means. Only months between visits are written, no dates.

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv (assay_type, gex_id, rna_sample_id, patient_id,
              timepoint); data/paper/cohort-clinical-metadata.csv (rna_sample_id, sample_type, disease_status,
              sample_date, n_tumor_cell; SMM risk groups of the IMWG 2/20/20 model, risk.py); data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad (X, log-normalized, CSR; var names; obs
              index, gex_id, annotation, disease_status, sample_type); data/paper/signatures/consensus-modules_genes.csv
              (gene, context: the 16 signatures); data/paper/signatures/per-sample-mean-expr-score.pkl (patient_id,
              sample_type, annotation, disease_status, mean_expr:<signature>; check and between-tumor SD);
              data/paper/classifier/cell-level-prediction-knn30.pkl ("scPC-MM Score", the MST score of each cell);
              data/paper/classifier/patient_id_mean_probability.pkl ("MST Score" per unit; between-tumor SD);
              data/cnv/plasma-cell-cell-level-filtered-clone-250306.pkl (clone_id, numbat_annotation and driver arm
              CNV flags per malignant PC analyzed by Numbat).

Outputs:      results/stage_and_progression/longitudinal_visits.tsv: per individual x time point x tissue: months since the first visit, stage, malignant and Numbat-analyzed PCs, mean scores.
              results/stage_and_progression/longitudinal_clones.tsv: per individual x time point x tissue: percentage of Numbat-analyzed malignant PCs per clone and per driver arm CNV.
              results/stage_and_progression/longitudinal_changes.tsv: first vs last time point per individual x tissue x score: change (between-tumor SD), Mann-Whitney P, change within the largest shared clone.
              results/stage_and_progression/longitudinal_log.txt: visits, checks and summaries.

Dependencies: Python 3.8 with requirements.txt (h5py, numpy, pandas, scipy); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import gc

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "longitudinal_log.txt"), "w")
H5 = str(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad"))
MIN_CELLS = 30
RISK = {"Pt-28": "risk rose", "Pt-160": "risk fell"}               # IMWG 2/20/20 risk group at the two visits
FIG3D = ["histone_chaperone_gene", "B_cell_activation", "MHC-I_gene_module"]
DRIVERS = ["1q+", "1p-", "13q-", "14q-", "16q-", "17p-", "3q+", "6p+", "8p-", "8q+", "12p-", "20q-"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def read_obs(f, cols):
    g = f["obs"]
    out = {}
    for c in cols:
        it = g[c]
        if isinstance(it, h5py.Group) and "codes" in it:
            cats, codes = np.array(dec(it["categories"][...]) + ["nan"]), it["codes"][...]
            out[c] = cats[np.where(codes >= 0, codes, len(cats) - 1)]
        else:
            out[c] = dec(it[...])
    return pd.DataFrame(out, index=dec(g[g.attrs.get("_index", "_index")][...]))


# ---- 1. individuals, samples and visits ---------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
meta = meta[meta["assay_type"] == "GEX5"]
pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
w_bm = pub[pub["sample_type"] == "BM"].groupby(["patient_id", "disease_status"])["n_tumor_cell"].sum()   # malignant BM PCs per unit
pub["disease_status"] = stages(pub["patient_id"], pub["sample_type"], pub["disease_status"])      # IMWG risk groups
smp = meta[["gex_id", "rna_sample_id", "patient_id", "timepoint"]].merge(
    pub[["rna_sample_id", "sample_type", "disease_status", "sample_date"]], on="rna_sample_id", how="inner")
smp["date"] = pd.to_datetime(smp["sample_date"])
tp = smp.groupby("patient_id")["timepoint"].nunique()
pts = sorted(tp[tp > 1].index)
smp = smp[smp["patient_id"].isin(pts)].copy()
first = smp.groupby("patient_id")["date"].min()
visit_date = smp.groupby(["patient_id", "timepoint"])["date"].min()
smp["months"] = [round((visit_date[(p, t)] - first[p]).days / 30.44, 1) for p, t in zip(smp["patient_id"], smp["timepoint"])]
order = smp.groupby("patient_id")["months"].rank(method="dense").astype(int)
smp["visit"] = order
log("individuals with > 1 time point: %d; samples: %d" % (len(pts), len(smp)))
log(smp.groupby("patient_id").apply(lambda d: " | ".join("visit %d (%.1f mo): %s %s" % (v, m, t, s) for v, m, t, s in
    d.sort_values(["visit", "sample_type"])[["visit", "months", "sample_type", "disease_status"]].drop_duplicates().values)).to_string())

# ---- 2. cells of these individuals; signature scores from the log-normalized matrix -------------
f = h5py.File(H5, "r")
var = np.array(dec(f["var"][f["var"].attrs.get("_index", "_index")][...]))
obs = read_obs(f, ["gex_id", "annotation", "disease_status", "sample_type"])
obs["row"] = np.arange(len(obs))
cells = obs[obs["gex_id"].isin(smp["gex_id"])].copy()
cells["cell"] = cells.index
cells = cells.sort_values("row")
s_info = smp.set_index("gex_id")
cells["patient_id"] = cells["gex_id"].map(s_info["patient_id"])
cells["visit"] = cells["gex_id"].map(s_info["visit"])
cells["months"] = cells["gex_id"].map(s_info["months"])
cells["stage"] = cells["gex_id"].map(s_info["disease_status"])
del obs
gc.collect()
log("PCs of these individuals: %d (malignant %d)" % (len(cells), (cells["annotation"] == "tumor").sum()))

sigs = pd.read_csv(data("paper", "signatures", "consensus-modules_genes.csv"))
names = sorted(sigs["context"].unique())
pos = pd.Series(np.arange(len(var)), index=var)
W = np.zeros((len(var), len(names)), dtype=np.float32)
for j, n in enumerate(names):
    g = [x for x in sigs.loc[sigs["context"] == n, "gene"] if x in pos.index]
    W[pos[g].values, j] = 1.0 / len(g)
cols = np.where(W.any(axis=1))[0]
W = W[cols]
X = f["X"]
indptr = X["indptr"][:]
rows = cells["row"].values
breaks = np.where(np.diff(rows) > 1)[0] + 1
scores = np.zeros((len(rows), len(names)), dtype=np.float32)
k = 0
for run in np.split(rows, breaks):
    a, b = indptr[run[0]], indptr[run[-1] + 1]
    m = sp.csr_matrix((X["data"][a:b], X["indices"][a:b], indptr[run[0]:run[-1] + 2] - a), shape=(len(run), len(var)))
    scores[k:k + len(run)] = m[:, cols] @ W
    k += len(run)
f.close()
cells[names] = scores
log("signature scores computed for %d cells in %d contiguous reads" % (len(cells), len(breaks) + 1))

# check against the saved per-stage means, on units made of a single library (the saved means of units
# pooling several samples are not plain means over cells)
ref = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
ref_p = ref[ref["patient_id"].astype(str).isin(pts)]
keys = ["patient_id", "sample_type", "annotation", "disease_status"]
mine = cells.groupby(keys)[names].mean()
nlib = cells.groupby(keys)["gex_id"].nunique()
diffs = {"tumor": [], "normal": []}
for _, r in ref_p.iterrows():
    key = tuple(str(r[k]) for k in keys)
    if key in mine.index and nlib[key] == 1:
        diffs[key[2]].append(np.abs(mine.loc[key, names].values - r[["mean_expr:" + n for n in names]].values.astype(float)).max())
for k, v in diffs.items():
    log("check vs saved per-stage means (single-library units, %s PCs): %d units, max absolute difference %.1e" % (k, len(v), max(v)))
# between-tumor SD of each score: BM tumor units of the whole cohort


def imwg_bm_units(tab, pcol, cols):
    """BM tumor units with IMWG risk groups; units of one patient that share a group are pooled (cell-count weights)."""
    t = tab[(tab["sample_type"] == "BM") & (tab["annotation"] == "tumor")].copy()
    t[pcol], t["disease_status"] = t[pcol].astype(str), t["disease_status"].astype(str)
    t["w"] = [w_bm.get(k, np.nan) for k in zip(t[pcol], t["disease_status"])]
    t["disease_status"] = stages(t[pcol], t["sample_type"], t["disease_status"])
    return t.groupby([pcol, "disease_status"]).apply(lambda g: pd.Series(
        {c: np.average(g[c].astype(float), weights=g["w"]) if len(g) > 1 else float(g[c].iloc[0]) for c in cols}))


bm_ref = imwg_bm_units(ref, "patient_id", ["mean_expr:" + n for n in names])
SD = {n: bm_ref["mean_expr:" + n].std() for n in names}

# MST per cell (cells of the classifier output are named <barcode>-1-<rna_sample_id>)
pred = pd.read_pickle(data("paper", "classifier", "cell-level-prediction-knn30.pkl"))["scPC-MM Score"]
idx = cells["cell"].values
g2r = smp.set_index("gex_id")["rna_sample_id"]
paper_names = [i.split("-1-", 1)[0] + "-1-" + g2r[i.split("-1-", 1)[1]] for i in idx]
cells["MST"] = pred.reindex(paper_names).values
log("cells with an MST score: %.4f" % cells["MST"].notna().mean())
mbm = pd.read_pickle(data("paper", "classifier", "patient_id_mean_probability.pkl"))
mbm = imwg_bm_units(mbm, "sample", ["MST Score"])
SD["MST"] = mbm["MST Score"].std()
del pred
gc.collect()

# Numbat clones
cl = pd.read_pickle(data("cnv", "plasma-cell-cell-level-filtered-clone-250306.pkl"))
cl = cl[cl.index.isin(set(idx))][["clone_id", "numbat_annotation"] + DRIVERS]
cells = cells.join(cl, on="cell")
del cl
gc.collect()
tum = cells[cells["annotation"] == "tumor"].copy()
tum["in_numbat"] = tum["clone_id"].notna() & (tum["numbat_annotation"] == "tumor")
log("malignant PCs with a Numbat clone (tumor): %d of %d" % (tum["in_numbat"].sum(), len(tum)))

# ---- 3. per visit summaries ---------------------------------------------------------------------
scores_all = names + ["MST"]
V = tum.groupby(["patient_id", "visit", "sample_type"]).agg(months=("months", "first"), stage=("stage", "first"),
                                                             malignant_PCs=("MST", "size"), numbat_PCs=("in_numbat", "sum"))
V = V.join(tum.groupby(["patient_id", "visit", "sample_type"])[scores_all].mean())
V.reset_index().to_csv(os.path.join(OUT, "longitudinal_visits.tsv"), sep="\t", index=False)
log("\nper visit (malignant PCs):\n" + V[["months", "stage", "malignant_PCs", "numbat_PCs"] + FIG3D + ["MST"]].round(3).to_string())

# clone composition and driver CNVs per visit
crow = []
for (p, v, t), d in tum[tum["in_numbat"]].groupby(["patient_id", "visit", "sample_type"]):
    r = dict(patient_id=p, visit=v, sample_type=t, numbat_PCs=len(d))
    for c, n in d["clone_id"].value_counts(normalize=True).items():
        r["clone " + c.rsplit(":", 1)[-1]] = round(100 * n, 1)
    for e in DRIVERS:
        r[e] = round(100 * d[e].mean(), 1)
    crow.append(r)
C = pd.DataFrame(crow)
C.to_csv(os.path.join(OUT, "longitudinal_clones.tsv"), sep="\t", index=False)
log("\nNumbat clones and driver CNVs (% of Numbat-analyzed malignant PCs):\n" + C.fillna("").to_string())

# ---- 4. first vs last time point --------------------------------------------------------------
rows_out = []
for (p, t), d in tum.groupby(["patient_id", "sample_type"]):
    n = d.groupby("visit").size()
    ok = sorted(n[n >= MIN_CELLS].index)
    if len(ok) < 2:
        continue
    v0, v1 = ok[0], ok[-1]
    a, b = d[d["visit"] == v0], d[d["visit"] == v1]
    # largest Numbat clone present at both time points (>= 30 cells each)
    shared = None
    cnt = d[d["in_numbat"]].groupby(["clone_id", "visit"]).size().unstack(fill_value=0)
    if v0 in cnt.columns and v1 in cnt.columns:
        cnt = cnt[(cnt[v0] >= MIN_CELLS) & (cnt[v1] >= MIN_CELLS)]
        if len(cnt):
            shared = cnt.sum(axis=1).idxmax()
    # clone shift: largest change in the fraction of Numbat-analyzed malignant PCs in one clone (points)
    fr = d[d["in_numbat"]].groupby("visit")["clone_id"].value_counts(normalize=True).unstack(fill_value=0)
    shift = 100 * (fr.loc[v1] - fr.loc[v0]).abs().max() if v0 in fr.index and v1 in fr.index else np.nan
    for s in scores_all:
        x, y = a[s].dropna(), b[s].dropna()
        r = dict(patient_id=p, sample_type=t, risk=RISK.get(p, "stable"),
                 stages="%s -> %s" % (a["stage"].iloc[0], b["stage"].iloc[0]), months=b["months"].iloc[0] - a["months"].iloc[0],
                 n_first=len(x), n_last=len(y), clone_shift_points=shift, score=s, change_sd=(y.mean() - x.mean()) / SD[s],
                 P=mannwhitneyu(x, y, alternative="two-sided")[1])
        if shared is not None:
            xs, ys = a.loc[a["clone_id"] == shared, s].dropna(), b.loc[b["clone_id"] == shared, s].dropna()
            r.update(shared_clone_cells="%d/%d" % (len(xs), len(ys)), change_sd_shared_clone=(ys.mean() - xs.mean()) / SD[s],
                     P_shared_clone=mannwhitneyu(xs, ys, alternative="two-sided")[1])
        rows_out.append(r)
D = pd.DataFrame(rows_out)
D.to_csv(os.path.join(OUT, "longitudinal_changes.tsv"), sep="\t", index=False)
show = D[D["score"].isin(FIG3D + ["MST"])]
log("\nfirst vs last time point (change in between-tumor SD):\n" + show.round(3).to_string())
log("\n|change| in SD, median by risk group and score:\n" +
    D[D["score"].isin(FIG3D + ["MST"])].assign(abs_change=lambda x: x["change_sd"].abs())
    .groupby(["sample_type", "risk", "score"])["abs_change"].median().round(2).to_string())
log("\ndone")
