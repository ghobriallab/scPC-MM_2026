#!/usr/bin/env python3
"""Origin of the 30 nearest neighbors of each PC in the graph of the Fig. 1C UMAP, same-day and tissue mixing
tests, and per-patient UMAPs of same-visit BM and PB samples (Results; Methods, scRNA-seq data normalization and
clustering; Supplementary Fig. 3).

Purpose:      Asks whether the structure of the Fig. 1C UMAP follows batches, using the graph of that UMAP
              (each cell and its 30 nearest neighbors in PCA space). Samples collected on the same day are treated
              as one batch (batch label = collection date). Tests are observed / expected neighbors
              (1 = cells mix as if the factor did not matter).
              1. Percentage of a cell's neighbors from the same individual (individuals with >= 30 cells of the type).
              2a. Neighbors from other individuals with the same cell type and tissue: share sampled on the same day
                  against the share expected by chance (95% CI from 1,000 bootstrap resamples of individuals,
                  default_rng(42)); composition of each cell's neighbors (same sample, same individual's other samples,
                  other individuals on the same day or on other days), averaged per individual.
              2b. Malignant BM PCs of patients with BM samples from two or more visits: neighbors from another visit.
              2c. Same-visit BM and PB samples: neighbors from the other tissue among the patient's PCs of that visit.
              3. Per-patient UMAPs (immunoglobulin, mitochondrial, ribosomal and sex genes removed; highly variable
                 genes, scaling, PCA, 31 neighbors, UMAP; random_state 0) of the same-visit pair with the most malignant
                 PCs in its smaller sample at MGUS, SMM and MM (>= 100 malignant PCs per tissue; rule set in advance).

Inputs:       data/paper/anndata/plasma-cell-GEX5-umap.h5ad (Fig. 1C object): obs patient_id, annotation, sample_type,
              disease_status, total_counts; obsm X_umap; obsp nb_pca_distances; uns pca, nb_pca, umap, hvg parameters.
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: gex_id, total_counts (same cell order; checked).
              data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad and plasma-cell-GEX5-log1p-250207-var-names.txt.
              data/metadata/tumor-cohort-metadata-final-240325.csv: gex_id, sample_date, timepoint.

Outputs:      results/preprocessing/batch_same_individual.tsv: % of neighbors from the same individual.
              results/preprocessing/batch_mixing.tsv: observed and expected shares per unit and test.
              results/preprocessing/batch_neighbor_composition.tsv: neighbor composition per individual and cell type.
              results/preprocessing/batch_bm_pb_pairs.tsv: same-visit BM and PB pairs with their PC numbers.
              results/preprocessing/batch_patient_umaps.tsv.gz: UMAP coordinates of the selected pairs (Pt-### only).
              results/preprocessing/batch_neighbor_composition_log.txt: graph parameters and test results.

Dependencies: Python 3.8 with requirements.txt (h5py, numpy, pandas, scipy, scanpy, anndata); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import re

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

OUT = result_dir("preprocessing")
FH = open(OUT / "batch_neighbor_composition_log.txt", "w")
UMAP = str(data("paper", "anndata", "plasma-cell-GEX5-umap.h5ad"))
LOG1P = str(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad"))
VAR_NAMES = data("h5", "agg", "plasma-cell-GEX5-log1p-250207-var-names.txt")       # genes of LOG1P, one per line
MIN_CELLS = 30


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def read_col(g, c):
    it = g[c]
    if isinstance(it, h5py.Group) and "codes" in it:
        cats, codes = dec(it["categories"][...]), it["codes"][...]
        return np.where(codes >= 0, cats[np.maximum(codes, 0)], "nan")
    return it[...]


# ---- 0. the Fig. 1C object: parameters, cells, neighbor graph -------------------------------------------------------------
with h5py.File(UMAP, "r") as f:
    for k in ["pca", "nb_pca", "umap", "hvg"]:
        def walk(name, obj, k=k):
            if isinstance(obj, h5py.Dataset) and obj.size < 20:
                log("  uns/%s/%s = %s" % (k, name, obj[()]))
        f["uns"][k].visititems(walk)
    log("obsm keys: %s; obsp keys: %s" % (list(f["obsm"].keys()), list(f["obsp"].keys())))
    o = f["obs"]
    obs = pd.DataFrame({c: read_col(o, c) for c in ["patient_id", "annotation", "sample_type", "disease_status", "total_counts"]})
    g = f["obsp"]["nb_pca_distances"]
    D = sp.csr_matrix((g["data"][...], g["indices"][...], g["indptr"][...]), shape=(len(obs), len(obs)))
    xy = f["obsm"]["X_umap"][...]
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[["gex_id", "total_counts"]]
assert len(ann) == len(obs) and np.allclose(ann["total_counts"].values, obs["total_counts"].values.astype(float))
obs["gex_id"] = ann["gex_id"].astype(str).values
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv")).drop_duplicates("gex_id")
meta = meta.set_index(meta["gex_id"].astype(str))
obs["day"] = obs["gex_id"].map(meta["sample_date"])
obs["visit"] = obs["gex_id"].map(meta["timepoint"]).astype(str)
log("cells %d; neighbors per cell: %s; cells without processing day: %d"
    % (len(obs), np.unique(np.diff(D.indptr)).tolist(), obs["day"].isna().sum()))
nb = D.indices.reshape(len(obs), -1)                                   # 30 neighbors of each cell (self excluded)
pat = obs["patient_id"].values
mal = (obs["annotation"] == "tumor").values

# ---- 1. neighbors from the same individual --------------------------------------------------------------------------------
pcode = pd.factorize(pat)[0]
same = (pcode[nb] == pcode[:, None]).mean(axis=1)
S = pd.DataFrame({"patient": pat, "cell_type": np.where(mal, "malignant", "normal"), "same": same})
S = S.groupby(["patient", "cell_type"]).agg(n_cells=("same", "size"), pct_same_individual=("same", lambda v: 100 * v.mean())).reset_index()
S = S[S["n_cells"] >= MIN_CELLS]
S.to_csv(OUT / "batch_same_individual.tsv", sep="\t", index=False)
log("\n1. %% of neighbors from the same individual (individuals with >= %d cells of the type):" % MIN_CELLS)
log(S.groupby("cell_type")["pct_same_individual"].describe().round(1).to_string())


# ---- 2. batch tests ------------------------------------------------------------------------------------------------------------
def mixing(cells, pool, other):
    """mean over `cells` of: share of neighbors in `pool` that are in `other`, and the same share expected by chance
    (`other` members of the pool among all pool members except the cell itself). Returns (observed, expected, n)."""
    inpool = pool[nb[cells]]
    n_in = inpool.sum(axis=1)
    keep = n_in > 0
    obs_share = (inpool & other[nb[cells]]).sum(axis=1)[keep] / n_in[keep]
    exp_share = (other.sum() - other[cells][keep]) / (pool.sum() - pool[cells][keep])
    return obs_share.mean(), exp_share.mean(), int(keep.sum())


# a. same day: among a cell's neighbors from OTHER individuals with the same cell type and tissue, the share sampled on
#    the same day, against the share expected if neighbors were drawn at random from those cells
ct = np.where(mal, "malignant", "normal")
tis = obs["sample_type"].values
dcode = pd.factorize(obs["day"].fillna("NA"))[0]
dcode[obs["day"].isna().values] = -1
key = pd.DataFrame({"ct": ct, "tis": tis, "pat": pcode, "day": dcode})
n_ctt = key.groupby(["ct", "tis"])["day"].transform("size").values
n_ctt_pat = key.groupby(["ct", "tis", "pat"])["day"].transform("size").values
n_ctt_day = key.groupby(["ct", "tis", "day"])["pat"].transform("size").values
n_ctt_pat_day = key.groupby(["ct", "tis", "pat", "day"])["day"].transform("size").values
expected = (n_ctt_day - n_ctt_pat_day) / (n_ctt - n_ctt_pat)
ctc, tisc = mal.astype(np.int8), pd.factorize(tis)[0].astype(np.int8)
elig = (pcode[nb] != pcode[:, None]) & (ctc[nb] == ctc[:, None]) & (tisc[nb] == tisc[:, None])
n_el = elig.sum(axis=1)
observed = (elig & (dcode[nb] == dcode[:, None])).sum(axis=1) / np.maximum(n_el, 1)
ok = (n_el > 0) & (dcode >= 0)
DAY = pd.DataFrame({"patient": pat, "cell_type": ct, "observed": observed, "expected": expected})[ok]
rng = np.random.default_rng(42)
rows = []
for c in ["normal", "malignant"]:
    q = DAY[DAY["cell_type"] == c]
    per = q.groupby("patient")[["observed", "expected"]].sum()
    boot = []
    for _ in range(1000):
        b = per.iloc[rng.integers(0, len(per), len(per))]
        boot.append(b["observed"].sum() / b["expected"].sum())
    log("  %s PCs, neighbors from other individuals processed the same day (same cell type and tissue): observed %.2f%% vs "
        "expected %.2f%%; ratio %.2f (95%% CI %.2f-%.2f; %d individuals, %d cells)"
        % (c, 100 * q["observed"].mean(), 100 * q["expected"].mean(), q["observed"].sum() / q["expected"].sum(),
           np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(per), len(q)))
    for p_, z in q.groupby("patient"):
        if len(z) >= MIN_CELLS:
            rows.append(dict(test="%s PCs: other individuals processed the same day" % c, unit=p_, n_cells=len(z),
                             observed=z["observed"].mean(), expected=z["expected"].mean()))
# neighbor composition: same sample, same individual's other samples, other individuals processed the same day, others
skey = pd.factorize(obs["patient_id"] + "|" + obs["sample_type"] + "|" + obs["visit"])[0]
same_sample = skey[nb] == skey[:, None]
same_pat = pcode[nb] == pcode[:, None]
same_day_other = (~same_pat) & (dcode[nb] == dcode[:, None]) & (dcode[:, None] >= 0)
comp = pd.DataFrame({"patient": pat, "cell_type": ct, "same_sample": same_sample.mean(axis=1),
                     "same_individual_other_sample": (same_pat & ~same_sample).mean(axis=1),
                     "other_individual_same_day": same_day_other.mean(axis=1),
                     "other_individual_other_day": ((~same_pat) & ~same_day_other).mean(axis=1)})
C = comp.groupby(["patient", "cell_type"]).mean().mul(100)
C = C[comp.groupby(["patient", "cell_type"]).size() >= MIN_CELLS].reset_index()
C.to_csv(OUT / "batch_neighbor_composition.tsv", sep="\t", index=False)
log("  neighbor composition (%% of the 30 neighbors; median over individuals):")
log(C.groupby("cell_type").median(numeric_only=True).round(1).to_string())
# b. malignant BM PCs of one patient across visits (different sampling days)
bm = (obs["sample_type"] == "BM").values
for p in np.unique(pat[mal & bm]):
    m = mal & bm & (pat == p)
    v = obs["visit"].values
    visits = [x for x in np.unique(v[m]) if (m & (v == x)).sum() >= MIN_CELLS]
    if len(visits) < 2:
        continue
    m = m & np.isin(v, visits)
    for x in visits:
        cells = np.where(m & (v == x))[0]
        o_, e_, n_ = mixing(cells, m, m & (v != x))
        rows.append(dict(test="malignant PCs: other visits of the same patient", unit="%s visit %s" % (p, x), n_cells=n_, observed=o_, expected=e_))
# c. same-visit BM and PB samples (other tissue, same patient and visit)
pairs = []
for (p, x), q in obs.groupby(["patient_id", "visit"]):
    if set(q["sample_type"]) >= {"BM", "PB"}:
        for ct, flag in (("malignant", mal), ("normal", ~mal)):
            m = flag & (pat == p) & (obs["visit"].values == x)
            nbm, npb = (m & bm).sum(), (m & ~bm).sum()
            if ct == "malignant":
                st = stages([p], [q["sample_type"].iloc[0]], [q["disease_status"].iloc[0]])[0]   # IMWG risk group
                pairs.append(dict(patient=p, visit=x, stage=st, malignant_BM=nbm, malignant_PB=npb,
                                  normal_BM=((~mal) & (pat == p) & (obs["visit"].values == x) & bm).sum(),
                                  normal_PB=((~mal) & (pat == p) & (obs["visit"].values == x) & ~bm).sum()))
            if min(nbm, npb) < MIN_CELLS:
                continue
            for t, tm in (("BM", bm), ("PB", ~bm)):
                cells = np.where(m & tm)[0]
                o_, e_, n_ = mixing(cells, m, m & ~tm)
                rows.append(dict(test="%s PCs: other tissue of the same visit" % ct, unit="%s visit %s %s" % (p, x, t),
                                 n_cells=n_, observed=o_, expected=e_))
M = pd.DataFrame(rows)
M["ratio"] = M["observed"] / M["expected"]
M["patient"] = M["unit"].str.split().str[0]
M.to_csv(OUT / "batch_mixing.tsv", sep="\t", index=False)
log("\n2. observed / expected neighbors (1 = mixing as if the factor did not matter):")
for t, q in M.groupby("test"):
    # one value per patient: cell-weighted mean of its units
    r = q.groupby("patient").apply(lambda z: np.average(z["observed"], weights=z["n_cells"]) / np.average(z["expected"], weights=z["n_cells"]))
    log("  %s: %d patients (%d units); ratio median %.2f (IQR %.2f-%.2f); observed %.1f%% vs expected %.1f%% (medians)"
        % (t, len(r), len(q), r.median(), r.quantile(0.25), r.quantile(0.75), 100 * q["observed"].median(), 100 * q["expected"].median()))
P = pd.DataFrame(pairs)
P.to_csv(OUT / "batch_bm_pb_pairs.tsv", sep="\t", index=False)
log("same-visit BM+PB pairs: %d (%d patients); with >= %d malignant PCs in both: %d"
    % (len(P), P["patient"].nunique(), MIN_CELLS, ((P["malignant_BM"] >= MIN_CELLS) & (P["malignant_PB"] >= MIN_CELLS)).sum()))

# ---- 3. per-patient UMAPs of same-visit BM and PB samples --------------------------------------------------------------------
P["group"] = P["stage"].map(lambda s: "SMM" if "SMM" in s else s)
P["smaller"] = P[["malignant_BM", "malignant_PB"]].min(axis=1)
sel = P[(P["smaller"] >= 100) & P["group"].isin(["MGUS", "SMM", "MM"])].sort_values("smaller", ascending=False).groupby("group").head(1)
sel = sel.set_index("group").loc[[g for g in ["MGUS", "SMM", "MM"] if g in set(sel["group"])]].reset_index()
log("\n3. selected pairs (rule: >= 100 malignant PCs in each tissue; most malignant PCs in the smaller sample per stage):")
log(sel[["group", "patient", "stage", "malignant_BM", "malignant_PB", "normal_BM", "normal_PB"]].to_string(index=False))
genes = np.array([x.rstrip() for x in open(VAR_NAMES)])
drop = np.array([bool(re.match(r"^(IG[HKL]|MT-|RP[SL])", x)) or x in ("DDX3Y", "RPS4Y1", "XIST") for x in genes])
coords = []
with h5py.File(LOG1P, "r") as f:
    tc = read_col(f["obs"], "total_counts")
    assert np.allclose(tc.astype(float), obs["total_counts"].values.astype(float))   # same cell order as the Fig. 1C object
    X = f["X"]
    indptr = X["indptr"]
    for _, r in sel.iterrows():
        rows_ = np.where((pat == r["patient"]) & (obs["visit"].values == r["visit"]))[0]
        blocks = np.split(rows_, np.where(np.diff(rows_) != 1)[0] + 1)          # contiguous runs of rows (one per library)
        mats = []
        for b in blocks:
            a, z = indptr[b[0]], indptr[b[-1] + 1]
            ip = indptr[b[0]:b[-1] + 2] - a
            mats.append(sp.csr_matrix((X["data"][a:z], X["indices"][a:z], ip), shape=(len(b), len(genes))))
        A = ad.AnnData(sp.vstack(mats).tocsr()[:, ~drop])
        A.var_names = genes[~drop]
        sc.pp.highly_variable_genes(A, min_mean=0.0125, max_mean=3, min_disp=0.2)
        A = A[:, A.var["highly_variable"]].copy()
        sc.pp.scale(A, max_value=10)
        sc.tl.pca(A, svd_solver="arpack", random_state=0)
        sc.pp.neighbors(A, n_neighbors=31, random_state=0)
        sc.tl.umap(A, random_state=0)
        # mixing of BM and PB malignant cells in the patient's own graph
        Dp = A.obsp["distances"].tocsr()
        nbp = Dp.indices.reshape(A.n_obs, -1)
        t = obs["sample_type"].values[rows_] == "BM"
        mm = mal[rows_]
        share = [((~t[nbp[i]]) & mm[nbp[i]]).sum() / max(mm[nbp[i]].sum(), 1) if t[i] else (t[nbp[i]] & mm[nbp[i]]).sum() / max(mm[nbp[i]].sum(), 1)
                 for i in np.where(mm)[0]]
        exp = np.where(t[mm], (mm & ~t).sum() / (mm.sum() - 1), (mm & t).sum() / (mm.sum() - 1))
        log("  %s (%s): %d cells, %d HVGs; malignant BM/PB neighbors in the patient's own UMAP graph: observed %.1f%% vs expected %.1f%%"
            % (r["patient"], r["stage"], A.n_obs, A.n_vars, 100 * np.mean(share), 100 * exp.mean()))
        coords.append(pd.DataFrame({"patient": r["patient"], "stage": r["stage"], "tissue": obs["sample_type"].values[rows_],
                                    "cell_type": np.where(mal[rows_], "malignant", "normal"),
                                    "umap1": A.obsm["X_umap"][:, 0], "umap2": A.obsm["X_umap"][:, 1],
                                    "fig1c_umap1": xy[rows_, 0], "fig1c_umap2": xy[rows_, 1]}))
pd.concat(coords).to_csv(OUT / "batch_patient_umaps.tsv.gz", sep="\t", index=False)
log("done")
