#!/usr/bin/env python3
"""Activity of the 16 signatures across bone marrow cell types, and its association with disease stage in bone marrow
immune cells, in two public scRNA-seq datasets (Supplementary Fig. 7C; Results; Methods, signature activity in other bone
marrow cell types).

Purpose:      Signature activity = mean log-normalized expression (counts per 10,000, log1p) of the signature's genes in
              each cell. 1. Tirier et al., Nat Commun 2021 (GSE161801): whole bone marrow of patients with MM, PCs and
              other cells profiled together, with the authors' cell-type labels. Per signature and cell type: the median
              over samples of the sample mean, and the AUROC of activity for malignant PCs against cells of that type
              (within-sample pairs pooled over samples; 1 = always higher in malignant PCs, 0.5 = no difference).
              2. Zavidij et al., Nat Cancer 2020 (GSE124310): CD45+CD138- bone marrow immune cells (no PCs) of healthy
              donors and patients with MGUS, low- and high-risk SMM and MM, with the authors' cell-type labels. Spearman
              correlation of the sample mean with stage (healthy 0, MGUS 1, low-risk SMM 2, high-risk SMM 3, MM 4) per
              cell type with >= 3 stages and >= 8 samples, Benjamini-Hochberg over all signature x cell-type tests.
              Units with fewer than 20 cells of a type are left out.

Inputs:       data/signature/hotspot/consensus-modules_genes.csv: signature genes (index), column context = signature.
              data/h5/external/Tirier_NatCommun2021/expression/*.csv: UMI counts, genes x cells, one file per library.
              data/h5/external/Tirier_NatCommun2021/GSE161801_K43R_metadata_table.csv: Cell_barcode (index), patient,
              timepoint, major_celltype, celltype_1 (authors' labels).
              data/h5/external/Zavidij_NatCancer2020/log1p_count.h5ad: X (CSR, log-normalized), obs sample_id (stage
              prefix: NBM, MGUS, SMMl, SMMh, MM), cell_type (authors' labels).

Outputs:      results/external_datasets/signature_activity_bm_cell_types.tsv: signature x cell type: samples, cells, median, AUROC.
              results/external_datasets/signature_stage_bm_immune_cells.tsv: signature x cell type: samples, rho with stage, P, q.
              results/external_datasets/signature_activity_units.tsv: sample x cell type mean activities (public sample labels).
              results/external_datasets/signature_activity_bm_cell_types_log.txt: cell counts and summaries.

Dependencies: Python 3.8 with requirements.txt (pandas, numpy, scipy, scikit-learn, statsmodels, h5py); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import glob

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

OUT = result_dir("external_datasets")
FH = open(os.path.join(OUT, "signature_activity_bm_cell_types_log.txt"), "w")
MIN_CELLS = 20
TIR = data("h5", "external", "Tirier_NatCommun2021")
ZAV = data("h5", "external", "Zavidij_NatCancer2020", "log1p_count.h5ad")
# authors' labels -> common cell types (Tirier: major_celltype, celltype_1; small mixed labels left out)
TIRIER_TYPES = {("PCs", "Myeloma"): "Malignant PCs", ("PCs", "PC"): "Normal PCs", ("B", "B"): "B cells",
                ("B", "B_im"): "B cells", ("B", "PreB"): "Pre-B cells", ("B", "ProB"): "Pre-B cells",
                ("T_NK", "NK_dim"): "NK cells", ("T_NK", "NK_bright"): "NK cells",
                ("Myeloid", "Mono/Macro_CD14"): "CD14+ monocytes", ("Myeloid", "Mono/Macro_CD16"): "CD16+ monocytes",
                ("Myeloid", "cDC1"): "Myeloid DCs", ("Myeloid", "cDC2"): "Myeloid DCs", ("Myeloid", "pDC"): "pDCs",
                ("Er", "Er"): "Erythroid cells"}
ZAVIDIJ_TYPES = {"T-cells": "T cells", "CD14+ Monocytes": "CD14+ monocytes", "NK cells": "NK cells", "B-cells": "B cells",
                 "HPCs": "Progenitors", "CD16+ Monocytes": "CD16+ monocytes", "mDCs": "Myeloid DCs",
                 "pre-B-cells": "Pre-B cells", "pDCs": "pDCs", "Immature Neutrophils": "Immature neutrophils"}
STAGE = {"NBM": 0, "MGUS": 1, "SMMl": 2, "SMMh": 3, "MM": 4}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def tirier_type(major, fine):
    if major == "T_NK" and (major, fine) not in TIRIER_TYPES:
        return "T cells"                                     # all T-cell subsets, incl. NKT, gamma-delta, cycling
    if major == "Pro":
        return "Progenitors"
    return TIRIER_TYPES.get((major, fine))


sig = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
SIGS = sorted(sig["context"].unique())
SIG_GENES = sorted(set(sig.index))


def weights(genes):
    """genes x signatures matrix; column j averages the measured genes of signature j (zeros included)"""
    gi = {g: i for i, g in enumerate(genes)}
    W = np.zeros((len(genes), len(SIGS)))
    for j, s in enumerate(SIGS):
        idx = [gi[g] for g in sig.index[sig["context"] == s] if g in gi]
        W[idx, j] = 1.0 / len(idx)
    return W


# ---- 1. Tirier et al.: PCs and other bone marrow cells of the same samples ------------------------------------------------
meta = pd.read_csv(TIR / "GSE161801_K43R_metadata_table.csv", index_col=0, low_memory=False)
meta["cell_type"] = [tirier_type(a, b) for a, b in zip(meta["major_celltype"].astype(str), meta["celltype_1"].astype(str))]
meta["unit"] = meta["patient"].astype(str) + "_" + meta["timepoint"].astype(str)        # sorted fractions pooled per visit
acts = []
for f in sorted(glob.glob(str(TIR / "expression" / "*.csv"))):
    X = pd.read_csv(f, index_col=0)                          # UMI counts, genes x cells
    cells = X.columns[X.columns.isin(meta.index)]
    tot = X[cells].sum(0).values.astype(float)
    sub = X.reindex(SIG_GENES)[cells].fillna(0).values.astype(float)
    sub = np.log1p(sub / tot * 1e4)                          # counts per 10,000, log1p
    present = [g for g in SIG_GENES if g in X.index]
    W = weights(present)
    a = pd.DataFrame(sub[[SIG_GENES.index(g) for g in present]].T @ W, index=cells, columns=SIGS)
    acts.append(a)
    log("Tirier %s: %d cells" % (os.path.basename(f), len(cells)))
A = pd.concat(acts).join(meta[["unit", "cell_type"]])
log("Tirier signature genes measured: %d of %d" % (len(present), len(SIG_GENES)))
A = A[A["cell_type"].notna()]
cnt = A.groupby(["unit", "cell_type"]).size()
ok = cnt[cnt >= MIN_CELLS].index
A = A.set_index(["unit", "cell_type"]).loc[lambda d: d.index.isin(ok)].reset_index()
U_t = A.groupby(["unit", "cell_type"])[SIGS].mean().reset_index().assign(dataset="Tirier et al. 2021")
log("Tirier cells used: %d in %d samples; per type: %s" % (len(A), A["unit"].nunique(), A["cell_type"].value_counts().to_dict()))
rows = []
mal = A[A["cell_type"] == "Malignant PCs"]
for ct, d in A.groupby("cell_type"):
    for s in SIGS:
        auroc = np.nan
        if ct != "Malignant PCs":
            # AUROC pooled over samples that have both malignant PCs and this cell type (within-sample pairs)
            num = den = 0.0
            for u in set(d["unit"]) & set(mal["unit"]):
                x1, x0 = mal.loc[mal["unit"] == u, s].values, d.loc[d["unit"] == u, s].values
                num += roc_auc_score(np.r_[np.ones(len(x1)), np.zeros(len(x0))], np.r_[x1, x0]) * len(x1) * len(x0)
                den += len(x1) * len(x0)
            auroc = num / den if den else np.nan
        u = U_t[U_t["cell_type"] == ct]
        rows.append(dict(signature=s, cell_type=ct, n_samples=len(u), n_cells=len(d), median_activity=u[s].median(),
                         AUROC_malignant_PCs_vs_type=auroc))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "signature_activity_bm_cell_types.tsv"), sep="\t", index=False)
top = T.loc[T.groupby("signature")["median_activity"].idxmax(), ["signature", "cell_type"]]
log("\nTirier: cell type with the highest activity per signature:\n" + top.to_string(index=False))

# ---- 2. Zavidij et al.: non-PC immune cells across disease stages ---------------------------------------------------------
Z = h5py.File(str(ZAV), "r")
obs = pd.DataFrame({k: Z["obs"][k][:].astype(str) if not isinstance(Z["obs"][k], h5py.Group) else
                    Z["obs"][k]["categories"][:].astype(str)[Z["obs"][k]["codes"][:]] for k in ("sample_id", "cell_type")})
if "__categories" in Z["obs"]:                               # older anndata: categories stored separately
    for k in ("sample_id", "cell_type"):
        if k in Z["obs"]["__categories"]:
            obs[k] = Z["obs"]["__categories"][k][:].astype(str)[Z["obs"][k][:]]
genes = Z["var"]["_index"][:].astype(str)
X = sp.csr_matrix((Z["X"]["data"][:], Z["X"]["indices"][:], Z["X"]["indptr"][:]), shape=(len(obs), len(genes)))
gi = {g: i for i, g in enumerate(genes)}
present = [g for g in SIG_GENES if g in gi]
log("\nZavidij signature genes measured: %d of %d" % (len(present), len(SIG_GENES)))
act = np.asarray(X[:, [gi[g] for g in present]] @ weights(present))
B = pd.DataFrame(act, columns=SIGS)
B["unit"] = obs["sample_id"].values
B["cell_type"] = obs["cell_type"].str.strip().map(ZAVIDIJ_TYPES).values
B["stage"] = B["unit"].str.replace(r"-\d+$", "", regex=True).map(STAGE)
assert B["stage"].notna().all() and B["cell_type"].notna().all()
log("Zavidij cells %d, samples per stage: %s" % (len(B), B.drop_duplicates("unit")["stage"].value_counts().sort_index().to_dict()))
cnt = B.groupby(["unit", "cell_type"]).size()
U_z = B.groupby(["unit", "cell_type"])[SIGS + ["stage"]].mean()
U_z = U_z[cnt >= MIN_CELLS].reset_index().assign(dataset="Zavidij et al. 2020")
rows = []
for ct, u in U_z.groupby("cell_type"):
    for s in SIGS:
        if u["stage"].nunique() < 3 or len(u) < 8:
            continue
        rho, p = stats.spearmanr(u["stage"], u[s])
        rows.append(dict(signature=s, cell_type=ct, n_samples=len(u), rho_with_stage=rho, P=p))
S = pd.DataFrame(rows)
S["q"] = multipletests(S["P"], method="fdr_bh")[1]
S.to_csv(os.path.join(OUT, "signature_stage_bm_immune_cells.tsv"), sep="\t", index=False)
log("\nZavidij: stage correlations in non-PC cells, %d tests, q < 0.05: %d\n%s"
    % (len(S), (S["q"] < 0.05).sum(), S[S["q"] < 0.05].round(4).to_string(index=False)))
pd.concat([U_t, U_z]).to_csv(os.path.join(OUT, "signature_activity_units.tsv"), sep="\t", index=False)
log("done")
FH.close()
