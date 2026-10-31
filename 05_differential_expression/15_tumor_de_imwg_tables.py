#!/usr/bin/env python3
"""Within-patient differential expression tables of malignant vs normal BM PCs labelled with the IMWG 2/20/20 risk
groups (Fig. 2A,D-G; Supplementary Figs. 5, 6; Results).

Purpose:      Columns of the per-tumor tables are named <patient>_BM_<stage>_tumor_<k> (one tumor = the malignant
              population k of one patient at one disease stage, BM samples of that stage pooled). Each column takes the
              stage given by risk.py. Where two tumors of one patient now share a risk group (a patient sampled at two
              visits), their cells form one tumor and the comparison is run again on the pooled cells, as in
              13_scrna_de_tumor_vs_normal.py: two-sided Wilcoxon rank-sum test of the malignant population against the
              normal PCs of the same tissue and stage (scanpy rank_genes_groups, Benjamini-Hochberg adjusted P), on the
              log-normalized expression of the cohort object. The same computation is first run on the patient's
              recorded tumors and must reproduce their saved columns. The other tumors keep their results.

Inputs:       data/paper/degs/scrna/BM_scRNA_tumor_DE_FDR.csv, BM_scRNA_tumor_DE_logFoldChange.csv: genes x tumors.
              data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad: X (log-normalized, CSR), var names, obs gex_id,
              sample_type, annotation, disease_status, tumor_id.
              data/metadata/tumor-cohort-metadata-final-240325.csv: gex_id, patient_id.

Outputs:      results/differential_expression/BM_scRNA_tumor_DE_FDR.csv, BM_scRNA_tumor_DE_logFoldChange.csv: genes x tumors.
              results/differential_expression/tumor_de_imwg_log.txt: relabelled and pooled tumors, reproduction check.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, h5py, anndata, scanpy); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py, risk.py
from config import data, result_dir
from risk import stages

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

OUT = result_dir("differential_expression")
FH = open(os.path.join(OUT, "tumor_de_imwg_log.txt"), "w")
MIN_CELLS = 30


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def obs_col(f, k):
    """One obs column of an h5ad file (categoricals stored as codes + categories)."""
    g = f["obs"][k]
    if isinstance(g, h5py.Group):
        return dec(g["categories"][...])[g["codes"][...]]
    if "__categories" in f["obs"] and k in f["obs"]["__categories"]:
        return dec(f["obs"]["__categories"][k][...])[g[...]]
    return dec(g[...])


def patient_object(f, obs, patient):
    """AnnData of one patient's PCs (rows read in contiguous runs)."""
    rows = np.where((obs["patient_id"] == patient).values)[0]
    X, ip = f["X"], f["X"]["indptr"]
    parts = []
    for run in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1):
        a, b = ip[run[0]], ip[run[-1] + 1]
        parts.append(sp.csr_matrix((X["data"][a:b], X["indices"][a:b], ip[run[0]:run[-1] + 2] - a),
                                   shape=(len(run), X.attrs["shape"][1])))
    o = obs.iloc[rows].reset_index(drop=True).astype(str)
    o.index = o.index.astype(str)
    return ad.AnnData(X=sp.vstack(parts).tocsr(), obs=o, var=pd.DataFrame(index=dec(f["var"][f["var"].attrs["_index"]][...])))


def wilcoxon(A, stage_col, tissue, stage, k):
    """Malignant population k vs normal PCs of the same tissue and stage; None below 30 cells in either group."""
    o = A.obs[["sample_type", stage_col, "annotation", "k"]].astype(str)
    grp = o["sample_type"] + ":" + o[stage_col] + ":" + o["annotation"] + np.where(o["annotation"] == "tumor", ":" + o["k"], "")
    A.obs["group"] = pd.Categorical(grp)
    t, n = "%s:%s:tumor:%s" % (tissue, stage, k), "%s:%s:normal" % (tissue, stage)
    if (A.obs["group"] == t).sum() < MIN_CELLS or (A.obs["group"] == n).sum() < MIN_CELLS:
        return None
    sc.tl.rank_genes_groups(A, "group", groups=[t], reference=n, method="wilcoxon", key_added="de")
    return sc.get.rank_genes_groups_df(A, key="de", group=t).set_index("names")


rd = lambda n: pd.read_csv(data("paper", "degs", "scrna", n), index_col=0)
fdr, lfc = rd("BM_scRNA_tumor_DE_FDR.csv"), rd("BM_scRNA_tumor_DE_logFoldChange.csv")
assert list(fdr.columns) == list(lfc.columns)
parts = [c.split("_") for c in fdr.columns]                       # patient, BM, stage, tumor, k
new = ["_".join([p[0], p[1], s] + p[3:]) for p, s in zip(parts, stages([p[0] for p in parts], [p[1] for p in parts],
                                                                          [p[2] for p in parts]))]
names = pd.Series(new, index=fdr.columns)
log("tumors: %d; relabelled: %d" % (len(names), (names != names.index).sum()))
groups = names.groupby(names).groups
pooled = {k: list(v) for k, v in groups.items() if len(v) > 1}
log("tumors pooled: %d" % len(pooled))

cols_f = {k: fdr[list(v)[0]] for k, v in groups.items() if len(v) == 1}
cols_l = {k: lfc[list(v)[0]] for k, v in groups.items() if len(v) == 1}
if pooled:
    f = h5py.File(str(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad")), "r")
    obs = pd.DataFrame({k: obs_col(f, k) for k in ["gex_id", "sample_type", "annotation", "disease_status", "tumor_id"]})
    meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), usecols=["gex_id", "patient_id"])
    obs["patient_id"] = obs["gex_id"].map(meta.drop_duplicates("gex_id").set_index("gex_id")["patient_id"])
    for name, old in pooled.items():
        patient, tissue, stage, _, k = name.split("_")
        A = patient_object(f, obs, patient)
        A.obs["k"] = [str(t).split(":")[-1] for t in A.obs["tumor_id"]]
        A.obs["imwg"] = stages(A.obs["patient_id"], A.obs["sample_type"], A.obs["disease_status"])
        for c in old:                                             # the recorded tumors must be reproduced
            p_ = c.split("_")
            r = wilcoxon(A, "disease_status", p_[1], p_[2], p_[4])
            dq = np.nanmax(np.abs(r["pvals_adj"].reindex(fdr.index) - fdr[c]))
            dl = np.nanmax(np.abs(r["logfoldchanges"].reindex(fdr.index) - lfc[c]))
            assert dq < 1e-12 and dl < 1e-5, (dq, dl)
            log("  check, recorded tumor of %s (%s): max |difference| q %.1e, log2 FC %.1e" % (patient, p_[2], dq, dl))
        r = wilcoxon(A, "imwg", tissue, stage, k)
        assert r is not None
        cols_f[name], cols_l[name] = r["pvals_adj"].reindex(fdr.index), r["logfoldchanges"].reindex(fdr.index)
        log("pooled tumor %s from %d tumors: %d malignant and %d normal PCs" % (
            name, len(old), int((A.obs["group"] == "%s:%s:tumor:%s" % (tissue, stage, k)).sum()),
            int((A.obs["group"] == "%s:%s:normal" % (tissue, stage)).sum())))
    f.close()
order = sorted(cols_f)
F, L = pd.DataFrame(cols_f)[order], pd.DataFrame(cols_l)[order]
F.index.name = L.index.name = fdr.index.name
F.to_csv(os.path.join(OUT, "BM_scRNA_tumor_DE_FDR.csv"))
L.to_csv(os.path.join(OUT, "BM_scRNA_tumor_DE_logFoldChange.csv"))
log("tumors written: %d (main populations %d)" % (len(order), sum(c.endswith("_0") for c in order)))
FH.close()
