#!/usr/bin/env python3
"""Per-cell table of signature gene expression and signature scores for all PCs, and the trimmed signatures (top 5, 10,
15 and 20 genes of each signature) used as classifier feature sets (Methods, signature activity and classifiers).

Purpose:      1. Per-cell log-normalized expression of the union of the signature genes and of the genes of six
              stage-specific DEG lists (1,731 genes), with the cell's redcap_id, sample_type, catissue_id and gex_id,
              joined with the per-individual score tables of 06_signature_scores.py (files concatenated in name order).
              2. Trimmed signatures: for each signature, its genes sorted by Hotspot Z-score (descending) and the top 5,
              10, 15 and 20 kept, named MTS<n>:<signature>. Must be run with PYTHONHASHSEED=0.

Inputs:       data/signature/hotspot/consensus-modules_genes.csv (05_consensus_signatures.py): index = gene; context, Z.
              data/h5/agg/annotated-cell-log1p-240327.h5ad: log-normalized expression; obs redcap_id, sample_type,
              catissue_id, gex_id.
              data/signature/gene_list/cross_tvn_de/: early_downreg_36genes.txt, early_upreg_111genes.txt,
              late_downreg_89genes.txt, late_upreg_84genes.txt, smm-on_downreg_45genes.txt, smm-on_upreg_98genes.txt.
              data/signature/score/cohort/*.cohort.score.csv (06_signature_scores.py).

Outputs:      data/signature/score/cell-level-module-gene-expr-auc.pkl: one row per cell; obs columns, gene expression and
              the mean_expr:, sum_expr: and auc: scores.
              data/signature/hotspot/trimmed-consensus-modules_genes.csv: index = gene; context = MTS<n>:<signature>, C, Z,
              Pval, FDR.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import glob

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)
HS = data("signature", "hotspot")
DE = data("signature", "gene_list", "cross_tvn_de")

gene_module = pd.read_csv(HS / "consensus-modules_genes.csv", index_col=0)

# ---- expression of the signature and stage DEG genes in every cell ---------------------------------------------------------
adata = sc.read(data("h5", "agg", "annotated-cell-log1p-240327.h5ad"))
module_genes = list(set(gene_module.index))
de_genes = []
for name in ["early_downreg_36genes.txt", "early_upreg_111genes.txt", "late_downreg_89genes.txt", "late_upreg_84genes.txt",
             "smm-on_downreg_45genes.txt", "smm-on_upreg_98genes.txt"]:
    de_genes.extend([x.rstrip() for x in open(DE / name)])
de_genes = list(set(de_genes))
sig_genes = sorted(set(module_genes).union(set(de_genes)))
expr = sc.get.obs_df(adata, keys=["redcap_id", "sample_type", "catissue_id", "gex_id"] + sig_genes)

# ---- per-individual signature scores ----------------------------------------------------------------------------------------
input_files = sorted(glob.glob(str(data("signature", "score", "cohort") / "*.cohort.score.csv")))
scores = pd.concat([pd.read_csv(input_file, index_col=0) for input_file in input_files])
expr = expr.join(scores)
expr.to_pickle(data("signature", "score", "cell-level-module-gene-expr-auc.pkl"))

# ---- trimmed signatures: top 5, 10, 15 and 20 genes by Hotspot Z-score ------------------------------------------------------
trimmed_sigs = []
for sig_name, df in gene_module.groupby("context"):
    df = df.sort_values("Z", ascending=False)
    for top_n in [5, 10, 15, 20]:
        sub_sig = df.loc[df.index[:top_n]].copy()
        sub_sig["context"] = "MTS{}:".format(top_n) + sig_name
        trimmed_sigs.append(sub_sig)
pd.concat(trimmed_sigs).to_csv(HS / "trimmed-consensus-modules_genes.csv")
