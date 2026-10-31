#!/usr/bin/env python3
"""Per-cell tables of the published gene-set scores used by the classifiers for all PCs, with the expression of the
SenMayo and NF-kB index genes (Methods, Development of malignancy and MST scores).

Purpose:      For each gene set, concatenates the per-individual tables of 12_published_signature_scores.py (files in name
              order). For senmayo, the scores are joined to the per-cell log-normalized expression of its genes in the
              full annotated object; for nfkb, in the 5' PC object. Usage: 13_aggregate_published_scores.py [study ...];
              without arguments every study. Must be run with PYTHONHASHSEED=0.

Inputs:       data/signature/score/<folder>/*<suffix> (12_published_signature_scores.py).
              data/h5/agg/annotated-cell-log1p-240327.h5ad and data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad:
              log-normalized expression.
              data/pathway/external/senmayo_Saul_etal_NatCommun_2022.txt, nfkb_index_Annunziata_etal_CancerCell2007.txt.

Outputs:      data/signature/score/: cell-level-senescence-expr-auc.pkl, nfkb-index-cell-level-score.pkl,
              lin-etal-ras-signature-cell-level-score.pkl: one row per cell.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import gc
import glob

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)
EXT = data("pathway", "external")
SCORE = data("signature", "score")

# study: (score folder, file suffix, output table)
STUDIES = {
    "senmayo": ("aging", ".cohort.aging.csv", "cell-level-senescence-expr-auc.pkl"),
    "nfkb": ("nfkb", ".out.csv", "nfkb-index-cell-level-score.pkl"),
    "ras": ("ras_signature", ".ras-out.csv", "lin-etal-ras-signature-cell-level-score.pkl"),
}


def load_gene_list(input_file):
    return [x.rstrip() for x in open(input_file)]


def gene_keys(study):
    """genes whose expression is kept next to the scores"""
    if study == "senmayo":
        return load_gene_list(EXT / "senmayo_Saul_etal_NatCommun_2022.txt")
    if study == "nfkb":
        return load_gene_list(EXT / "nfkb_index_Annunziata_etal_CancerCell2007.txt")
    return None


def scores_of(study):
    folder, suffix, _ = STUDIES[study]
    input_files = sorted(glob.glob(str(SCORE / folder / ("*" + suffix))))
    return pd.concat([pd.read_csv(input_file, index_col=0) for input_file in input_files])


def aggregate(study, adata=None):
    out = SCORE / STUDIES[study][2]
    if adata is None:
        scores_of(study).to_pickle(out)
        return
    expr = sc.get.obs_df(adata, keys=gene_keys(study))
    expr = expr.join(scores_of(study))
    expr.to_pickle(out)


studies = sys.argv[1:] or list(STUDIES)
assert set(studies) <= set(STUDIES), "studies: " + ", ".join(STUDIES)
# studies with gene expression from the full annotated object, then from the 5' PC object, then scores only
for obj, group in [("annotated-cell-log1p-240327.h5ad", ["senmayo"]),
                   ("plasma-cell-GEX5-log1p-250207.h5ad", ["nfkb"])]:
    todo = [s for s in group if s in studies]
    if todo:
        adata = sc.read(data("h5", "agg", obj))
        for study in todo:
            aggregate(study, adata)
        del adata
        gc.collect()
for study in ["ras"]:
    if study in studies:
        aggregate(study)
