#!/usr/bin/env python3
"""Per-cell table of the trimmed signature scores for all PCs, and the feature-column lists of the trimmed signatures
used by the classifiers (Methods, classifiers).

Purpose:      Concatenates the per-individual tables of 08_trimmed_signature_scores.py (files in name order) and writes,
              for each score type (mean_expr, auc) and trim level (MTS5, MTS10, MTS15, MTS20), the list of its 16 feature
              columns. Must be run with PYTHONHASHSEED=0.

Inputs:       data/signature/score/cohort_trimmed/*.trimmed-out.csv (08_trimmed_signature_scores.py): one row per cell;
              mean_expr:MTS<n>:<signature>, auc:MTS<n>:<signature>.

Outputs:      data/signature/score/cell-level-trimmed-module-gene-expr-auc.pkl: one row per cell, all trimmed scores.
              data/classifier/datasets/feature_columns/{mean_expr,auc}_MTS{5,10,15,20}_16_gene_modules.txt: feature
              column names, one per line.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import glob

import numpy as np
import pandas as pd

np.random.seed(0)
FEATURES = data("classifier", "datasets", "feature_columns")
os.makedirs(FEATURES, exist_ok=True)

# ---- per-individual trimmed signature scores ----------------------------------------------------------------------------------
input_files = sorted(glob.glob(str(data("signature", "score", "cohort_trimmed") / "*.trimmed-out.csv")))
scores = pd.concat([pd.read_csv(input_file, index_col=0) for input_file in input_files])
scores.to_pickle(data("signature", "score", "cell-level-trimmed-module-gene-expr-auc.pkl"))


# ---- feature sets of the classifiers: one file per score type and trim level ------------------------------------------------
def write_feature_lines(feats, filename):
    with open(filename, "w") as fo:
        for ft in feats:
            fo.write(ft + "\n")


features = scores.columns
for ty in ["mean_expr", "auc"]:
    for tr in ["MTS5", "MTS10", "MTS15", "MTS20"]:
        prefix = ty + ":" + tr
        cols = list(features[features.str.startswith(prefix)])
        write_feature_lines(cols, FEATURES / "{}_{}_16_gene_modules.txt".format(ty, tr))
