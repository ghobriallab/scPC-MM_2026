#!/usr/bin/env python3
"""kNN smoothing of the per-cell classifier features of the training and test sets, for k = 10 to 100 neighbors
(Methods, classifier development; Supplementary Fig. 12C).

Purpose:      Adds the NF-kB index and RAS signature scores (mean expression and AUCell) to the training and test tables,
              then, for each k, replaces every feature of a cell by the mean, over the cell and its k nearest neighbors,
              of the feature value times a weight (1 for the cell itself, exp(-distance / k) for each neighbor).
              Neighbors: Euclidean, ball tree, in the PCA space of the set's own cells (counts normalized to 10,000 per
              cell, log1p, highly variable genes with min_mean 0.0125, max_mean 3, min_disp 0.4, without RPS/RPL, MT- and
              IGH/IGK/IGL genes, scaled with max_value 10, ARPACK PCA). Two versions: neighbors within the whole set
              ("cohort") and within each individual ("patient", one PCA per individual). Seeds 42 (random, numpy); the
              script restarts itself with PYTHONHASHSEED=0 when that variable is not set.
              Usage: python3 02_knn_smoothing_train_test.py [k ...] (default: 10 20 30 40 50 60 70 80 90 100, one after
              the other).

Inputs:       data/h5/agg/plasma-cell-GEX5-counts-250207.h5ad: raw counts of all PCs.
              data/classifier/datasets/sc-train-expr_134samples.pkl, sc-test-expr_47samples.pkl (01_training_test_split.py):
              features and annotation columns (sample_type, redcap_id, catissue_id, gex_id, annotation, pct_counts_ig,
              pct_counts_mt, phase, G2M_score, S_score, total_counts, tumor_id, cytogenetics, hrd_status, disease_status).
              data/signature/score/nfkb-index-cell-level-score.pkl, lin-etal-ras-signature-cell-level-score.pkl: mean_expr:
              and auc: columns.

Outputs:      data/classifier/datasets/sc-train-expr_134samples.pkl, sc-test-expr_47samples.pkl: rewritten with the NF-kB
              and RAS columns (columns already present are left as they are).
              data/classifier/datasets/sc-train-cohort-smoothed-knn<k>-expr_134samples.pkl,
              sc-test-cohort-smoothed-knn<k>-expr_47samples.pkl, sc-train-patient-smoothed-knn<k>-expr_134samples.pkl,
              sc-test-patient-smoothed-knn<k>-expr_47samples.pkl: smoothed features + the annotation columns.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata, scikit-learn); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

if not os.getenv("PYTHONHASHSEED"):                                     # restart with PYTHONHASHSEED=0
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import random

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

random.seed(SEED)
np.random.seed(SEED)

DATASETS = ("classifier", "datasets")
# annotation columns carried along unchanged; every other column is a feature to smooth
MASK = ["sample_type", "redcap_id", "catissue_id", "gex_id", "annotation", "pct_counts_ig", "pct_counts_mt", "phase",
        "G2M_score", "S_score", "total_counts", "tumor_id", "cytogenetics", "hrd_status", "disease_status"]


def compute_pca(adata):
    a = adata.copy()

    # standard processing
    sc.pp.normalize_total(a, target_sum=1e4, inplace=True)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, min_mean=0.0125, max_mean=3, min_disp=0.4)

    # ribosomal, mitochondrial and immunoglobulin genes left out of the highly variable genes
    a.var.loc[a.var_names.str.match("^RP[SL]"), "highly_variable"] = False
    a.var.loc[a.var_names.str.match("^MT-"), "highly_variable"] = False
    a.var.loc[a.var_names.str.match("^IG[HKL]"), "highly_variable"] = False

    a.raw = a
    a = a[:, a.var.highly_variable]
    sc.pp.scale(a, max_value=10)

    sc.tl.pca(a, svd_solver="arpack")
    return a


def compute_knn(data, latent_obsm_key="X_pca", n_neighbors=30, is_input_latent=False):
    """k nearest neighbors of every cell (the cell itself excluded)"""
    if not is_input_latent:
        coords = data.obsm[latent_obsm_key]
    else:
        coords = data.copy()

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", n_jobs=-1).fit(coords)
    dist, ind = nbrs.kneighbors()
    return dist, ind


def compute_weights(distances, use_kernel=True):
    if use_kernel:
        return np.exp(-distances * 1.0 / distances.shape[1])
    else:
        d = distances.copy()
        d.fill(1)
        return d  # equal weights


def compute_smoothed_scores(expr, feats_names, cell_ids, index, weights):
    """mean over the cell and its neighbors of the weighted feature values (weight 1 for the cell itself)"""
    W = np.hstack((np.full(weights.shape[0], 1).reshape(-1, 1), weights))
    I = np.hstack((np.arange(index.shape[0]).reshape(-1, 1), index))

    nb = expr.loc[cell_ids[I.reshape(-1)], feats_names] * W.reshape(-1, 1)

    ci = 0
    window = weights.shape[1] + 1
    knn_expr = pd.DataFrame(index=cell_ids, columns=feats_names)
    for i in range(0, nb.shape[0], window):
        knn_expr.loc[cell_ids[ci]] = nb[i:i + window].mean()
        ci += 1
    return knn_expr


def add_scores():
    """NF-kB index and RAS signature scores (mean_expr: and auc: columns) joined to the training and test tables"""
    nfkb_index = pd.read_pickle(data("signature", "score", "nfkb-index-cell-level-score.pkl"))
    ras_signature = pd.read_pickle(data("signature", "score", "lin-etal-ras-signature-cell-level-score.pkl"))
    for name in ("sc-train-expr_134samples.pkl", "sc-test-expr_47samples.pkl"):
        expr = pd.read_pickle(data(*DATASETS, name))
        for scores in (nfkb_index, ras_signature):
            columns = [x for x in scores.columns if (x.startswith("mean_expr:") or x.startswith("auc:")) and
                       x not in expr.columns]
            expr = expr.join(scores[columns])
        expr.to_pickle(data(*DATASETS, name))


def smooth(nnb, a_train, a_test, train_expr, test_expr, feats_names):
    """cohort-level and patient-level smoothing with k = nnb"""
    # ---- cohort-level kNN: neighbors within the whole training (test) set
    print("start cohort kNN = {}".format(nnb), flush=True)
    a_tr = compute_pca(a_train)
    a_ts = compute_pca(a_test)

    dist, ind = compute_knn(a_tr, latent_obsm_key="X_pca", n_neighbors=nnb)
    weights = compute_weights(dist)
    knn_expr = compute_smoothed_scores(train_expr, feats_names, a_tr.obs.index, ind, weights)
    knn_expr = knn_expr.join(train_expr[MASK])
    knn_expr.to_pickle(data(*DATASETS, "sc-train-cohort-smoothed-knn{}-expr_134samples.pkl".format(nnb)))

    dist, ind = compute_knn(a_ts, latent_obsm_key="X_pca", n_neighbors=nnb)
    weights = compute_weights(dist)
    knn_expr = compute_smoothed_scores(test_expr, feats_names, a_ts.obs.index, ind, weights)
    knn_expr = knn_expr.join(test_expr[MASK])
    knn_expr.to_pickle(data(*DATASETS, "sc-test-cohort-smoothed-knn{}-expr_47samples.pkl".format(nnb)))

    # ---- patient-level kNN: one PCA and neighbor search per individual
    print("start sample kNN = {}".format(nnb), flush=True)
    for a_set, expr, name in ((a_train, train_expr, "sc-train-patient-smoothed-knn{}-expr_134samples.pkl"),
                              (a_test, test_expr, "sc-test-patient-smoothed-knn{}-expr_47samples.pkl")):
        knn_expr = []
        for redcap_id, _ in a_set.obs.groupby("redcap_id"):
            a_pat = a_set[a_set.obs["redcap_id"] == redcap_id].copy()
            a_pat = compute_pca(a_pat)

            dist, ind = compute_knn(a_pat, latent_obsm_key="X_pca", n_neighbors=nnb)
            weights = compute_weights(dist)

            knn_i = compute_smoothed_scores(expr, feats_names, a_pat.obs.index, ind, weights)
            knn_expr.append(knn_i)
        knn_expr = pd.concat(knn_expr)
        knn_expr = knn_expr.join(expr[MASK])
        knn_expr.to_pickle(data(*DATASETS, name.format(nnb)))


parser = argparse.ArgumentParser(description="kNN smoothing of the training and test features")
parser.add_argument("k", nargs="*", type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    help="numbers of nearest neighbors")
args = parser.parse_args()

add_scores()

adata = sc.read(data("h5", "agg", "plasma-cell-GEX5-counts-250207.h5ad"))
train_expr = pd.read_pickle(data(*DATASETS, "sc-train-expr_134samples.pkl"))
test_expr = pd.read_pickle(data(*DATASETS, "sc-test-expr_47samples.pkl"))
a_train = adata[train_expr.index].copy()
a_test = adata[test_expr.index].copy()
del adata
feats_names = [col for col in train_expr.columns if col not in MASK]

for nnb in args.k:
    smooth(nnb, a_train, a_test, train_expr, test_expr, feats_names)
print("done")
