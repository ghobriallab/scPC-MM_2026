#!/usr/bin/env python3
"""kNN smoothing of the classifier features of all PCs, k = 30 and 50: the input of the cell-level malignancy and MST
scores (Methods, classifier development).

Purpose:      Same smoothing as 02_knn_smoothing_train_test.py, on every GEX5 cell of the annotated object at once. PCA of
              all cells (counts normalized to 10,000 per cell, log1p, highly variable genes with min_mean 0.0125,
              max_mean 3, min_disp 0.4, without RPS/RPL, MT- and IGH/IGK/IGL genes, scaled with max_value 10, ARPACK PCA);
              approximate nearest neighbors with pynndescent (NNDescent with k + 1 neighbors, the first one, the cell
              itself, dropped); each feature replaced by the mean, over the cell and its k neighbors, of the value times
              a weight (1 for the cell itself, exp(-distance / k) for each neighbor). Features: mean expression and
              AUCell scores of the 16 signatures and of the six stage-wise DE gene sets, SenMayo scores, the nine
              translocation partner genes and the five CNV-arm scores. Seeds 42 (random, numpy); the script restarts
              itself with PYTHONHASHSEED=0 when that variable is not set.

Inputs:       data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_6_de_gene_sets.txt,
              mean_expr_1_aging_signature.txt, cytogenetics_genes.txt, auc_16_gene_modules.txt, auc_6_de_gene_sets.txt,
              auc_1_aging_signature.txt, mean_expr_fc_driver_cnv_arms.txt (01_training_test_split.py).
              data/signature/score/cell-level-copy-number-full-gene-expr.pkl, cell-level-cytogenetics-gene-expr.pkl,
              cell-level-senescence-expr-auc.pkl, cell-level-module-gene-expr-auc.pkl: the feature columns above.
              data/h5/agg/annotated-cell-counts-240327.h5ad: raw counts; obs assay_type (GEX5 cells kept).

Outputs:      data/signature/score/cell-level-cohort-smoothed-knn30.pkl, cell-level-cohort-smoothed-knn50.pkl: smoothed
              features of every GEX5 cell.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata, scikit-learn) and pynndescent; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

if not os.getenv("PYTHONHASHSEED"):                                     # restart with PYTHONHASHSEED=0
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import random

import numpy as np
import pandas as pd
import scanpy as sc
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors

random.seed(SEED)
np.random.seed(SEED)


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


def compute_knn(data, latent_obsm_key="X_pca", n_neighbors=30, is_input_latent=False, approx_neighbors=False):
    """k nearest neighbors of every cell (the cell itself excluded)"""
    if not is_input_latent:
        coords = data.obsm[latent_obsm_key]
    else:
        coords = data.copy()

    if approx_neighbors:
        index = NNDescent(coords, n_neighbors=n_neighbors + 1)
        ind, dist = index.neighbor_graph
        ind, dist = ind[:, 1:], dist[:, 1:]
    else:
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
    W = np.hstack((np.full(weights.shape[0], 1)[:, None], weights))
    I = np.hstack((np.arange(index.shape[0])[:, None], index))

    nb = expr.loc[cell_ids[I].reshape(-1), feats_names] * W.reshape(-1)[:, None]

    ci = 0
    window = weights.shape[1] + 1
    knn_expr = pd.DataFrame(index=cell_ids, columns=feats_names)
    for i in range(0, nb.shape[0], window):
        knn_expr.loc[cell_ids[ci]] = nb[i:i + window].mean()
        ci += 1
    return knn_expr


def load_list(name):
    return [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", name))]


# ---- per-cell features --------------------------------------------------------------------------------------------------
mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
mean_de_modules = load_list("mean_expr_6_de_gene_sets.txt")
mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
cyto_genes = load_list("cytogenetics_genes.txt")
auc_gene_modules = load_list("auc_16_gene_modules.txt")
auc_de_modules = load_list("auc_6_de_gene_sets.txt")
auc_aging_sigs = load_list("auc_1_aging_signature.txt")
cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

feats = mean_gene_modules + mean_de_modules + mean_aging_sigs + cyto_genes
feats = feats + auc_gene_modules + auc_de_modules + auc_aging_sigs + cnv_arms

cnv_scores = pd.read_pickle(data("signature", "score", "cell-level-copy-number-full-gene-expr.pkl"))[cnv_arms]
cyto_expr = pd.read_pickle(data("signature", "score", "cell-level-cytogenetics-gene-expr.pkl"))[cyto_genes]
aging_expr = pd.read_pickle(data("signature", "score", "cell-level-senescence-expr-auc.pkl"))[mean_aging_sigs + auc_aging_sigs]

module_expr = pd.read_pickle(data("signature", "score", "cell-level-module-gene-expr-auc.pkl"))
module_expr = module_expr[mean_gene_modules + auc_gene_modules + mean_de_modules + auc_de_modules]

expr = module_expr.join(aging_expr).join(cyto_expr).join(cnv_scores)

# ---- PCA of all GEX5 cells and smoothing ----------------------------------------------------------------------------------
adata = sc.read(data("h5", "agg", "annotated-cell-counts-240327.h5ad"))
adata = adata[adata.obs["assay_type"] == "GEX5"]
adata = compute_pca(adata)
coords = adata.obsm["X_pca"].copy()

for nnb in [30, 50]:
    dist, ind = compute_knn(coords, latent_obsm_key="X_pca", n_neighbors=nnb, is_input_latent=True, approx_neighbors=True)
    weights = compute_weights(dist)
    knn_expr = compute_smoothed_scores(expr, feats, adata.obs.index, ind, weights)
    knn_expr.to_pickle(data("signature", "score", "cell-level-cohort-smoothed-knn{}.pkl".format(nnb)))
    print("wrote cell-level-cohort-smoothed-knn{}.pkl".format(nnb), flush=True)
