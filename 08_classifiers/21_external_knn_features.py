#!/usr/bin/env python3
"""kNN-smoothed features of the cells of the public single-cell datasets, the inputs of their malignancy and MST scores
(Fig. 5C; Extended Data Figs. 7C and 8E; Methods, calculation of malignancy and MST scores on external single-cell data).

Purpose:      Smooths each cell's features over its k nearest neighbors (k = 30, as for the cohort; option --k): PCA of
              the dataset's log-normalized counts (highly variable genes with min_mean 0.0125, max_mean 3, min_disp 0.4,
              without ribosomal RP[SL], mitochondrial MT- and immunoglobulin IG[HKL] genes; scaled with max_value 10;
              arpack), k nearest neighbors of each cell in PCA space (ball tree, the cell itself excluded), then for each
              feature the mean of the k + 1 values of the cell and its neighbors, each times a weight (1 for the cell,
              exp(-distance / k) for a neighbor). Features: mean expression and AUCell scores of the 16 signatures, the
              six stage-wise DE gene sets and SenMayo, the five CNV-arm scores, and the translocation partner genes
              measured in the dataset (taken from the counts when the feature table does not hold all nine). One output
              per dataset: the datasets named on the command line, or all of them. Seeds 42 (Python random, numpy); the
              script restarts itself with PYTHONHASHSEED=0 when that variable is not set.

Inputs:       data/h5/external/<dataset folder>/log1p_count.h5ad: log-normalized counts (X) of each dataset script.
              data/signature/external/features/<dataset>_features.pkl: per-cell features of the dataset (the signature
              scores of 18_external_signature_scores.py and the CNV-arm scores of 19_external_cnv_arm_scores.py).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_6_de_gene_sets.txt,
              mean_expr_1_aging_signature.txt, auc_16_gene_modules.txt, auc_6_de_gene_sets.txt,
              auc_1_aging_signature.txt, mean_expr_fc_driver_cnv_arms.txt, cytogenetics_genes.txt.

Outputs:      data/signature/external/<dataset>_features-knn<k>.pkl: per cell, the smoothed features.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, scikit-learn); config.py.
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

DATASETS = {  # output name: folder under data/h5/external
    "boiarsky_natcommun2022": "Boiarsky_NatCommun2022",
    "ledergor_natmedicine2018": "CA3Data_Ledergor2018",
    "dang_cancercell2023": "Dang_CancerCell2023",
}
FEATURE_COLUMNS = data("classifier", "datasets", "feature_columns")


def compute_pca(adata):
    """PCA of the scaled highly variable genes (on a copy of the log-normalized object)"""
    a = adata.copy()
    a.uns["log1p"]["base"] = None
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


def compute_knn(adata, latent_obsm_key="X_pca", n_neighbors=30):
    """distances and indices of the k nearest neighbors of every cell (the cell itself excluded)"""
    coords = adata.obsm[latent_obsm_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", n_jobs=-1).fit(coords)
    dist, ind = nbrs.kneighbors()
    return dist, ind


def compute_weights(distances, use_kernel=True):
    """neighbor weights exp(-distance / k), or equal weights"""
    if use_kernel:
        return np.exp(-distances * 1.0 / distances.shape[1])
    else:
        d = distances.copy()
        d.fill(1)
        return d


def compute_smoothed_scores(expr, feats_names, adata, index, weights):
    """mean over the cell (weight 1) and its neighbors of each weighted feature value"""
    W = np.hstack((np.full(weights.shape[0], 1)[:, None], weights))
    I = np.hstack((np.arange(index.shape[0])[:, None], index))

    nb = expr.loc[adata.obs_names[I].reshape(-1), feats_names] * W.reshape(-1)[:, None]

    ci = 0
    window = weights.shape[1] + 1
    knn_expr = pd.DataFrame(index=adata.obs.index, columns=feats_names)
    for i in range(0, nb.shape[0], window):
        knn_expr.loc[adata.obs_names[ci]] = nb[i:i + window].mean()
        ci += 1
    return knn_expr


def load_list(input_file):
    return [x.rstrip() for x in open(input_file)]


def out(*parts):
    """an output file in the processed-data tree; its folder is created if needed"""
    p = data(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main(dataset, k):
    a = sc.read(data("h5", "external", DATASETS[dataset], "log1p_count.h5ad"))
    var_space = set(a.var_names)
    expr = pd.read_pickle(data("signature", "external", "features", "{}_features.pkl".format(dataset)))

    # feature names of each group, in the order of the lists
    mean_gene_modules = load_list(FEATURE_COLUMNS / "mean_expr_16_gene_modules.txt")
    mean_de_genes = load_list(FEATURE_COLUMNS / "mean_expr_6_de_gene_sets.txt")
    mean_aging_sigs = load_list(FEATURE_COLUMNS / "mean_expr_1_aging_signature.txt")
    auc_gene_modules = load_list(FEATURE_COLUMNS / "auc_16_gene_modules.txt")
    auc_de_genes = load_list(FEATURE_COLUMNS / "auc_6_de_gene_sets.txt")
    auc_aging_sigs = load_list(FEATURE_COLUMNS / "auc_1_aging_signature.txt")
    cyto_genes = load_list(FEATURE_COLUMNS / "cytogenetics_genes.txt")
    cnv_arms = load_list(FEATURE_COLUMNS / "mean_expr_fc_driver_cnv_arms.txt")

    # partner genes measured in the dataset, taken from the counts when the feature table does not hold all of them
    valid_cyto_genes = list(set(cyto_genes).intersection(var_space))
    missing_cyto_genes = list(set(cyto_genes).difference(var_space))
    if len(valid_cyto_genes) > 0 and len(cyto_genes) != len(set(expr.columns).intersection(set(cyto_genes))):
        cyto_expr = sc.get.obs_df(a, keys=valid_cyto_genes)
        expr = expr.join(cyto_expr)
    print("{} cyto genes missing: {}".format(dataset, ",".join(missing_cyto_genes)))

    feats_names = (mean_gene_modules + mean_de_genes + mean_aging_sigs + auc_gene_modules + auc_de_genes
                   + auc_aging_sigs + cnv_arms + valid_cyto_genes)

    a = compute_pca(a)
    dist, ind = compute_knn(a, latent_obsm_key="X_pca", n_neighbors=k)
    weights = compute_weights(dist)
    knn_expr = compute_smoothed_scores(expr, feats_names, a, ind, weights)
    knn_expr.to_pickle(out("signature", "external", "{}_features-knn{}.pkl".format(dataset, k)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kNN-smoothed features of the cells of the public single-cell datasets")
    parser.add_argument("--k", type=int, default=30, help="k for nearest neighbors (default: 30)")
    parser.add_argument("datasets", nargs="*", help="datasets to process (default: all): " + ", ".join(DATASETS))
    args = parser.parse_args()
    for d in args.datasets:
        if d not in DATASETS:
            parser.error("unknown dataset: " + d)
    for d in args.datasets or list(DATASETS):
        main(d, args.k)
