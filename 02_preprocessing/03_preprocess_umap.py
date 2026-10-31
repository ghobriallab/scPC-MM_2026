#!/usr/bin/env python3
"""Normalization, highly variable genes, PCA, Harmony integration, neighbor graphs and UMAPs of one aggregated plasma
cell object (input of the Leiden clustering).

Purpose:      Total counts scaled to 10,000 per cell and log1p; highly variable genes (min_mean 0.01, max_mean 3,
              min_disp from the command line, default 0.2) without immunoglobulin genes (^IG[HKL]) and the sex genes
              DDX3Y, RPS4Y1 and XIST; the log-normalized matrix is kept in .raw; highly variable genes scaled
              (max_value 10); PCA (arpack). Harmony (max_iter_harmony 50) corrects the PCs by library (gex_id;
              X_sid_harmony) and by assay version (assay_type, GEX3 or GEX5, read from the library name when absent;
              X_gex_harmony). A 31-nearest-neighbor graph and a UMAP are computed on each of the three representations
              (graphs nb_pca, nb_sid_harmony, nb_gex_harmony; UMAPs X_umap_pca, X_umap_sid_harmony, X_umap_gex_harmony).
              The script restarts itself with PYTHONHASHSEED=0 when the variable is not set.
              Run once per object, e.g. python3 02_preprocessing/03_preprocess_umap.py --min-disp 0.2
              plasma-cell-aug2023_raw.h5ad. Objects and min_disp: plasma-cell-, tumor-cell- and
              bm-tumor-cell-aug2023_raw.h5ad (0.2); normal-cell- and bm-normal-cell-aug2023_raw.h5ad (0.2 and 0.3);
              pb-normal-cell- and pb-tumor-cell-aug2023_raw.h5ad (0.3).

Inputs:       data/h5/agg/<object>.h5ad: raw counts in X; obs gex_id (and assay_type when present).

Outputs:      data/h5/agg/<object>_md<min_disp>_umap.h5ad: highly variable genes (scaled), .raw log-normalized, obsm
              X_pca, X_sid_harmony, X_gex_harmony and the three UMAPs, the three graphs.
              data/h5/agg/<object>_md<min_disp>.log: cell and gene numbers and the steps run.

Dependencies: Python 3.8 with requirements.txt (scanpy, anndata) and harmonypy; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

# Restart with PYTHONHASHSEED=0 when it is not set, so that hash-dependent orderings are reproducible
if not os.getenv("PYTHONHASHSEED"):
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse

import scanpy as sc


def compute_umap(adata, key):
    """UMAP on one neighbor graph, stored as obsm X_umap_<name> and uns umap_<name> (name = graph key without 'nb_')."""
    name = key[len("nb_"):]
    sc.tl.umap(adata, neighbors_key=key)
    adata.uns["umap_" + name] = adata.uns["umap"]
    adata.obsm["X_umap_" + name] = adata.obsm["X_umap"].copy()
    adata.uns.pop("umap")
    adata.obsm.pop("X_umap")


parser = argparse.ArgumentParser(description="Preprocessing, Harmony and UMAPs of one object")
parser.add_argument("--min-disp", help="minimum normalized dispersion of highly variable genes", type=float, default=0.2)
parser.add_argument("object", help="file name of the object in data/h5/agg/")
args = parser.parse_args()

prefix = "{}_md{}".format(os.path.basename(args.object).split(".h5ad")[0], args.min_disp)
log = open(str(data("h5", "agg", prefix + ".log")), "w")
adata = sc.read_h5ad(str(data("h5", "agg", args.object)))
log.write("min_disp = {}\n".format(args.min_disp))
log.write("{}: n_obs={}, n_var={}\n".format(prefix, adata.n_obs, adata.n_vars))

# Normalization and highly variable genes; immunoglobulin and sex genes are left out of the highly variable genes
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.01, max_mean=3, min_disp=args.min_disp)
adata.var.loc[adata.var_names.str.match("^IG[HKL]"), "highly_variable"] = False
adata.var.loc[adata.var_names.isin(["DDX3Y", "RPS4Y1", "XIST"]), "highly_variable"] = False
log.write("{}: n_hvg={}\n".format(prefix, adata.var["highly_variable"].sum()))

adata.raw = adata
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)

log.write("{}: PCA\n".format(prefix))
sc.tl.pca(adata, svd_solver="arpack")

# Harmony on the PCs, by library and by assay version
log.write("{}: Harmony\n".format(prefix))
sc.external.pp.harmony_integrate(adata, "gex_id",
                                 basis="X_pca", adjusted_basis="X_sid_harmony",
                                 max_iter_harmony=50)
if ("assay_type" not in adata.obs) and ("gex_id" in adata.obs):
    adata.obs["assay_type"] = ["GEX" + gid.split("_GEX_")[1] for gid in adata.obs["gex_id"]]
sc.external.pp.harmony_integrate(adata, "assay_type",
                                 basis="X_pca", adjusted_basis="X_gex_harmony",
                                 max_iter_harmony=50)

# Neighbor graphs with and without Harmony
log.write("{}: Neighbors\n".format(prefix))
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_pca", key_added="nb_pca")
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_sid_harmony", key_added="nb_sid_harmony")
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_gex_harmony", key_added="nb_gex_harmony")

log.write("{}: UMAP\n".format(prefix))
compute_umap(adata, "nb_pca")
compute_umap(adata, "nb_sid_harmony")
compute_umap(adata, "nb_gex_harmony")
adata.write(str(data("h5", "agg", prefix + "_umap.h5ad")))

log.write("{}: Done!".format(prefix))
log.close()
