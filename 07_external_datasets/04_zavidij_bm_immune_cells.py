#!/usr/bin/env python3
"""AnnData objects of the bone marrow immune cells of Zavidij et al., Nat Cancer 2020 (GSE124310), for the signature
activity in bone marrow immune cells across disease stages (Methods, signature activity in other bone marrow cell types).

Purpose:      Reads the 10x count matrix of every sample (one folder per sample; cell names prefixed with the sample),
              joins the authors' cell-type labels (barcode suffix set to -1) and keeps the labeled cells. Writes the
              counts, then the log-normalized counts (counts per 10,000, log1p), then, from the log-normalized file, the
              scaled object: highly variable genes (min_mean 0.0125, max_mean 3, min_disp 0.2) without immunoglobulin
              (IG[HKL]), ribosomal (RP[SL]), mitochondrial (MT-) and sex-linked (DDX3Y, RPS4Y1, XIST) genes, scaled
              (max_value 10), PCA (arpack), Harmony integration of the PCA by sample (at most 50 iterations), graphs of 31
              nearest neighbors on the PCA (nb_pca) and on the Harmony embedding (nb_harmony), and a UMAP of each graph.
              Run with PYTHONHASHSEED=0.

Inputs:       data/h5/external/Zavidij_NatCancer2020/<sample>/: 10x count matrix of each sample (every sub-folder).
              data/h5/external/Zavidij_NatCancer2020/Zavidij_NatCancer2020_CellAnnotation.csv: Sample, Cell barcode,
              Cell type (the authors' labels).

Outputs:      data/h5/external/Zavidij_NatCancer2020/umi_count.h5ad: UMI counts; obs sample_id, cell_type.
              data/h5/external/Zavidij_NatCancer2020/log1p_count.h5ad: log-normalized counts.
              data/h5/external/Zavidij_NatCancer2020/scaled_umap.h5ad: scaled highly variable genes, X_pca, X_harmony,
              neighbor graphs, X_umap_pca and X_umap_harmony; raw = log-normalized counts of all genes.

Dependencies: Python 3.8 with requirements.txt (pandas, anndata, scanpy) and harmonypy (scanpy.external); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import warnings

import anndata as ad
import pandas as pd
import scanpy as sc

warnings.simplefilter(action="ignore", category=FutureWarning)

DIR = data("h5", "external", "Zavidij_NatCancer2020")


def compute_umap(adata, key):
    """UMAP of one neighbor graph, stored as X_umap_pca or X_umap_harmony"""
    sc.tl.umap(adata, neighbors_key=key)
    adata.uns["umap_" + key.lstrip("nb_")] = adata.uns["umap"]
    adata.obsm["X_umap_" + key.lstrip("nb_")] = adata.obsm["X_umap"].copy()
    adata.uns.pop("umap")
    adata.obsm.pop("X_umap")


# ---- 1. counts of every sample ------------------------------------------------------------------------------------------
samples = sorted(d for d in os.listdir(DIR) if (DIR / d).is_dir())
adata = []
for sample in samples:
    a = sc.read_10x_mtx(DIR / sample)
    a.obs["sample_id"] = sample
    a.obs.index = [sample + "_" + idx for idx in a.obs.index]
    adata.append(a)
adata = ad.concat(adata)

# ---- 2. the authors' cell-type labels; unlabeled cells are left out ----------------------------------------------------
cell_type = pd.read_csv(DIR / "Zavidij_NatCancer2020_CellAnnotation.csv")
cell_type.index = cell_type["Sample"] + "_" + cell_type["Cell barcode"]
cell_type.index = ["-".join(x.split("-")[:-1]) + "-1" for x in cell_type.index]
cell_type = cell_type[["Cell type"]].rename(columns={"Cell type": "cell_type"})
adata.obs = adata.obs.join(cell_type)
adata = adata[~adata.obs["cell_type"].isna()]
adata.write(DIR / "umi_count.h5ad")

# ---- 3. log-normalized counts -------------------------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata)
adata.write(DIR / "log1p_count.h5ad")

# ---- 4. from the log-normalized file: highly variable genes, scaling, PCA, Harmony, neighbor graphs and UMAPs -----------
adata = sc.read(DIR / "log1p_count.h5ad")
adata.uns["log1p"]["base"] = None                                                       # natural log, not stored in the file
adata.raw = adata
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.2)
adata.var.loc[adata.var_names.str.match("^IG[HKL]"), "highly_variable"] = False
adata.var.loc[adata.var_names.str.match("^RP[SL]"), "highly_variable"] = False
adata.var.loc[adata.var_names.str.match("^MT-"), "highly_variable"] = False
adata.var.loc[adata.var_names.isin(["DDX3Y", "RPS4Y1", "XIST"]), "highly_variable"] = False

adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.external.pp.harmony_integrate(adata,
                                 "sample_id",
                                 basis="X_pca",
                                 adjusted_basis="X_harmony",
                                 max_iter_harmony=50)
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_pca", key_added="nb_pca")
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_harmony", key_added="nb_harmony")
compute_umap(adata, key="nb_pca")
compute_umap(adata, key="nb_harmony")
adata.write(DIR / "scaled_umap.h5ad")
