#!/usr/bin/env python3
"""AnnData objects of the PCs of Boiarsky et al., Nat Commun 2022 (GSE193531): counts, log-normalized counts and embedding,
for the external validation of the malignancy and MST scores (Fig. 5C; Extended Data Figs. 7C and 8E).

Purpose:      Builds an AnnData object from the published UMI count matrix (genes x cells) and cell metadata (row i of the
              metadata = column i of the matrix), renames the metadata columns sample_ID, disease_stage and
              normal_or_neoplastic to sample_id, disease_status and annotation, and recodes normal bone marrow (NBM) as
              Donor, neoplastic cells as tumor and unlabeled cells as "?". Writes the counts, then the log-normalized
              counts (counts per 10,000, log1p), then the scaled object: highly variable genes (min_mean 0.0125, max_mean
              3, min_disp 0.5) without immunoglobulin (IG[HKL]), ribosomal (RP[SL]), mitochondrial (MT-) and sex-linked
              (DDX3Y, RPS4Y1, XIST) genes, scaled (max_value 10), PCA (arpack), graph of 31 nearest neighbors on the PCA
              (key nb_pca) and UMAP. Run with PYTHONHASHSEED=0.

Inputs:       data/h5/external/Boiarsky_NatCommun2022/GSE193531_umi-count-matrix.csv: UMI counts, genes x cells.
              data/h5/external/Boiarsky_NatCommun2022/GSE193531_cell-level-metadata.csv: cell metadata (index = cell),
              sample_ID, disease_stage, normal_or_neoplastic and the other published columns.

Outputs:      data/h5/external/Boiarsky_NatCommun2022/umi_count.h5ad: UMI counts and cell metadata.
              data/h5/external/Boiarsky_NatCommun2022/log1p_count.h5ad: log-normalized counts.
              data/h5/external/Boiarsky_NatCommun2022/scaled_umap.h5ad: scaled highly variable genes, X_pca, neighbor
              graph and X_umap; raw = log-normalized counts of all genes.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy, anndata, scanpy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import warnings

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

warnings.simplefilter(action="ignore", category=FutureWarning)

DIR = data("h5", "external", "Boiarsky_NatCommun2022")

# ---- 1. AnnData of the published counts and cell metadata ---------------------------------------------------------------
X = pd.read_csv(DIR / "GSE193531_umi-count-matrix.csv", index_col=0)                    # genes x cells
obs = pd.read_csv(DIR / "GSE193531_cell-level-metadata.csv", index_col=0)
a = ad.AnnData(csr_matrix(X.T))
a.obs = obs.copy()                                                                      # row i = column i of the counts
a.var_names = list(X.index)

# column names and labels shared with the other datasets
a.obs.rename(columns={"sample_ID": "sample_id",
                      "disease_stage": "disease_status",
                      "normal_or_neoplastic": "annotation"}, inplace=True)
a.obs["disease_status"] = list(a.obs["disease_status"])
a.obs.loc[a.obs["disease_status"] == "NBM", "disease_status"] = "Donor"
a.obs["annotation"] = list(a.obs["annotation"])
a.obs.loc[a.obs["annotation"] == "neoplastic", "annotation"] = "tumor"
a.obs.loc[a.obs["annotation"].isna(), "annotation"] = "?"
a.write(DIR / "umi_count.h5ad")

# ---- 2. log-normalized counts -------------------------------------------------------------------------------------------
sc.pp.normalize_total(a, target_sum=1e4, inplace=True)
sc.pp.log1p(a)
a.write(DIR / "log1p_count.h5ad")

# ---- 3. highly variable genes, scaling, PCA, neighbor graph and UMAP ----------------------------------------------------
a.raw = a
sc.pp.highly_variable_genes(a, min_mean=0.0125, max_mean=3, min_disp=0.5)
a.var.loc[a.var_names.str.match("^IG[HKL]"), "highly_variable"] = False
a.var.loc[a.var_names.str.match("^RP[SL]"), "highly_variable"] = False
a.var.loc[a.var_names.str.match("^MT-"), "highly_variable"] = False
a.var.loc[a.var_names.isin(["DDX3Y", "RPS4Y1", "XIST"]), "highly_variable"] = False

a = a[:, a.var.highly_variable]
sc.pp.scale(a, max_value=10)
sc.tl.pca(a, svd_solver="arpack")
sc.pp.neighbors(a, n_neighbors=31, use_rep="X_pca", key_added="nb_pca")
sc.tl.umap(a, neighbors_key="nb_pca")
a.write(DIR / "scaled_umap.h5ad")
