#!/usr/bin/env python3
"""AnnData objects of the PCs of Ledergor et al., Nat Med 2018 (GSE117156): counts, log-normalized counts and embedding,
for the external validation of the malignancy and MST scores (Fig. 5C; Extended Data Figs. 7C and 8E).

Purpose:      Reads the published MARS-seq counts (10x matrix format) with the cell table (joined by cell) and the sample
              table (joined by sample). Removes the patients with AL amyloidosis (samples AL*), sets the disease stage from
              the sample name (hip: Donor, MGUS, SMM, MM), keeps the cells that passed the authors' quality control (cell
              type given) and were assigned to a cluster, and labels the cells of the authors' clusters 1 and 2 normal and
              all others tumor (annotation). Writes the counts, then the log-normalized counts (counts per 10,000, log1p),
              then the scaled object: highly variable genes (min_mean 0.0125, max_mean 3, min_disp 0.5) without
              immunoglobulin (IG[HKL]), ribosomal (RP[SL]), mitochondrial (MT-) and sex-linked (DDX3Y, RPS4Y1, XIST)
              genes, scaled (max_value 10), PCA (arpack), graph of 31 nearest neighbors on the PCA (key nb_pca) and
              UMAP. Run with PYTHONHASHSEED=0.

Inputs:       data/h5/external/CA3Data_Ledergor2018/custom_feature_bc_matrix/: UMI counts in 10x matrix format.
              data/h5/external/CA3Data_Ledergor2018/Cells.csv: per cell (index), sample, cell_type, cluster and the other
              published columns. data/h5/external/CA3Data_Ledergor2018/Meta-data.csv: per sample (index), cancer_type,
              technology and the other published columns.

Outputs:      data/h5/external/CA3Data_Ledergor2018/umi_count.h5ad: UMI counts and cell metadata.
              data/h5/external/CA3Data_Ledergor2018/log1p_count.h5ad: log-normalized counts.
              data/h5/external/CA3Data_Ledergor2018/scaled_umap.h5ad: scaled highly variable genes, X_pca, neighbor graph
              and X_umap; raw = log-normalized counts of all genes.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, anndata, scanpy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import warnings

import numpy as np
import pandas as pd
import scanpy as sc

warnings.simplefilter(action="ignore", category=FutureWarning)

DIR = data("h5", "external", "CA3Data_Ledergor2018")

# ---- 1. AnnData of the published counts with the cell and sample tables -------------------------------------------------
a = sc.read_10x_mtx(DIR / "custom_feature_bc_matrix")
meta = pd.read_csv(DIR / "Meta-data.csv", index_col=0)
a.var.drop(columns=["gene_ids", "feature_types"], inplace=True)
a.obs = a.obs.join(pd.read_csv(DIR / "Cells.csv", index_col=0))
a.obs = a.obs.join(meta[meta.columns.difference(a.obs.columns)], on="sample", how="left")
a.obs.drop(columns=["umap1", "umap2", "g1s_score", "g2m_score", "cell_cycle_phase", "mp_top_score"], inplace=True)

# patients with AL amyloidosis are left out
a = a[~a.obs["sample"].str.startswith("AL")]

# disease stage from the sample name
a.obs["disease_status"] = np.nan
a.obs.loc[a.obs["sample"].str.startswith("hip"), "disease_status"] = "Donor"
a.obs.loc[a.obs["sample"].str.startswith("MGUS"), "disease_status"] = "MGUS"
a.obs.loc[a.obs["sample"].str.startswith("SMM"), "disease_status"] = "SMM"
a.obs.loc[a.obs["sample"].str.startswith("MM"), "disease_status"] = "MM"
a.obs.drop(columns=["cancer_type"], inplace=True)
a.obs.rename(columns={"sample": "sample_id",
                      "cell_type": "orig.celltype"}, inplace=True)

# cells that passed the authors' quality control and were assigned to a cluster
a = a[~a.obs["orig.celltype"].isna()]
a = a[a.obs["cluster"] != "unassigned"]

# the authors' clusters 1 and 2 are normal PCs, all other clusters malignant PCs
a.obs["annotation"] = "tumor"
a.obs.loc[a.obs["cluster"].isin(["1", "2"]), "annotation"] = "normal"
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
