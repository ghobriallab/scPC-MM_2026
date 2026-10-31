#!/usr/bin/env python3
"""Combined AnnData objects of the PCs of the three external validation datasets (Ledergor et al. 2018, Boiarsky et al.
2022, Dang et al. 2023), for the stage-free signature test and the compact MGUS-MM classifier (Results and Methods,
MGUS-MM classifier).

Purpose:      Joins the counts of the three datasets on their shared genes, with the study in obs source and the
              technology in obs technology (MARS-seq, 10x-scRNA-seq). Writes the counts, then the log-normalized counts
              (counts per 10,000, log1p), then the scaled object: highly variable genes (min_mean 0.0125, max_mean 3,
              min_disp 0.5) without ribosomal (RP[SL]), mitochondrial (MT-) and immunoglobulin (IG[HKL]) genes, scaled
              (max_value 10), PCA (arpack), graph of 31 nearest neighbors on the PCA (nb_pca), Harmony integration of
              the PCA by study (at most 50 iterations) with its own graph (nb_harmony), and a UMAP of each graph. Seeds 42
              (Python random, numpy). Run with PYTHONHASHSEED=0.

Inputs:       data/h5/external/CA3Data_Ledergor2018/umi_count.h5ad (02_ledergor_natmedicine2018.py),
              data/h5/external/Boiarsky_NatCommun2022/umi_count.h5ad (01_boiarsky_natcommun2022.py),
              data/h5/external/Dang_CancerCell2023/umi_count.h5ad (03_dang_cancercell2023.py): UMI counts and obs
              sample_id, disease_status, annotation and the other columns of each dataset.

Outputs:      data/h5/external/Ledergor2018-Boiarsky2022-Dang2023_umi_count.h5ad: UMI counts of the three studies.
              data/h5/external/Ledergor2018-Boiarsky2022-Dang2023_log1p_count.h5ad: log-normalized counts.
              data/h5/external/Ledergor2018-Boiarsky2022-Dang2023_scaled_umap.h5ad: scaled highly variable genes, X_pca,
              X_harmony, neighbor graphs, X_umap_pca and X_umap_harmony; raw = log-normalized counts of all genes.

Dependencies: Python 3.8 with requirements.txt (numpy, anndata, scanpy) and harmonypy (scanpy.external); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

import random
import warnings

import anndata as ad
import numpy as np
import scanpy as sc

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)
EXT = data("h5", "external")
PREFIX = "Ledergor2018-Boiarsky2022-Dang2023"


def compute_umap(a, key):
    """UMAP of one neighbor graph, stored as X_umap_pca or X_umap_harmony"""
    sc.tl.umap(a, neighbors_key=key)
    a.uns["umap_" + key.lstrip("nb_")] = a.uns["umap"]
    a.obsm["X_umap_" + key.lstrip("nb_")] = a.obsm["X_umap"].copy()
    a.uns.pop("umap")
    a.obsm.pop("X_umap")


# ---- 1. counts of the three studies, joined on their shared genes -------------------------------------------------------
adatas = [
    sc.read(EXT / "CA3Data_Ledergor2018" / "umi_count.h5ad"),
    sc.read(EXT / "Boiarsky_NatCommun2022" / "umi_count.h5ad"),
    sc.read(EXT / "Dang_CancerCell2023" / "umi_count.h5ad"),
]
adatas[0].obs["source"] = "Ledergor_etal_NatMed2018"                    # obs technology (MARS-seq) is in the sample table
adatas[1].obs["technology"] = "10x-scRNA-seq"
adatas[1].obs["source"] = "Boiarsky_etal_NatCommun2022"
adatas[2].obs["technology"] = "10x-scRNA-seq"
adatas[2].obs["source"] = "Dang_etal_CancerCell2023"
adatas[2].obs["cluster"] = np.nan
adata = ad.concat(adatas)
adata.write(EXT / (PREFIX + "_umi_count.h5ad"))

# ---- 2. log-normalized counts -------------------------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata)
adata.write(EXT / (PREFIX + "_log1p_count.h5ad"))

# ---- 3. highly variable genes, scaling, PCA, Harmony by study, neighbor graphs and UMAPs --------------------------------
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.var.loc[adata.var_names.str.match("^RP[SL]"), "highly_variable"] = False
adata.var.loc[adata.var_names.str.match("^MT-"), "highly_variable"] = False
adata.var.loc[adata.var_names.str.match("^IG[HKL]"), "highly_variable"] = False

adata.raw = adata
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_pca", key_added="nb_pca")
sc.external.pp.harmony_integrate(adata,
                                 "source",
                                 basis="X_pca",
                                 adjusted_basis="X_harmony",
                                 max_iter_harmony=50)
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_harmony", key_added="nb_harmony")
compute_umap(adata, "nb_pca")
compute_umap(adata, "nb_harmony")
adata.write(EXT / (PREFIX + "_scaled_umap.h5ad"))
