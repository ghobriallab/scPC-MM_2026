#!/usr/bin/env python3
"""Aggregated plasma cell objects: counts of the cohort, annotated (malignant and normal) PCs, log-normalized
expression, UMAP of all PCs and UMAP coordinate tables (Methods, scRNA-seq data normalization and clustering).

Purpose:      Starts from the aggregated counts of the PCs passing QC (200-5,000 genes, 400-50,000 counts, < 15%
              mitochondrial counts). redcap_id of each cell = redcap_id of its library in the cohort table; the serial,
              batch and source patient ID columns are dropped; libraries flagged as a potential sample swap or with
              another diagnosis (disease_status Other) are removed. The cell annotation selects the annotated PCs and
              splits them into malignant (tumor) and normal objects. All PCs: total counts scaled to 10,000 per cell
              and log1p; highly variable genes (min_mean 0.0125, max_mean 3, min_disp 0.2) without immunoglobulin
              (^IG[HKL]), ribosomal (^RP[SL]), mitochondrial (^MT-) and sex genes (DDX3Y, RPS4Y1, XIST); scaling
              (max_value 10), PCA (arpack), 31-nearest-neighbor graph (nb_pca) and UMAP. UMAP coordinates are saved for
              four UMAP objects; the log-normalized expression of the annotated PCs is saved alone and as .raw of their
              UMAP object. Run with PYTHONHASHSEED=0.

Inputs:       data/h5/agg/plasma-cell-jan2024_counts.h5ad: counts; obs gex_id, batch, serial_id, serial_barcode and the
              source patient ID column (*_PANGEA_ID).
              data/metadata/tumor-cohort-metadata-all-libraries.csv (gex_id index): PotentialSampleSwap, disease_status,
              redcap_id. data/annotation/plasma-cell-annotation-final-240327.csv (cell index): annotation and the other
              per-cell columns. data/h5/agg/annotated-cell-umap-240327.h5ad, tumor-cell-umap-240327.h5ad and
              normal-cell-umap-240327.h5ad: UMAP objects of the annotated, malignant and normal PCs (obsm X_umap).

Outputs:      data/metadata/tumor-gex-added_240227.tsv: gex_id of the libraries of the batch labeled 'new tumors'.
              data/h5/agg/plasma-cell-counts-240325.h5ad, plasma-annotated-cell-counts-240327.h5ad,
              tumor-cell-counts-240327.h5ad, normal-cell-counts-240327.h5ad: counts.
              data/h5/agg/plasma-cell-log1p-240325.h5ad; var-dispersion-hvg-240325.csv (HVG_original, HVG_final);
              plasma-cell-umap-240325.h5ad.
              data/annotation/{normal-cell,tumor-cell,annotated-cell}-umap-coords-240329.csv and
              plasma-cell-ALL-umap-coords-240329.csv: umap1, umap2 per cell.
              data/h5/agg/annotated-cell-log1p-240327.h5ad and annotated-cell-full-240327.h5ad.

Dependencies: Python 3.8 with requirements.txt (pandas, scanpy, anndata); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import pandas as pd
import scanpy as sc


def agg(name):
    """An aggregated object or table in data/h5/agg/ (as a string path)."""
    return str(data("h5", "agg", name))


# ---- 1. Counts of all PCs of the cohort --------------------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-all-libraries.csv"), index_col=0)
# Libraries left out: potential sample swaps and diagnoses other than MGUS, SMM, MM or healthy donor
remove_gex_id = list(meta.loc[(meta["PotentialSampleSwap"] == "Swap") | (meta["disease_status"] == "Other")].index)

adata = sc.read_h5ad(agg("plasma-cell-jan2024_counts.h5ad"))
# Libraries of the batch labeled 'new tumors'
adata.obs.loc[adata.obs["batch"] == "new tumors", ["gex_id"]].drop_duplicates().to_csv(
    data("metadata", "tumor-gex-added_240227.tsv"), index=False)

# Patient ID of each cell = redcap_id (Final_PANGEA_ID) of its library in the cohort table; the serial, batch and
# source patient ID (*_PANGEA_ID) columns are dropped
adata.obs["redcap_id"] = list(adata.obs["gex_id"].astype(str).map(meta["redcap_id"]))
drop = ["serial_id", "serial_barcode", "batch"] + [c for c in adata.obs.columns if c.endswith("_PANGEA_ID")]
adata.obs.drop(columns=drop, inplace=True)

adata = adata[~adata.obs["gex_id"].isin(remove_gex_id)]
adata.obs["gex_id"] = list(adata.obs["gex_id"])
adata.write(agg("plasma-cell-counts-240325.h5ad"))

# ---- 2. Annotated, malignant and normal PCs (counts) -------------------------------------------------------------
adata = sc.read(agg("plasma-cell-counts-240325.h5ad"))
annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
col_to_join = list(set(annot.columns).difference(set(adata.obs.columns)))
adata.obs = adata.obs.join(annot[col_to_join])
adata = adata[~adata.obs["annotation"].isna()]
tumor = adata[adata.obs["annotation"] == "tumor"].copy()
normal = adata[adata.obs["annotation"] == "normal"].copy()

# The annotation columns only select the cells; they are not stored in the objects
adata.obs.drop(columns=col_to_join, inplace=True)
adata.write(agg("plasma-annotated-cell-counts-240327.h5ad"))
tumor.obs.drop(columns=col_to_join, inplace=True)
tumor.write(agg("tumor-cell-counts-240327.h5ad"))
normal.obs.drop(columns=col_to_join, inplace=True)
normal.write(agg("normal-cell-counts-240327.h5ad"))

# ---- 3. Log-normalized expression, highly variable genes and UMAP of all PCs -------------------------------------
adata = sc.read_h5ad(agg("plasma-cell-counts-240325.h5ad"))
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata)
adata.write(agg("plasma-cell-log1p-240325.h5ad"))

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.2)
var = adata.var.copy()
var["HVG_original"] = list(var["highly_variable"])
# Immunoglobulin, ribosomal, mitochondrial and sex genes are removed from the highly variable genes
adata.var.loc[adata.var_names.str.match("^IG[HKL]"), "highly_variable"] = False
adata.var.loc[adata.var_names.str.match("^RP[SL]"), "highly_variable"] = False
adata.var.loc[adata.var_names.str.match("^MT-"), "highly_variable"] = False
adata.var.loc[adata.var_names.isin(["DDX3Y", "RPS4Y1", "XIST"]), "highly_variable"] = False
var["HVG_final"] = list(adata.var["highly_variable"])
var.to_csv(agg("var-dispersion-hvg-240325.csv"))

adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=31, use_rep="X_pca", key_added="nb_pca")
sc.tl.umap(adata, neighbors_key="nb_pca")
adata.write(agg("plasma-cell-umap-240325.h5ad"))

# ---- 4. UMAP coordinates -----------------------------------------------------------------------------------------
umap_tables = [("normal-cell-umap-240327.h5ad", "normal-cell-umap-coords-240329.csv"),
               ("tumor-cell-umap-240327.h5ad", "tumor-cell-umap-coords-240329.csv"),
               ("annotated-cell-umap-240327.h5ad", "annotated-cell-umap-coords-240329.csv"),
               ("plasma-cell-umap-240325.h5ad", "plasma-cell-ALL-umap-coords-240329.csv")]
data("annotation").mkdir(parents=True, exist_ok=True)
for name, table in umap_tables:
    obj = sc.read_h5ad(agg(name), backed="r")
    df = sc.get.obs_df(obj, obsm_keys=[("X_umap", 0), ("X_umap", 1)])
    df.rename(columns={"X_umap-0": "umap1", "X_umap-1": "umap2"}, inplace=True)
    df.to_csv(data("annotation", table))

# ---- 5. Log-normalized expression of the annotated PCs -----------------------------------------------------------
cells = list(sc.read_h5ad(agg("annotated-cell-umap-240327.h5ad")).obs.index)
adata = sc.read_h5ad(agg("plasma-cell-log1p-240325.h5ad"))
adata = adata[cells]
adata.write(agg("annotated-cell-log1p-240327.h5ad"))

# UMAP object of the annotated PCs with their log-normalized expression of all genes in .raw
adata = sc.read_h5ad(agg("annotated-cell-umap-240327.h5ad"))
adata.raw = sc.read_h5ad(agg("annotated-cell-log1p-240327.h5ad"))
adata.write(agg("annotated-cell-full-240327.h5ad"))
