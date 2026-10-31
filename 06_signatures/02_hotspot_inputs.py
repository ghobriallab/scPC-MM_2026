#!/usr/bin/env python3
"""The two Hotspot input objects of signature discovery, restricted to the 4,980-gene pool: the downsampled BM and PB PCs,
and the downsampled BM PCs (Methods, detection of single-cell transcriptional signatures).

Purpose:      The gene pool is the union of three DEG lists: malignant PCs (vs normal PCs by stage and between stages, BM;
              2,657 genes), normal PCs (between stages, vs donors, BM vs PB; 3,744 genes) and malignant PCs between paired
              BM and PB samples (605 genes), 4,980 genes in total. Cells are those of the downsampled UMI tables of
              01_downsample_cells.py (BM and PB: 340,816 cells; BM: 257,799 cells). QC metrics are computed on all genes
              first, so obs total_counts is the number of UMIs per cell over the whole transcriptome (the size factor of
              the Hotspot model). For each input: raw counts of the pool genes kept in layers['counts'], normalize_total
              (1e4) and log1p, highly variable gene statistics (min_mean 0.0125, max_mean 3, min_disp 0.2; the flag itself
              is dropped), raw = log-normalized values, scale (max_value 10) and PCA (arpack; X_pca is the latent space of
              the Hotspot kNN graph). Must be run with PYTHONHASHSEED=0.

Inputs:       data/h5/agg/annotated-cell-counts-240327.h5ad: raw UMI counts of all PCs (cells x 39,390 genes).
              data/signature/gene_list/tumor_de/bm_tumor_unionDE_2657genes.txt, tumor_de/bm_vs_ctc_DE_605genes.txt and
              normal_de/npc_core_bm_normal_unionDE_3744genes.txt: one gene per line.
              data/signature/umi/raw/BM-PB_cells.parquet and BM_cells.parquet (01_downsample_cells.py): only their columns
              (the cell barcodes) are used.

Outputs:      data/signature/hotspot/mm-BM-PB-4980-DEs/mm-BM-PB-4980-DEs_hotspot_input.h5ad: BM and PB cells.
              data/signature/hotspot/mm-BM-4980-full-DEs_hotspot_input.h5ad: BM cells.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata, pyarrow); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)
GL = data("signature", "gene_list")
HS = data("signature", "hotspot")


def load_genes(input_file):
    return [g.rstrip() for g in open(input_file)]


def cell_barcodes(name):
    """cell barcodes (columns) of a downsampled UMI table"""
    df = pd.read_parquet(data("signature", "umi", "raw", name),
                         thrift_container_size_limit=500000000,
                         thrift_string_size_limit=500000000)
    return df.columns


def hotspot_input(a, genes, output_file):
    """counts layer, log-normalized raw, scaled X and PCA of the cells of a, restricted to genes"""
    a = a[:, a.var.index.isin(genes)].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4, inplace=True)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, min_mean=0.0125, max_mean=3, min_disp=0.2)
    a.var.drop(columns=["highly_variable"], inplace=True)
    a.raw = a
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, svd_solver="arpack")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    a.write(output_file)


# ---- all PCs with QC metrics over the whole transcriptome ---------------------------------------------------------------
adata = sc.read(data("h5", "agg", "annotated-cell-counts-240327.h5ad"))
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

# ---- the 4,980-gene pool ------------------------------------------------------------------------------------------------
tumor_de = GL / "tumor_de" / "bm_tumor_unionDE_2657genes.txt"
ctc_de = GL / "tumor_de" / "bm_vs_ctc_DE_605genes.txt"
normal_de = GL / "normal_de" / "npc_core_bm_normal_unionDE_3744genes.txt"
union_genes = list(set(load_genes(tumor_de) + load_genes(normal_de) + load_genes(ctc_de)))

# ---- BM and PB cells, then BM cells ---------------------------------------------------------------------------------------
hotspot_input(adata[cell_barcodes("BM-PB_cells.parquet")], union_genes,
              HS / "mm-BM-PB-4980-DEs" / "mm-BM-PB-4980-DEs_hotspot_input.h5ad")
hotspot_input(adata[cell_barcodes("BM_cells.parquet")], union_genes,
              HS / "mm-BM-4980-full-DEs_hotspot_input.h5ad")
