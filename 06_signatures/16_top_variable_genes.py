#!/usr/bin/env python3
"""The 5,000 most highly variable genes of the 1,649,290 PCs, for the test of their overlap with the gene pool of
signature discovery (Methods, detection of single-cell transcriptional signatures).

Purpose:      scanpy highly_variable_genes (default flavor 'seurat', n_top_genes = 5000) on the log-normalized expression
              of all PCs of 5' libraries; writes the gene table with the highly_variable flag and the dispersion
              statistics. Must be run with PYTHONHASHSEED=0.

Inputs:       data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad: X = log-normalized expression (1,649,290 PCs x 39,390 genes).

Outputs:      data/paper/plasma-cell-top-5000hvg.csv: index = gene; highly_variable, means, dispersions,
              dispersions_norm.

Dependencies: Python 3.8 with requirements.txt (numpy, scanpy, anndata); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import scanpy as sc

np.random.seed(0)

adata = sc.read(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad"))
adata.uns["log1p"]["base"] = None          # log base read by highly_variable_genes: None = natural logarithm
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
adata.var.to_csv(data("paper", "plasma-cell-top-5000hvg.csv"))
