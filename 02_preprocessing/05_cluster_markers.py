#!/usr/bin/env python3
"""Marker genes of the Leiden clusters of one plasma cell object (cell-type annotation).

Purpose:      Takes the matrix in .raw of the object, scales the total counts of each cell to 10,000 and applies
              log1p, then ranks the genes of each cluster of hmy_sid_leiden_<r> (Leiden clusters on the library-
              corrected Harmony graph; r as written in the column name) against all other cells with the Wilcoxon
              rank-sum test (scanpy rank_genes_groups; results in uns hmy_sid_leiden_<r>_wilcoxon). The script
              restarts itself with PYTHONHASHSEED=0 when the variable is not set.
              Run once per object and resolution, e.g. python3 02_preprocessing/05_cluster_markers.py 4.0 <object>.h5ad

Inputs:       data/h5/agg/<object>.h5ad: .raw and the obs column hmy_sid_leiden_<r>.

Outputs:      data/h5/agg/<object>_de_<r>.h5ad: normalized, log-transformed object with the marker table in uns.

Dependencies: Python 3.8 with requirements.txt (scanpy, anndata); config.py.
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

parser = argparse.ArgumentParser(description="Marker genes of Leiden clusters")
parser.add_argument("resolution", help="resolution as written in the cluster column name, e.g. 4.0")
parser.add_argument("object", help="file name of the object in data/h5/agg/")
args = parser.parse_args()

prefix = os.path.basename(args.object).split(".h5ad")[0]
adata = sc.read_h5ad(str(data("h5", "agg", args.object)))
adata = adata.raw.to_adata()
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata)

r = args.resolution
sc.tl.rank_genes_groups(adata, "hmy_sid_leiden_" + r, method="wilcoxon", key_added="hmy_sid_leiden_" + r + "_wilcoxon")
adata.write(str(data("h5", "agg", prefix + "_de_" + r + ".h5ad")))
