#!/usr/bin/env python3
"""Leiden clusters of one preprocessed plasma cell object at one resolution, on the PCA graph and on the two Harmony
graphs (cell-type annotation).

Purpose:      Leiden clustering (scanpy defaults, random_state 0) at the resolution given on the command line, rounded
              to one decimal, on each of the three 31-nearest-neighbor graphs of the object: nb_pca, nb_sid_harmony and
              nb_gex_harmony (columns pca_leiden_<r>, hmy_sid_leiden_<r> and hmy_gex_leiden_<r>). The script restarts
              itself with PYTHONHASHSEED=0 when the variable is not set.
              Run once per object and resolution; resolutions 3.5 to 18 in steps of 0.5, e.g.
              for r in $(seq 3.5 0.5 18); do python3 02_preprocessing/04_leiden_clusters.py $r <object>_umap.h5ad; done

Inputs:       data/h5/agg/<object>.h5ad written by 03_preprocess_umap.py: graphs nb_pca, nb_sid_harmony, nb_gex_harmony.

Outputs:      data/h5/agg/<object>_leiden_<r>.csv: the obs table of the object with the three cluster columns.

Dependencies: Python 3.8 with requirements.txt (scanpy, anndata) and leidenalg, python-igraph; config.py.
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

parser = argparse.ArgumentParser(description="Leiden clustering at one resolution")
parser.add_argument("leiden_res", type=float, help="Leiden resolution")
parser.add_argument("object", help="file name of the object in data/h5/agg/")
args = parser.parse_args()

prefix = os.path.basename(args.object).split(".h5ad")[0]
adata = sc.read_h5ad(str(data("h5", "agg", args.object)))
r = round(args.leiden_res, 1)

print("leiden resolution = " + str(r))
sc.tl.leiden(adata, resolution=r, key_added="pca_leiden_" + str(r), neighbors_key="nb_pca")
sc.tl.leiden(adata, resolution=r, key_added="hmy_sid_leiden_" + str(r), neighbors_key="nb_sid_harmony")
sc.tl.leiden(adata, resolution=r, key_added="hmy_gex_leiden_" + str(r), neighbors_key="nb_gex_harmony")
adata.obs.to_csv(str(data("h5", "agg", prefix + "_leiden_" + str(r) + ".csv")))
