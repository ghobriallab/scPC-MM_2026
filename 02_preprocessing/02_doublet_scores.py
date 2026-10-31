#!/usr/bin/env python3
"""Doublet scores and calls of every cell of one library with scds, scDblFinder and Scrublet (cell QC).

Purpose:      Raw counts of one library, copied to the 'counts' layer, are passed to R through anndata2ri:
              scds cxds_bcds_hybrid (cxds, bcds and hybrid scores; set.seed(1001)); computeDoubletDensity on
              log-normalized counts (top 2,000 highly variable genes, 10 PCs with set.seed(1001), density with
              set.seed(1002)); findDoubletClusters on graph-based clusters of the PCs (cells of clusters with
              P > 0.05 are called doublets); scDblFinder with default settings. Scrublet in Python: expected doublet
              rate 0.06, min_counts 2, min_cells 3, genes above the 85th percentile of variability, 30 PCs, automatic
              threshold. A tool that fails leaves its columns empty (NaN). The script restarts itself with
              PYTHONHASHSEED=0 when the variable is not set.
              Run once per library, one process each, for every per-library object in data/h5/filtered/, e.g.
              for f in data/h5/filtered/*.h5ad; do python3 02_preprocessing/02_doublet_scores.py "$(basename "$f" .h5ad)"; done

Inputs:       data/h5/filtered/<library>.h5ad: raw counts of the cells of one library (library name on the command line).

Outputs:      data/h5/doublets/<library>.dbl.csv: one row per cell: dbl_cxds_score, dbl_bcds_score, dbl_hybrid_score,
              dbl_cDD_score, dbl_fDC_class, dbl_scDF_score, dbl_scDF_class, dbl_scrublet_score, dbl_scrublet_class.
              data/h5/doublets/<library>.scrublet.log: Scrublet threshold and doublet rates.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata) and scrublet, rpy2, anndata2ri; R with
              the Bioconductor packages scuttle, scran, scater, bluster, scDblFinder and scds; config.py.
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

import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
import anndata2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

anndata2ri.activate()           # AnnData objects passed to R arrive as SingleCellExperiment objects

R_SCDBLFINDER = """
function(sce) {
  library(scuttle)
  library(scran)
  library(scater)
  library(bluster)
  library(scDblFinder)

  # method 1: computeDoubletDensity
  sce = logNormCounts(sce)
  hvgs = getTopHVGs(modelGeneVar(sce), n=2000)
  set.seed(1001)
  sce = runPCA(sce, ncomponents=10, subset_row=hvgs)
  sce = runTSNE(sce, dimred="PCA")
  set.seed(1002)
  m1.score = computeDoubletDensity(sce, subset.row=hvgs)

  # method 2: findDoubletClusters on graph-based clusters of the PCs
  clusters = clusterRows(reducedDim(sce, "PCA"), NNGraphParam())
  tab = findDoubletClusters(sce, clusters)
  m2.class = ifelse(tab[as.character(clusters),]$p.value > 0.05, "doublet", "singlet")

  # method 3: scDblFinder
  sce = scDblFinder(sce)
  m3.score = sce$scDblFinder.score
  m3.class = as.character(sce$scDblFinder.class)

  df = data.frame(index=colnames(sce),
                  dbl_cDD_score=m1.score,
                  dbl_fDC_class=m2.class,
                  dbl_scDF_score=m3.score,
                  dbl_scDF_class=m3.class,
                  row.names=1)
  return(df)
}"""


def empty(adata, columns):
    """Columns of NaN for every cell, used when a tool fails."""
    return pd.DataFrame(data={c: [np.nan for i in adata.obs.index] for c in columns}, index=adata.obs.index)


def run_scdblfinder(adata):
    """computeDoubletDensity, findDoubletClusters and scDblFinder (R)."""
    r_scdblfinder = ro.r(R_SCDBLFINDER)
    try:
        return r_scdblfinder(adata)
    except Exception:
        return empty(adata, ["dbl_cDD_score", "dbl_fDC_class", "dbl_scDF_score", "dbl_scDF_class"])


def run_scds(adata):
    """scds cxds, bcds and hybrid doublet scores (R)."""
    try:
        scds = importr("scds")
        ro.r["set.seed"](1001)
        x = scds.cxds_bcds_hybrid(adata)
        x = x.obs[["cxds_score", "bcds_score", "hybrid_score"]]
        return x.rename(columns={"cxds_score": "dbl_cxds_score",
                                 "bcds_score": "dbl_bcds_score",
                                 "hybrid_score": "dbl_hybrid_score"})
    except Exception:
        return empty(adata, ["dbl_cxds_score", "dbl_bcds_score", "dbl_hybrid_score"])


def run_scrublet(adata, library, out_dir):
    """Scrublet doublet scores and calls; threshold and rates written to <library>.scrublet.log."""
    scrub = scr.Scrublet(adata.X, expected_doublet_rate=0.06)
    try:
        scores, flags = scrub.scrub_doublets(min_counts=2,
                                             min_cells=3,
                                             min_gene_variability_pctl=85,
                                             n_prin_comps=30)
        with open(os.path.join(out_dir, library + ".scrublet.log"), "w") as fo:
            fo.write("auto_threshold={}\n".format(scrub.threshold_))
            fo.write("detected_doublet_rate={}\n".format(scrub.detected_doublet_rate_))
            fo.write("detectable_doublet_fraction={}\n".format(scrub.detectable_doublet_fraction_))
            fo.write("overall_doublet_rate={}\n".format(scrub.overall_doublet_rate_))
        return pd.DataFrame(data={"dbl_scrublet_score": scores,
                                  "dbl_scrublet_class": np.where(flags == True, "doublet", "singlet")},
                            index=adata.obs.index)
    except Exception:
        return empty(adata, ["dbl_scrublet_score", "dbl_scrublet_class"])


def doublet_tools(adata, library, out_dir):
    """The three tools side by side, one row per cell barcode."""
    adata.layers["counts"] = adata.X.copy()
    scds_ = run_scds(adata)
    scdbl_ = run_scdblfinder(adata)
    scrub_ = run_scrublet(adata, library, out_dir)
    bc = scds_.index
    return pd.concat([scds_.loc[bc], scdbl_.loc[bc], scrub_.loc[bc]], axis=1)


parser = argparse.ArgumentParser(description="Doublet scores of one library")
parser.add_argument("library", help="library name: data/h5/filtered/<library>.h5ad")
args = parser.parse_args()

out_dir = data("h5", "doublets")
out_dir.mkdir(parents=True, exist_ok=True)
adata = sc.read_h5ad(str(data("h5", "filtered", args.library + ".h5ad")))
dbl = doublet_tools(adata, args.library, str(out_dir))
dbl.to_csv(out_dir / (args.library + ".dbl.csv"))
