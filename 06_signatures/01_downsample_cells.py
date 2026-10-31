#!/usr/bin/env python3
"""PCs used for signature discovery: at most 1,000 malignant and 1,000 normal PCs per individual and tissue, with their
raw UMI tables and expression objects (Methods, detection of single-cell transcriptional signatures).

Purpose:      Cells of 5' libraries (assay_type GEX5) only, in six groups: patients' BM malignant, BM normal, PB malignant
              and PB normal PCs, and healthy donors' BM and PB PCs (disease_status HD). Within each group, an individual
              (redcap_id) with more than 1,000 cells keeps 1,000 drawn at random (Python random module, seed 444; the
              groups are drawn in the order listed and the individuals in the order of their cell counts); the others keep
              all their cells. Writes the six barcode lists, the raw UMI counts of the selected cells as genes x cells
              tables (BM cells; BM and PB cells; BM malignant cells; normal PCs) and the matching subsets of the annotated
              expression objects.

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv: index = gex_id; disease_status.
              data/h5/agg/annotated-cell-counts-240327.h5ad: raw UMI counts of all PCs; obs gex_id, redcap_id,
              sample_type (BM/PB), assay_type.
              data/annotation/plasma-cell-annotation-final-240327.csv: one row per cell; annotation (tumor/normal) and the
              other cell annotations, added to obs.
              data/h5/agg/annotated-cell-full-240327.h5ad and data/h5/agg/normal-cell-umap-240327.h5ad: objects subset for
              the signature inputs.

Outputs:      data/barcode/downsample/{BM,PB}_{tumor,normal,donor}_1000cell.txt: selected cell barcodes, one per line.
              data/signature/umi/raw/BM_cells.parquet, BM-PB_cells.parquet, BM_tumor_cells.parquet, BM-PB_npcs.parquet:
              raw UMI counts, genes x cells.
              data/h5/sig/BM_cells.h5ad, BM_tumor_cells.h5ad, BM-PB_cells.h5ad (subsets of the full object) and
              BM-PB_npcs.h5ad (subset of the normal-PC object).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata, pyarrow); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import gc
import random

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)
MAX_CELLS = 1000
BARCODES = data("barcode", "downsample")
UMI = data("signature", "umi", "raw")
SIG = data("h5", "sig")
for d in (BARCODES, UMI, SIG):
    os.makedirs(d, exist_ok=True)


def downsample_cell(obs, unit="group", max_cell=3000):
    """barcodes of obs, keeping at most max_cell cells (drawn at random) per value of unit"""
    index = []
    for key, n_cell in dict(obs[unit].value_counts()).items():
        cell_index = list(obs[obs[unit] == key].index)
        if n_cell > max_cell:
            cell_index = random.sample(cell_index, max_cell)
        index.extend(cell_index)
    return index


def write_list(x, output_file):
    with open(output_file, "w") as f:
        for i in x:
            f.write(i + "\n")


def get_umi_counts(a, index, output_file):
    """raw counts of the cells in index, written as a genes x cells table"""
    genes = list(a.var_names)
    X = pd.DataFrame(a[index, genes].X.todense())
    X.columns = genes
    X.index = index
    print("writing " + str(output_file))
    X.T.to_parquet(output_file)


# ---- cell annotation and disease stage ----------------------------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
adata = sc.read(data("h5", "agg", "annotated-cell-counts-240327.h5ad"))
annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
attach = list(set(annot.columns).difference(set(adata.obs.columns)))
adata.obs = adata.obs.join(annot[attach])
adata.obs["disease_status"] = list(meta.loc[adata.obs["gex_id"], "disease_status"])

# ---- downsampling: at most 1,000 cells per individual in each group --------------------------------------------------------
obs = adata.obs
patient = (obs["disease_status"] != "HD") & (obs["assay_type"] == "GEX5")
donor = (obs["disease_status"] == "HD") & (obs["assay_type"] == "GEX5")
groups = [("BM_tumor", patient & (obs["sample_type"] == "BM") & (obs["annotation"] == "tumor")),
          ("BM_normal", patient & (obs["sample_type"] == "BM") & (obs["annotation"] == "normal")),
          ("PB_tumor", patient & (obs["sample_type"] == "PB") & (obs["annotation"] == "tumor")),
          ("PB_normal", patient & (obs["sample_type"] == "PB") & (obs["annotation"] == "normal")),
          ("BM_donor", donor & (obs["sample_type"] == "BM")),
          ("PB_donor", donor & (obs["sample_type"] == "PB"))]
random.seed(444)
cells = {}
for name, criteria in groups:
    cells[name] = downsample_cell(obs[criteria], unit="redcap_id", max_cell=MAX_CELLS)
    write_list(cells[name], BARCODES / (name + "_1000cell.txt"))
bm_tumor, bm_normal, pb_tumor, pb_normal = cells["BM_tumor"], cells["BM_normal"], cells["PB_tumor"], cells["PB_normal"]
bm_donor, pb_donor = cells["BM_donor"], cells["PB_donor"]

# ---- raw UMI tables of the selected cells -----------------------------------------------------------------------------------
adata = adata[bm_donor + bm_tumor + bm_normal + pb_donor + pb_tumor + pb_normal]
gc.collect()
get_umi_counts(adata, bm_tumor + bm_normal + bm_donor, UMI / "BM_cells.parquet")
gc.collect()
get_umi_counts(adata, bm_tumor + bm_normal + bm_donor + pb_tumor + pb_normal + pb_donor, UMI / "BM-PB_cells.parquet")
gc.collect()
get_umi_counts(adata, bm_tumor, UMI / "BM_tumor_cells.parquet")
gc.collect()
get_umi_counts(adata, bm_normal + bm_donor + pb_normal + pb_donor, UMI / "BM-PB_npcs.parquet")
del adata
gc.collect()

# ---- subsets of the annotated objects -------------------------------------------------------------------------------------
full = sc.read(data("h5", "agg", "annotated-cell-full-240327.h5ad"))
full[bm_tumor + bm_normal + bm_donor].write(SIG / "BM_cells.h5ad")
full[bm_tumor].write(SIG / "BM_tumor_cells.h5ad")
full[bm_tumor + bm_normal + bm_donor + pb_tumor + pb_normal + pb_donor].write(SIG / "BM-PB_cells.h5ad")
del full
gc.collect()
npcs = sc.read(data("h5", "agg", "normal-cell-umap-240327.h5ad"))
npcs[bm_normal + bm_donor + pb_normal + pb_donor].write(SIG / "BM-PB_npcs.h5ad")
