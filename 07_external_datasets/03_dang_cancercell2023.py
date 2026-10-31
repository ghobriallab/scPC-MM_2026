#!/usr/bin/env python3
"""AnnData objects of the PCs of Dang et al., Cancer Cell 2023 (EGAS00001006694): counts, log-normalized counts and
embedding, for the external validation of the malignancy and MST scores (Fig. 5C; Supplementary Figs. 13C, 14E).

Purpose:      Writes the cell-barcode list of every count file (Cell Ranger filtered matrices, one per sample). For each
              sample of the authors' PC annotation, loads the count file whose cell barcodes match it (the file of the same
              name, except for the four samples mapped in the code), names cells <barcode>-<sample>, sets the disease stage
              from the sample name (MGUS, SMM, NDMM: MM, nBM: Donor) and makes gene names unique; samples are joined on
              their shared genes. Keeps the annotated cells, with the authors' sample and label (Malignant_PC: tumor,
              Normal_PC: normal). Writes the counts, then the
              log-normalized counts (counts per 10,000, log1p), then the scaled object: highly variable genes (min_mean
              0.0125, max_mean 3, min_disp 0.5) without immunoglobulin (IG[HKL]), ribosomal (RP[SL]), mitochondrial (MT-)
              and sex-linked (DDX3Y, RPS4Y1, XIST) genes, scaled (max_value 10), PCA (arpack), graph of 31 nearest
              neighbors on the PCA (key nb_pca) and UMAP. Run with PYTHONHASHSEED=0.

Inputs:       data/h5/external/Dang_CancerCell2023/reprocessed_h5/<sample>.filtered_feature_bc_matrix.h5: UMI counts.
              data/h5/external/Dang_CancerCell2023/PC.annotation.txt: Barcode, SampleID, CellType (the authors' PCs).

Outputs:      data/h5/external/Dang_CancerCell2023/barcode/<sample>.barcode.csv: cell barcodes of each count file.
              data/h5/external/Dang_CancerCell2023/umi_count.h5ad: UMI counts; obs patient, disease_status, sample_id,
              annotation.
              data/h5/external/Dang_CancerCell2023/log1p_count.h5ad: log-normalized counts.
              data/h5/external/Dang_CancerCell2023/scaled_umap.h5ad: scaled highly variable genes, X_pca, neighbor graph
              and X_umap; raw = log-normalized counts of all genes.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, anndata, scanpy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import glob
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.simplefilter(action="ignore", category=FutureWarning)

DIR = data("h5", "external", "Dang_CancerCell2023")

# ---- 1. cell-barcode list of every count file ---------------------------------------------------------------------------
(DIR / "barcode").mkdir(exist_ok=True)
for in_file in sorted(glob.glob(str(DIR / "reprocessed_h5" / "*.h5"))):
    ai = sc.read_10x_h5(in_file)
    file_base = os.path.basename(in_file).split(".")[0]
    with open(DIR / "barcode" / (file_base + ".barcode.csv"), "w") as fo:
        for bc in ai.obs_names:
            fo.write(bc + "\n")

# ---- 2. counts of each annotated sample ---------------------------------------------------------------------------------
annotation = pd.read_csv(DIR / "PC.annotation.txt", sep="\t")
# annotated sample -> count file whose barcodes match it
sample_swap = {"SMM15": "MGUS02s",
               "MGUS02s": "SMM02s",
               "SMM02s": "SMM05s",
               "SMM05s": "MGUS18"}
path = str(DIR / "reprocessed_h5" / "{}.filtered_feature_bc_matrix.h5")

parts = []
for sample_id in sorted(annotation["SampleID"].unique()):
    load_id = sample_id
    print("processing: " + sample_id)
    if sample_id in sample_swap.keys():
        load_id = sample_swap[sample_id]
        print("  count file: " + load_id)
    ai = sc.read_10x_h5(path.format(load_id))

    # disease stage from the sample name
    disease_status = np.nan
    if sample_id.startswith("SMM"):
        disease_status = "SMM"
    elif sample_id.startswith("MGUS"):
        disease_status = "MGUS"
    elif sample_id.startswith("NDMM"):
        disease_status = "MM"
    elif sample_id.startswith("nBM"):
        disease_status = "Donor"

    ai.obs["patient"] = sample_id
    ai.obs["disease_status"] = disease_status
    ai.obs.index = np.array(ai.obs.index) + "-" + sample_id
    ai.var_names_make_unique()
    parts.append(ai.copy())
a = ad.concat(parts)

# ---- 3. the authors' PCs, with their sample and malignant or normal label -----------------------------------------------
annotation["index"] = annotation["Barcode"] + "-" + annotation["SampleID"]
annotation = annotation.set_index("index", drop=True)
a = a[a.obs.index.isin(annotation.index)]
annotation.rename(columns={"SampleID": "sample_id",
                           "CellType": "annotation"}, inplace=True)
annotation.loc[annotation["annotation"] == "Malignant_PC", "annotation"] = "tumor"
annotation.loc[annotation["annotation"] == "Normal_PC", "annotation"] = "normal"
a.obs = a.obs.join(annotation[["sample_id", "annotation"]])
a.write(DIR / "umi_count.h5ad")

# ---- 4. log-normalized counts -------------------------------------------------------------------------------------------
sc.pp.normalize_total(a, target_sum=1e4, inplace=True)
sc.pp.log1p(a)
a.write(DIR / "log1p_count.h5ad")

# ---- 5. highly variable genes, scaling, PCA, neighbor graph and UMAP ----------------------------------------------------
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
