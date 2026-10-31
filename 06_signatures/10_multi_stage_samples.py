#!/usr/bin/env python3
"""Raw counts of the PCs of the individuals sampled at two disease stages, the input of the longitudinal UMAP (Supplementary Figs. 9A, 14C).

Purpose:      Libraries of 5' assays (assay_type GEX5) with a disease stage (Other excluded); SMM risk group from
              SMM_stage_lenient; HD renamed Donor. The individuals kept are those whose BM samples carry more than one
              disease stage (three SMM patients sampled at two risk levels); all their PCs (BM and PB, every visit) are
              written as a raw-count object. Must be run with PYTHONHASHSEED=0.

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv: assay_type, disease_status, SMM_stage_lenient,
              sample_type, redcap_id.
              data/h5/agg/plasma-cell-GEX5-counts-250207.h5ad: raw UMI counts of the PCs; obs redcap_id.

Outputs:      data/h5/pat/multi-stage-counts.h5ad: raw counts of the PCs of the selected individuals.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)

# ---- libraries with a disease stage -------------------------------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
meta = meta[meta["assay_type"] == "GEX5"]
meta = meta[meta["disease_status"] != "Other"]
lenient = ~meta["SMM_stage_lenient"].isna()
meta.loc[lenient, "disease_status"] = meta.loc[lenient, "SMM_stage_lenient"]
meta.loc[meta["disease_status"] == "HD", "disease_status"] = "Donor"

# ---- individuals whose BM samples carry more than one disease stage -------------------------------------------------------
x = meta.loc[meta["sample_type"] == "BM", ["disease_status", "redcap_id"]].drop_duplicates()
x = x["redcap_id"].value_counts()
multi_stage_samples = list(x[x > 1].index)
print("individuals with BM samples at more than one disease stage: %d" % len(multi_stage_samples))

# ---- raw counts of all their PCs --------------------------------------------------------------------------------------------
a_counts = sc.read(data("h5", "agg", "plasma-cell-GEX5-counts-250207.h5ad"))
a_pat = a_counts[a_counts.obs["redcap_id"].isin(multi_stage_samples)].copy()
os.makedirs(data("h5", "pat"), exist_ok=True)
a_pat.write(data("h5", "pat", "multi-stage-counts.h5ad"))
