#!/usr/bin/env python3
"""Per-patient count objects of all PCs and of the annotated (malignant and normal) PCs.

Purpose:      Splits the aggregated count objects by patient (redcap_id) and writes one object per patient; a '/' in a
              patient ID is written as '-' in the file name. Run with PYTHONHASHSEED=0.

Inputs:       data/h5/agg/plasma-cell-counts-240325.h5ad and plasma-annotated-cell-counts-240327.h5ad: obs redcap_id.

Outputs:      data/h5/pat/counts/all/<patient>.h5ad: all PCs of the patient.
              data/h5/pat/counts/annot/<patient>.h5ad: annotated PCs of the patient.

Dependencies: Python 3.8 with requirements.txt (scanpy, anndata); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import scanpy as sc

for source, folder in [("plasma-cell-counts-240325.h5ad", "all"),
                       ("plasma-annotated-cell-counts-240327.h5ad", "annot")]:
    adata = sc.read(str(data("h5", "agg", source)))
    out_dir = data("h5", "pat", "counts", folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    for redcap_id in adata.obs["redcap_id"].unique():
        output_file = out_dir / (redcap_id.replace("/", "-") + ".h5ad")
        adata[adata.obs["redcap_id"] == redcap_id].write(str(output_file))
