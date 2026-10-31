#!/usr/bin/env python3
"""Per-cell annotation table of the 1,649,290 PCs, read by the later steps (cell type, cytogenetic group, disease stage and
analysis units of every PC).

Purpose:      Annotation of every PC of the 5' PC object: cell annotations, cytogenetic group of the patient's tumor,
              disease stage (SMM risk group from SMM_stage_lenient; HD renamed Donor; the IMWG 2/20/20 groups are applied
              downstream with risk.py) and the group labels cell_categ (stage x tissue x cell type), sample_categ (stage x
              tissue x individual) and sample_pcg (unit = stage x tissue x individual x cell type).

Inputs:       data/h5/agg/plasma-cell-GEX5-umap-250207.h5ad: obs index (the PCs).
              data/annotation/plasma-cell-annotation-final-240327.csv: one row per cell; gex_id, redcap_id, sample_type,
              annotation, tumor_id and other annotations.
              data/metadata/cytogenetics-final-240327.csv: index = <redcap_id>:tumor:<k>; cytogenetics, hrd_status.
              data/metadata/tumor-cohort-metadata-final-240325.csv: index = gex_id; disease_status, SMM_stage_lenient.

Outputs:      data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: one row per PC.

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
STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
CYTO = ["t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)", "t(12;14)", "t(14;16)", "t(14;20)", "HRD", "Other"]
adata = sc.read(data("h5", "agg", "plasma-cell-GEX5-umap-250207.h5ad"), backed="r")

# ---- annotation of every PC ---------------------------------------------------------------------------------------------------
annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
cytogenetics = pd.read_csv(data("metadata", "cytogenetics-final-240327.csv"), index_col=0)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
lenient = ~meta["SMM_stage_lenient"].isna()
meta.loc[lenient, "disease_status"] = meta.loc[lenient, "SMM_stage_lenient"]
annot = annot.loc[adata.obs.index]
annot = annot.join(cytogenetics[["cytogenetics", "hrd_status"]], on="tumor_id")
annot["cytogenetics"] = pd.Categorical(annot["cytogenetics"], categories=CYTO)
annot["disease_status"] = list(meta.loc[annot["gex_id"], "disease_status"])
annot.loc[annot["disease_status"] == "HD", "disease_status"] = "Donor"

cell_categs = ["Donor:BM:normal", "MGUS:BM:normal", "LRSMM:BM:normal", "IRSMM:BM:normal", "HRSMM:BM:normal", "MM:BM:normal",
               "MGUS:BM:tumor", "LRSMM:BM:tumor", "IRSMM:BM:tumor", "HRSMM:BM:tumor", "MM:BM:tumor",
               "Donor:PB:normal", "MGUS:PB:normal", "LRSMM:PB:normal", "IRSMM:PB:normal", "HRSMM:PB:normal", "MM:PB:normal",
               "MGUS:PB:tumor", "LRSMM:PB:tumor", "IRSMM:PB:tumor", "HRSMM:PB:tumor", "MM:PB:tumor"]
annot["cell_categ"] = list(np.array(annot["disease_status"]) + ":" + np.array(annot["sample_type"]) + ":"
                           + np.array(annot["annotation"]))
annot["cell_categ"] = pd.Categorical(annot["cell_categ"], categories=cell_categs, ordered=True)

annot["sample_categ"] = list(np.array(annot["disease_status"]) + ":" + np.array(annot["sample_type"]) + ":"
                             + np.array(annot["redcap_id"]))
sample_group = sorted(annot["sample_categ"].unique())
ordered = []
for tissue in ["BM", "PB"]:
    for stage in STAGES:
        ordered += [x for x in sample_group if x.startswith(stage + ":" + tissue)]
sample_group = ordered
annot["sample_categ"] = pd.Categorical(annot["sample_categ"], categories=sample_group, ordered=True)

annot["sample_pcg"] = list(np.array(annot["disease_status"]) + ":" + np.array(annot["sample_type"]) + ":"
                           + np.array(annot["redcap_id"]) + ":" + np.array(annot["annotation"]))
sample_pcg_bm = ([sg + ":normal" for sg in sample_group if ":BM:" in sg]
                 + [sg + ":tumor" for sg in sample_group if (sg.split(":")[0] != "Donor") and (":BM:" in sg)])
sample_pcg_pb = ([sg + ":normal" for sg in sample_group if ":PB:" in sg]
                 + [sg + ":tumor" for sg in sample_group if (sg.split(":")[0] != "Donor") and (":PB:" in sg)])
annot["sample_pcg"] = pd.Categorical(annot["sample_pcg"], categories=sample_pcg_bm + sample_pcg_pb, ordered=True)
annot.to_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))

