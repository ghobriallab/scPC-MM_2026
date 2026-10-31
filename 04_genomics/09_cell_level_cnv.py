#!/usr/bin/env python3
"""CNV probabilities per malignant PC and segment, and the CNV clone of every malignant PC (Methods, Copy number
variant detection).

Purpose:      1. For every tumor, the Numbat posteriors of iteration 2 per cell and segment (joint_post_2.tsv)
                 restricted to amp, del, bamp and bdel (and, when the column cnv_state_expand exists, to rows whose
                 expanded state is part of the segment label); segment ID and CNV state taken from seg_label; kept for
                 cells of the plasma cell object annotated malignant (annotation "tumor").
              2. Clone of each malignant PC: the curated cell-level clone table (clean clone number, CNV profile,
                 subclonal CNVs and subclonal HRD per cell) joined with the Numbat clone, genotype, probability and
                 compartment of the same cell (clone_post_2.tsv of the tumors of the segment table); clone_id =
                 <tumor_name>:<clean clone>; clones with >= 100 cells kept.
              Must be run with PYTHONHASHSEED=0.

Inputs:       data/amp1q/cnv/SubcloneDetection/<tumor_name>/joint_post_2.tsv (Numbat): cell, seg_label, cnv_state,
              cnv_state_expand, p_cnv; clone_post_2.tsv (Numbat): cell, clone_opt, GT_opt, p_opt, compartment_opt.
              data/h5/agg/plasma-cell-GEX5-counts-250207.h5ad: obs annotation.
              data/cnv/cnv_segments_152_tumors.tsv (05_cnv_segments.py): tumor_name.
              data/amp1q/cnv/cnv_scripts/CellLevelNumbatData.csv: curated clone of each cell (Barcode, PatientID_Tumor,
              numbatclone, Clone_CNV_Profile, SubclonalCNVs, SubclonalHRD).

Outputs:      data/cnv/plasma-cell-cell-level-cnv-prob-250306.pkl: one row per malignant PC and segment: seg_id,
              cnv_state, p_cnv, tumor_name.
              data/cnv/plasma-cell-cell-level-filtered-clone-250306.pkl: one row per malignant PC of a clone with >= 100
              cells: clone_id, tumor_name, clean_clone_opt, manual_cnv, manual_subclonal_cnv, manual_subclonal_hrd,
              numbat_clone_opt, numbat_GT_opt, numbat_p_opt, numbat_annotation.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy); ls; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import shlex
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)

NUMBAT = data("amp1q", "cnv", "SubcloneDetection")

# ---- 1. CNV probabilities per cell and segment -----------------------------------------------------------------------
joint_post_files = subprocess.check_output("ls -1 %s/*/joint_post_2.tsv" % shlex.quote(str(NUMBAT)), shell=True)
joint_post_files = joint_post_files.decode("utf-8").rstrip("\n").split("\n")

cnv_of_interest = ["amp", "del", "bamp", "bdel"]

cell_agg = []
for joint_post_file in joint_post_files:
    tumor_name = os.path.basename(os.path.dirname(joint_post_file))

    cells = pd.read_csv(joint_post_file, sep="\t", index_col=0)

    cells = cells[cells["cnv_state"].isin(cnv_of_interest)]
    # outliers: keep rows whose expanded state is part of the segment label
    if "cnv_state_expand" in cells.columns:
        cells = cells[cells.apply(lambda x: x["cnv_state_expand"] in x["seg_label"], axis=1)]
    cells = cells[cells["cnv_state"].isin(cnv_of_interest)]

    # no CNV
    if cells.shape[0] == 0:
        continue

    # segment ID and state from the segment label, e.g. 1c(amp)
    cells["seg_id"] = cells["seg_label"].apply(lambda x: x.split("(")[0])
    cells["cnv_state"] = cells["seg_label"].apply(lambda x: x.split("(")[1].replace(")", ""))
    cells["tumor_name"] = tumor_name
    cells = cells[["seg_id", "cnv_state", "p_cnv", "tumor_name"]]

    cell_agg.append(cells)

cell_agg = pd.concat(cell_agg)

# malignant PCs of the plasma cell object
a = sc.read(str(data("h5", "agg", "plasma-cell-GEX5-counts-250207.h5ad")), backed="r")

clean_cell_agg = cell_agg[cell_agg.index.isin(a.obs.index)]
clean_cell_agg = clean_cell_agg.join(a.obs[["annotation"]])
clean_cell_agg = clean_cell_agg[clean_cell_agg["annotation"] == "tumor"].drop(columns="annotation")

clean_cell_agg.to_pickle(data("cnv", "plasma-cell-cell-level-cnv-prob-250306.pkl"))

# ---- 2. clone of every malignant PC ----------------------------------------------------------------------------------
segs = pd.read_csv(data("cnv", "cnv_segments_152_tumors.tsv"), sep="\t")

# Numbat clone assignments of the tumors with CNV segments
clones = []
for tumor_name in segs["tumor_name"].unique():
    clone_post = pd.read_csv(NUMBAT / tumor_name / "clone_post_2.tsv", index_col=0, sep="\t")
    clone_post["tumor_name"] = tumor_name
    clones.append(clone_post[["clone_opt", "GT_opt", "p_opt", "tumor_name", "compartment_opt"]])

clones = pd.concat(clones)
clones = clones.loc[clones.index.isin(a.obs.index)]
clones = clones.join(a.obs[["annotation"]])
clones = clones[clones["annotation"] == "tumor"]

# curated clones of each cell
filtered_clones = pd.read_csv(data("amp1q", "cnv", "cnv_scripts", "CellLevelNumbatData.csv"), index_col=0)
filtered_clones.drop(columns=["Barcode"], inplace=True)
filtered_clones.rename(columns={"PatientID_Tumor": "tumor_name", "numbatclone": "clone_opt"}, inplace=True)
filtered_clones = filtered_clones[filtered_clones.index.isin(a.obs.index)]
filtered_clones = filtered_clones.join(a.obs[["annotation"]])
filtered_clones = filtered_clones[filtered_clones["annotation"] == "tumor"]

# add the Numbat clone, genotype, probability and compartment of each cell
columns = ["clone_opt", "GT_opt", "p_opt", "compartment_opt"]
filtered_clones = filtered_clones.join(
    clones.loc[filtered_clones.index, columns].rename(columns={"clone_opt": "original_clone_opt"})
)

# keep clones with >= 100 cells
filtered_clones["clone_id"] = filtered_clones.apply(lambda x: x["tumor_name"] + ":" + str(x["clone_opt"]), axis=1)
freq = pd.DataFrame(filtered_clones["clone_id"].value_counts(dropna=False))
filtered_clones = filtered_clones[filtered_clones["clone_id"].isin(freq.loc[freq["clone_id"] >= 100].index)]

filtered_clones.rename(columns={"clone_opt": "clean_clone_opt",
                                "original_clone_opt": "numbat_clone_opt",
                                "GT_opt": "numbat_GT_opt",
                                "p_opt": "numbat_p_opt",
                                "compartment_opt": "numbat_annotation"}, inplace=True)
filtered_clones.rename(columns={"Clone_CNV_Profile": "manual_cnv",
                                "SubclonalCNVs": "manual_subclonal_cnv",
                                "SubclonalHRD": "manual_subclonal_hrd"}, inplace=True)

columns = ["clone_id", "tumor_name", "clean_clone_opt", "manual_cnv", "manual_subclonal_cnv",
           "manual_subclonal_hrd", "numbat_clone_opt", "numbat_GT_opt", "numbat_p_opt", "numbat_annotation"]
filtered_clones[columns].to_pickle(data("cnv", "plasma-cell-cell-level-filtered-clone-250306.pkl"))
