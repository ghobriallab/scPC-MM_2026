#!/usr/bin/env python3
"""Driver arm CNVs of every malignant PC with a CNV clone, and of every CNV segment (Methods, Copy number variant
detection).

Purpose:      Driver arms: the curated driver table (arm, target cytobands, CNV type) plus the HRD arms 3p, 5p, 5q, 7q,
              9p, 9q, 11p, 11q, 15p, 15q, 19p, 19q, 21p and 21q as amplifications. For each sample group (stage:tissue:
              record ID) and Numbat genotype (numbat_GT_opt), every amp/bamp (<arm>+) or del/bdel (<arm>-) segment of
              the genotype on a chromosome of these arms (LOH segments skipped) marks its cells for the arms it touches
              (p or q of its cytobands) when the segment's arms (> 75% covered) or cytobands include one of the arm's
              target regions; cells with neither call get <arm>0. A segment that marks a driver arm of the curated
              table with its listed CNV type gets that call in driver_arm_cnv. Must be run with PYTHONHASHSEED=0.

Inputs:       data/cnv/curated_cnv_drivers_250307.tsv: arm (index), target_region, cnv_type (amp, del or amp_or_del).
              data/cnv/cnv_segments_152_tumors.tsv (05_cnv_segments.py): seg_id, tumor_name, chr, cnv_state, arm,
              cytoband.
              data/cnv/plasma-cell-cell-level-filtered-clone-250306.pkl (09_cell_level_cnv.py): tumor_name,
              numbat_GT_opt.
              data/h5/agg/plasma-cell-GEX5-counts-250207.h5ad: obs annotation, disease_status, sample_type.

Outputs:      data/cnv/cnv_segments_152_tumors.tsv: the segment table with driver_arm_cnv (e.g. 1q+, 13q-).
              data/cnv/plasma-cell-cell-level-filtered-clone-250306.pkl: the clone table with redcap_id, annotation,
              disease_status, sample_type, group (stage:tissue), sample_group (stage:tissue:record ID) and 0/1 columns
              <arm>+, <arm>0, <arm>- for every arm.
              data/cnv/cnv_status_per_sample_group_250308.csv: number of cells per sample group and arm event.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy) and natsort; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import re

import natsort
import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)

SEGS = data("cnv", "cnv_segments_152_tumors.tsv")
CLONES = data("cnv", "plasma-cell-cell-level-filtered-clone-250306.pkl")

# ---- 1. driver arms and their target cytobands -----------------------------------------------------------------------
cnv_drivers = pd.read_csv(data("cnv", "curated_cnv_drivers_250307.tsv"), index_col=0, sep="\t")

# HRD arms (3q and 7p are already in the driver table)
hrd_arms = ["3p", "5p", "5q", "7q", "9p", "9q", "11p", "11q", "15p", "15q", "19p", "19q", "21p", "21q"]
hrd_info = pd.DataFrame({"arm": hrd_arms,
                         "target_region": hrd_arms,
                         "cnv_type": ["amp" for i in hrd_arms]}).set_index("arm")

cnv_collection = pd.concat([cnv_drivers, hrd_info])
cnv_collection["set"] = [set(r.split(",")) for r in cnv_collection["target_region"]]

# ---- 2. segments, clones and sample groups ---------------------------------------------------------------------------
segs = pd.read_csv(SEGS, sep="\t")
segs.index = segs.apply(lambda x: x["seg_id"] + "__" + x["tumor_name"], axis=1)
segs["driver_arm_cnv"] = np.nan

clones = pd.read_pickle(CLONES)

a = sc.read(str(data("h5", "agg", "plasma-cell-GEX5-counts-250207.h5ad")), backed="r")

clones["redcap_id"] = [x.replace("-", "/").split("_Tumor")[0] for x in clones["tumor_name"]]
clones = clones.join(a.obs[["annotation", "disease_status", "sample_type"]])
clones["group"] = clones.apply(lambda x: x["disease_status"] + ":" + x["sample_type"], axis=1)
clones["group"] = pd.Categorical(clones["group"],
                                 categories=["MGUS:BM", "LRSMM:BM", "IRSMM:BM", "HRSMM:BM", "MM:BM",
                                             "MGUS:PB", "LRSMM:PB", "IRSMM:PB", "HRSMM:PB", "MM:PB"])
clones["sample_group"] = clones.apply(lambda x: x["group"] + ":" + x["redcap_id"], axis=1)

events = [arm + cnv_flag for arm in natsort.natsorted(cnv_collection.index) for cnv_flag in ["+", "0", "-"]]
for event in events:
    clones[event] = 0

target_chroms = list(set([int(re.sub("[^0-9]", "", arm)) for arm in natsort.natsorted(cnv_collection.index)]))

suffix = {"amp": "+", "del": "-", "bamp": "+", "bdel": "-"}

cnv_driver_arms = []
for arm, row in cnv_drivers.iterrows():
    for s in row["cnv_type"].split("_or_"):
        cnv_driver_arms.append(arm + suffix[s])

# ---- 3. arm events per cell and driver calls per segment -------------------------------------------------------------
for sample_group, chunk in clones.groupby("sample_group"):
    mat = chunk.copy()

    mat = mat.loc[~mat["numbat_GT_opt"].isna()]
    for seg_ids, df in mat.groupby("numbat_GT_opt"):
        # (1) mark the CNVs of the cells of this genotype
        for seg_id in seg_ids.split(","):

            # skip LOH segments
            if seg_id.endswith("_loh"):
                continue

            index = seg_id.split("_")[0] + "__" + df["tumor_name"].unique()[0]
            cnv = segs.loc[index]

            # skip segments that are not amp, del, bamp or bdel
            state = suffix.get(cnv["cnv_state"], None)
            if state == None:
                continue

            chrom = cnv["chr"]
            # skip chromosomes without a driver or HRD arm
            if not chrom in target_chroms:
                continue

            # str() because arm is NaN for segments without a covered arm
            arms = [arm.replace("_arm", "") for arm in str(cnv["arm"]).split(",") if arm != "nan"]
            cytobands = cnv["cytoband"].split(",")
            cnv_pool = set(arms + cytobands)

            chrom = str(chrom)
            arm_hits = list(set([band.replace(chrom, "")[0] for band in cytobands]))
            arm_keys = [chrom + arm_hit for arm_hit in arm_hits]

            for arm_key in arm_keys:
                if not arm_key in list(cnv_collection.index):
                    continue

                hits = cnv_collection.loc[arm_key, "set"].intersection(cnv_pool)

                if len(hits) > 0:
                    # arm event with its CNV state
                    clones.loc[df.index, arm_key + state] = 1
                    # driver call of the segment
                    if arm_key + state in cnv_driver_arms:
                        segs.loc[index, "driver_arm_cnv"] = arm_key + state

# (2) cells without a call on an arm
neutral = [event for event in events if event.endswith("0")]
for _neu in neutral:
    _del = _neu[:-1] + "-"
    _amp = _neu[:-1] + "+"
    clones.loc[(clones[_del] == 0) & (clones[_amp] == 0), _neu] = 1

# ---- 4. outputs ------------------------------------------------------------------------------------------------------
segs.to_csv(SEGS, sep="\t", index=False)
clones.to_pickle(CLONES)

clone_freq = clones[["sample_group"] + events].groupby("sample_group").sum()
clone_freq.to_csv(data("cnv", "cnv_status_per_sample_group_250308.csv"))
