#!/usr/bin/env python3
"""Phylogeny of the CNV clones: tree position, genotype and driver arm CNVs of every clone in every sample group
(Methods, Copy number variant detection).

Purpose:      One row per clone and cell group (clone, stage, tissue; duplicates of the cell table dropped). Two clean
              clones whose cells carry two Numbat genotypes (Pt-182 clone 3, Pt-34 clone 7) get the genotype and Numbat
              clone of the rest of the clone for the cells with the extra segment. Tree: the Numbat mutation graph of
              the tumor restricted to its clones of >= 100 cells, rooted at the earliest of them (tumors with one such
              clone have no tree); per clone: root, leaf, depth (from that clone), parent, child, n_child, GT
              (segments) and subGT (segments not in the parent). Driver arm CNVs: one 0/1 column per driver call of the
              segment table (driver_arm_cnv), 1 when the clone's GT has a segment with that call, and arm_driver_hit,
              their number.
              progressor: 1 for patients with progression_status 1 in the cohort metadata, 0 for patients with
              progression_status 0 (0 when both occur), empty otherwise. Must be run with PYTHONHASHSEED=0.

Inputs:       data/cnv/cnv_segments_152_tumors.tsv (10_clone_driver_arms.py): seg_id, tumor_name, driver_arm_cnv.
              data/cnv/plasma-cell-cell-level-filtered-clone-250306.pkl (10_clone_driver_arms.py): clone_id, tumor_name,
              clean_clone_opt, numbat_GT_opt, numbat_clone_opt, redcap_id, annotation, disease_status, sample_type,
              group, sample_group.
              data/cnv/mutgraph/<tumor_name>.node.tsv and .edge.tsv (04_clone_mutation_graphs.R).
              data/metadata/tumor-cohort-metadata-final-240325.csv: patient_id, redcap_id, disease_status,
              SMM_stage_lenient, progression_status.

Outputs:      data/cnv/clone-phylogeny-250321.pkl: one row per clone and cell group: the columns above, root, leaf,
              depth, parent, child, n_child, GT, subGT, the driver arm CNV columns, arm_driver_hit and progressor.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas) and natsort; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import natsort
import numpy as np
import pandas as pd

np.random.seed(0)

# ---- 1. segments, clones and patient labels --------------------------------------------------------------------------
segs = pd.read_csv(data("cnv", "cnv_segments_152_tumors.tsv"), sep="\t")
segs.index = segs.apply(lambda x: x["seg_id"] + "__" + x["tumor_name"], axis=1)

clones = pd.read_pickle(data("cnv", "plasma-cell-cell-level-filtered-clone-250306.pkl"))
clone_info = ["clone_id", "tumor_name", "clean_clone_opt", "numbat_GT_opt", "numbat_clone_opt"]
sample_info = ["redcap_id", "annotation", "disease_status", "sample_type", "group", "sample_group"]
clones = clones[clone_info + sample_info]

cohort = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
pt2rec = cohort.drop_duplicates("patient_id").set_index("patient_id")["redcap_id"]


def tumor_name_of(patient):
    """Numbat tumor name of a patient: record ID with '-' for '/', then '_Tumor'."""
    return pt2rec[patient].replace("/", "-") + "_Tumor"


# ---- 2. one genotype per clean clone ---------------------------------------------------------------------------------
# cells with an extra segment take the genotype and Numbat clone of the rest of their clean clone
idx = (clones["clone_id"] == tumor_name_of("Pt-182") + ":3") & \
    (clones["numbat_GT_opt"] == "11a,15a,9a,19a,5a,7a_amp,13a,7a_loh")
clones.loc[idx, "numbat_GT_opt"] = "11a,15a,9a,19a,5a,7a_amp,13a"
clones.loc[idx, "numbat_clone_opt"] = 3

idx = (clones["clone_id"] == tumor_name_of("Pt-34") + ":7") & \
    (clones["numbat_GT_opt"] == "11a,7a,18a,19a,3a,22b,9a,5b,15a,5a,6e,8b,4b,2b,8a")
clones.loc[idx, "numbat_GT_opt"] = "11a,7a,18a,19a,3a,22b,9a,5b,15a,5a,6e,8b,4b,2b"
clones.loc[idx, "numbat_clone_opt"] = 7

clones = clones.drop_duplicates()


# ---- 3. clone trees --------------------------------------------------------------------------------------------------
class Node(object):
    def __init__(self, data, GT, depth):
        self.data = data
        self.depth = depth
        self.GT = GT
        self.subGT = None
        self.cytoband = None
        self.subcytoband = None
        self.parent = None
        self.children = []

    def __str__(self):
        _str = "name={}, depth={}, GT={} | parent={}, children={}"
        return _str.format(self.data,
                           self.depth,
                           self.GT,
                           self.get_parent_name(),
                           self.get_children_names())

    def is_root(self):
        return self.depth == 0

    def is_leaf(self):
        return self.depth > 0 and self.children == []

    def add_parent(self, obj):
        self.parent = obj

    def add_child(self, obj):
        self.children.append(obj)

    def set_subGT(self):
        # segments of the clone that its parent does not have
        if self.parent == None:
            self.subGT = self.GT
            return
        elif self.depth == 0:
            self.subGT = self.GT
            return
        elif str(self.parent.GT) == "nan":
            self.subGT = self.GT
        else:
            node_GT = set(self.GT.split(","))
            parent_GT = set(self.parent.GT.split(","))
            self.subGT = ",".join(natsort.natsorted(node_GT.difference(parent_GT)))
            return

    def get_parent_name(self):
        if self.parent == None:
            return None
        else:
            return self.parent.data

    def get_children_names(self):
        return [c.data for c in self.children]


def generate_tree(tumor_name, clones):
    tree = {}

    edge_path = data("cnv", "mutgraph", "{}.edge.tsv".format(tumor_name))
    node_path = data("cnv", "mutgraph", "{}.node.tsv".format(tumor_name))

    # no tree
    if not os.path.exists(edge_path):
        return tree

    edge_df = pd.read_csv(edge_path, sep="\t")
    node_df = pd.read_csv(node_path, sep="\t").set_index("clone")
    node_df.loc[node_df["parent"].isna(), "parent"] = 0
    node_df["parent"] = node_df["parent"].astype(dtype=int)

    valid_clones = clones.loc[clones["tumor_name"] == tumor_name, "clean_clone_opt"].unique()
    early_clone = node_df.loc[node_df.index.isin(valid_clones)].sort_values("depth").index[0]

    # no tree with a single clone
    if len(valid_clones) == 1:
        return tree

    # depths start from the earliest of the kept clones
    depth_offset = node_df.loc[early_clone, "depth"]
    node_df = node_df.loc[valid_clones].sort_values("depth")

    edge_idx = (edge_df["from"].isin(valid_clones) & edge_df["to"].isin(valid_clones)) & (edge_df["to"] != early_clone)
    edge_df = edge_df[edge_idx]

    # nodes
    for gt, row in node_df.iterrows():
        depth = row["depth"] - depth_offset
        tree[gt] = Node(gt, row["GT"], depth)

    # parents and children
    for _, row in edge_df.iterrows():
        parent = row["from"]
        child = row["to"]
        tree[parent].add_child(tree[child])
        tree[child].add_parent(tree[parent])

    # segments gained relative to the parent
    for gt in tree.keys():
        tree[gt].set_subGT()

    return tree


for tumor_name, df in clones.groupby("tumor_name"):
    tree = generate_tree(tumor_name, clones)
    if len(tree) == 0:
        continue

    # tree position of each clone
    root_id = [i for i in tree if tree[i].is_root()][0]
    for clone_id in df["clone_id"].unique():
        int_clone_id = int(clone_id.split(":")[-1])
        idx = clones["clone_id"] == clone_id

        clones.loc[idx, "root"] = tree[int_clone_id].is_root()
        clones.loc[idx, "leaf"] = tree[int_clone_id].is_leaf()
        clones.loc[idx, "depth"] = tree[int_clone_id].depth
        clones.loc[idx, "parent"] = tree[int_clone_id].get_parent_name()

        child = [str(c) for c in tree[int_clone_id].get_children_names()]
        clones.loc[idx, "child"] = ",".join(child)
        clones.loc[idx, "n_child"] = len(child)

        clones.loc[idx, "GT"] = tree[int_clone_id].GT
        clones.loc[idx, "subGT"] = tree[int_clone_id].subGT

clones.reset_index(drop=True, inplace=True)

# ---- 4. driver arm CNVs of each clone --------------------------------------------------------------------------------
arm_cnvs = natsort.natsorted(segs.loc[~segs["driver_arm_cnv"].isna(), "driver_arm_cnv"].unique())
for arm_cnv in arm_cnvs:
    clones[arm_cnv] = 0
    hits = set(segs.loc[segs["driver_arm_cnv"] == arm_cnv].index)
    for i, row in clones[~clones["GT"].isna()].iterrows():
        clone_segs = set([gt.split("_")[0] + "__" + row["tumor_name"] for gt in row["GT"].split(",")])
        if len(clone_segs.intersection(hits)) > 0:
            clones.loc[i, arm_cnv] = 1

clones["arm_driver_hit"] = clones[arm_cnvs].sum(axis=1)

# ---- 5. progression status -------------------------------------------------------------------------------------------
meta = cohort.copy()
meta = meta[meta["disease_status"] != "Other"]
meta.loc[~meta["SMM_stage_lenient"].isna(), "disease_status"] = meta.loc[~meta["SMM_stage_lenient"].isna(),
                                                                         "SMM_stage_lenient"]
meta.loc[meta["disease_status"] == "HD", "disease_status"] = "Donor"

progressors = meta.loc[meta["progression_status"] == 1, "redcap_id"].unique()
non_progressors = meta.loc[meta["progression_status"] == 0, "redcap_id"].unique()

clones["progressor"] = np.nan
clones.loc[clones["redcap_id"].isin(progressors), "progressor"] = 1
clones.loc[clones["redcap_id"].isin(non_progressors), "progressor"] = 0

clones.to_pickle(data("cnv", "clone-phylogeny-250321.pkl"))
