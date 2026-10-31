#!/usr/bin/env python3
"""Percentage of MM-like malignant PCs (MST score above 0.5) in each BM tumor and in each of its Numbat CNV clones (Figs. 5A
and 6; Results).

Purpose:      MM-like cells are malignant PCs with an MST score > 0.5, the cells the MGUS-MM classifier calls MM. A
              tumor is the malignant BM PCs of one patient at one disease stage (all BM samples at that stage pooled;
              SMM risk groups of the IMWG 2/20/20 model, risk.py), kept when it has at least 30 cells. Per tumor: number of cells, mean MST score and fraction of MM-like
              cells, and the 10th, 50th and 90th percentiles of the MST score; the mean MST score is checked against the
              saved per-tumor MST score. Per tumor and Numbat clone (clone 1, the diploid clone, and cells without a tumor
              clone left out): cells and MM-like cells. The statistics for SMM patients are computed from these tables by
              09_mm_like_cells_statistics.py, those across stages and clones by 11_mm_like_cells_out_of_sample.py.

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (per cell: gex_id, annotation, sample_type,
              disease_status); data/metadata/tumor-cohort-metadata-final-240325.csv (assay_type, gex_id, patient_id);
              data/classifier/probability/cell-level-prediction-knn30.pkl (MM = MST score per cell);
              data/paper/classifier/patient_id_mean_probability.pkl (sample, sample_type, annotation, disease_status,
              "MST Score"; check); data/cnv/plasma-cell-cell-level-filtered-clone-250306.pkl (clone_id, numbat_annotation).

Outputs:      results/stage_and_progression/mm_like_cells_tumors.tsv: per tumor: patient, disease stage, cells, mean MST score,
              fraction of MM-like cells, MST percentiles.
              results/stage_and_progression/mm_like_cells_clones.tsv: per tumor and Numbat CNV clone: cells, MM-like cells.
              results/stage_and_progression/mm_like_cells_log.txt: cell counts and the check.

Dependencies: Python 3.8 with requirements.txt (pandas); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import gc

import pandas as pd

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "mm_like_cells_log.txt"), "w")
MIN_CELLS = 30          # tumor = >= 30 malignant BM PCs of a patient at one stage, as in the paper
MM_LIKE = 0.5           # MST > 0.5: the classifier calls the cell MM


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


# ---- 0. cells: patient, stage, tissue, malignant/normal, MST ----------------------------------------
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
ann = ann[["gex_id", "annotation", "sample_type", "disease_status"]].copy()
gc.collect()
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
meta = meta[meta["assay_type"] == "GEX5"].drop_duplicates("gex_id").set_index("gex_id")
ann["patient_id"] = ann["gex_id"].map(meta["patient_id"])
ann["MST"] = pd.read_pickle(data("classifier", "probability", "cell-level-prediction-knn30.pkl"))["MM"].reindex(ann.index).values
gc.collect()
log("cells: %d; with a patient: %.4f; with an MST score: %.4f" % (len(ann), ann["patient_id"].notna().mean(), ann["MST"].notna().mean()))

# ---- 1. MM-like cells within each tumor -------------------------------------------------------------
# malignant BM PCs; one tumor = patient x disease stage (all BM samples at that stage pooled, as in the paper)
bm = ann[(ann["sample_type"] == "BM") & (ann["annotation"] == "tumor") & ann["patient_id"].notna()].copy()
del ann
gc.collect()
bm["disease_status"] = stages(bm["patient_id"], bm["sample_type"], bm["disease_status"])
bm["mm_like"] = bm["MST"] > MM_LIKE
T = bm.groupby(["patient_id", "disease_status"]).agg(cells=("MST", "size"), mean_mst=("MST", "mean"),
                                                     frac_mm_like=("mm_like", "mean")).reset_index()
T = T[T["cells"] >= MIN_CELLS]
Q = bm.groupby(["patient_id", "disease_status"])["MST"].quantile([0.1, 0.5, 0.9]).unstack()
Q.columns = ["q10_mst", "median_mst", "q90_mst"]
T = T.join(Q, on=["patient_id", "disease_status"])
T.to_csv(os.path.join(OUT, "mm_like_cells_tumors.tsv"), sep="\t", index=False)

# Numbat CNV clones within each tumor (clone 1 is the diploid clone, without a CNV call: left out)
cl = pd.read_pickle(data("cnv", "plasma-cell-cell-level-filtered-clone-250306.pkl"))[["clone_id", "numbat_annotation"]]
cl = cl[cl["numbat_annotation"] == "tumor"]["clone_id"].astype(str).str.rsplit(":", n=1).str[-1]
bm["clone"] = bm.index.map(cl).fillna("none").astype(str)
del cl
gc.collect()
C = bm[~bm["clone"].isin(["none", "1"])].groupby(["patient_id", "disease_status", "clone"]).agg(
    cells=("mm_like", "size"), mm_like=("mm_like", "sum")).reset_index()
C = C.merge(T[["patient_id", "disease_status"]], on=["patient_id", "disease_status"])       # tumors kept above
C.to_csv(os.path.join(OUT, "mm_like_cells_clones.tsv"), sep="\t", index=False)
log("malignant BM PCs in a Numbat CNV clone: %d of %d; tumors with a clone: %d" % (
    (~bm["clone"].isin(["none", "1"])).sum(), len(bm), C.groupby(["patient_id", "disease_status"]).ngroups))
ref = pd.read_pickle(data("paper", "classifier", "patient_id_mean_probability.pkl"))
ref = ref[(ref["sample_type"] == "BM") & (ref["annotation"] == "tumor")].rename(columns={"sample": "patient_id"})
ref["disease_status"] = stages(ref["patient_id"], ref["sample_type"], ref["disease_status"])
ref = ref[~ref.duplicated(["patient_id", "disease_status"], keep=False)]      # pooled tumors have no single saved value
chk = T.merge(ref[["patient_id", "disease_status", "MST Score"]], on=["patient_id", "disease_status"])
log("\ntumors: %d (%d patients); check vs the saved per-tumor MST score: %d matched, max abs difference %.1e"
    % (len(T), T["patient_id"].nunique(), len(chk), (chk["mean_mst"] - chk["MST Score"]).abs().max()))
log("\ndone")
FH.close()
