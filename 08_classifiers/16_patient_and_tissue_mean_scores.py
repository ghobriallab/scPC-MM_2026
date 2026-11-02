#!/usr/bin/env python3
"""Malignancy and MST scores of every PC and their means per patient and per tissue sample, by disease stage, tissue and
PC status, with the comparisons between groups (Fig. 4D,E; Fig. 5; Extended Data Fig. 8).

Purpose:      Scores every PC with the two final forests (15_final_models_and_cell_scores.py) on its kNN-30 smoothed
              features (malignancy tier: z-scored mean expression of the 16 signatures and senescence score + partner
              genes; MGUS-MM tier: AUCell scores of the 16 signatures + CNV-arm scores quantile-normalized, seed 42; both
              transformations over all cells). Malignancy score = probability of tumor, MST score = probability of MM.
              Units: patient (or tissue sample) x disease stage x tissue x PC status; per unit, the mean of both scores and
              of every smoothed signature score (mean expression and AUCell), and, for malignant units, the cytogenetic
              group of most of their cells. Tests: two-sided Wilcoxon rank-sum tests between every pair of disease stages
              (malignant or normal PCs, BM or PB) and between malignant and normal PCs of each stage; two-sided Wilcoxon
              signed-rank tests between paired BM and PB malignant PCs; Benjamini-Hochberg correction within each comparison
              set. Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/sc-tumor-model-knn30-RF.pkl, sc-disease-model-knn30-RF.pkl (15_final_models_and_cell_scores.py).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt, auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt.
              data/signature/score/cell-level-cohort-smoothed-knn30.pkl (03_knn_smoothing_all_cells.py).
              data/paper/anndata/plasma-cell-GEX5-umap.h5ad: obs patient_id (Pt-###/HD-##), tissue_id, sample_type,
              disease_status, annotation, cytogenetics.

Outputs:      data/classifier/probability/patient_id_mean_probability.pkl, tissue_id_mean_probability.pkl: per unit,
              "Malignancy Score", "scPC-MM Score", disease_status, sample_type, annotation, sample, group, the mean
              mean_expr: and auc: scores (patient table also cytogenetics).
              data/classifier/probability/patient_id_stats.pkl: the tests (comparison, p, fdr) by comparison set.
              data/classifier/probability/cell-level-prediction-knn30.pkl: per cell, normal, tumor, tn_pred, MGUS, MM,
              ds_pred, pred.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels, scanpy, anndata, scikit-learn 1.0.2);
              config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

import pickle
import random
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
import statsmodels.stats.multitest as multi
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)

PROBABILITY = ("classifier", "probability")
data(*PROBABILITY).mkdir(parents=True, exist_ok=True)


def load_list(name):
    return [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", name))]


def compute_zscore(df):
    ss = StandardScaler()
    ss.fit(df)
    return pd.DataFrame(ss.transform(df), index=df.index, columns=df.columns)


def quantile_transform(df):
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(df)
    return pd.DataFrame(qt.transform(df), index=df.index, columns=df.columns)


def format_data(expr, remove_smm=True):
    df = expr.copy()
    if remove_smm:
        df = df[~df["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
    df["class"] = list(df["disease_status"])
    df.loc[df["annotation"] == "normal", "class"] = "normal"

    mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
    mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
    cyto_genes = load_list("cytogenetics_genes.txt")
    auc_gene_modules = load_list("auc_16_gene_modules.txt")
    cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

    X_tn = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes])
    X_ds = df[auc_gene_modules].join(quantile_transform(df[cnv_arms]))

    y_tn = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
    y_ds = np.array(df["class"])
    g = pd.Categorical(list(df["patient_id"]), categories=sorted(set(df["patient_id"])))
    tn_feats = mean_gene_modules + mean_aging_sigs + cyto_genes
    ds_feats = auc_gene_modules + cnv_arms
    return y_tn, y_ds, g, X_tn, X_ds, tn_feats, ds_feats


# ---- 1. scores of every cell with the final forests ----------------------------------------------------------------------
with open(data("classifier", "sc-tumor-model-knn30-RF.pkl"), "rb") as ftn:
    model_tn = pickle.load(ftn)
with open(data("classifier", "sc-disease-model-knn30-RF.pkl"), "rb") as fds:
    model_ds = pickle.load(fds)

a = sc.read(data("paper", "anndata", "plasma-cell-GEX5-umap.h5ad"), backed="r")
all_expr = pd.read_pickle(data("signature", "score", "cell-level-cohort-smoothed-knn30.pkl"))
all_expr = all_expr.join(a.obs[["patient_id", "tissue_id", "sample_type", "disease_status", "annotation"]])
_, _, _, X_tn_all, X_ds_all, _, _ = format_data(all_expr, remove_smm=False)

tn_probs = pd.DataFrame(model_tn.predict_proba(X_tn_all), columns=model_tn.classes_, index=all_expr.index)
tn_probs["tn_pred"] = model_tn.classes_[np.argmax(tn_probs.values, axis=1)]

ds_probs = pd.DataFrame(model_ds.predict_proba(X_ds_all), columns=model_ds.classes_, index=all_expr.index)
ds_probs["ds_pred"] = model_ds.classes_[np.argmax(ds_probs.values, axis=1)]

# call: normal, or the MGUS-MM call of the cells called malignant
probs = tn_probs.join(ds_probs)
probs["pred"] = list(tn_probs["tn_pred"])
probs.loc[probs["pred"] == "tumor", "pred"] = ds_probs.loc[probs["pred"] == "tumor", "ds_pred"]

a.obs = a.obs.join(probs)
a.obs = a.obs.join(all_expr[[x for x in all_expr.columns if not (x in a.obs.columns)]])
a.obs.rename(columns={"tumor": "Malignancy Score", "MM": "scPC-MM Score"}, inplace=True)

# ---- 2. mean scores per patient and per tissue sample ----------------------------------------------------------------
a.obs["sample_disease"] = (np.array(a.obs["disease_status"]) + ":" + np.array(a.obs["sample_type"]) + ":" +
                           np.array(a.obs["annotation"]) + ":" + np.array(a.obs["patient_id"]))
a.obs["tissue_disease"] = (np.array(a.obs["disease_status"]) + ":" + np.array(a.obs["sample_type"]) + ":" +
                           np.array(a.obs["annotation"]) + ":" + np.array(a.obs["tissue_id"]))

patient_probs = a.obs[["sample_disease", "Malignancy Score", "scPC-MM Score"]].groupby("sample_disease").mean()
tissue_probs = a.obs[["tissue_disease", "Malignancy Score", "scPC-MM Score"]].groupby("tissue_disease").mean()


def process_mean_probs(df):
    """split the group label into disease stage, tissue, PC status and patient (or tissue sample)"""
    cell_categs = ["Donor:BM:normal", "MGUS:BM:normal", "LRSMM:BM:normal", "IRSMM:BM:normal", "HRSMM:BM:normal",
                   "MM:BM:normal", "MGUS:BM:tumor", "LRSMM:BM:tumor", "IRSMM:BM:tumor", "HRSMM:BM:tumor", "MM:BM:tumor",
                   "Donor:PB:normal", "MGUS:PB:normal", "LRSMM:PB:normal", "IRSMM:PB:normal", "HRSMM:PB:normal",
                   "MM:PB:normal", "MGUS:PB:tumor", "LRSMM:PB:tumor", "IRSMM:PB:tumor", "HRSMM:PB:tumor", "MM:PB:tumor"]
    X = df.copy()
    X["disease_status"] = [x.split(":")[0] for x in X.index]
    X["sample_type"] = [x.split(":")[1] for x in X.index]
    X["annotation"] = [x.split(":")[2] for x in X.index]
    X["sample"] = [x.split(":")[3] for x in X.index]
    X["group"] = pd.Categorical([":".join(x.split(":")[:3]) for x in X.index], categories=cell_categs)
    return X


patient_probs = process_mean_probs(patient_probs)
tissue_probs = process_mean_probs(tissue_probs)


# ---- 3. tests --------------------------------------------------------------------------------------------------------------
def categ_pair_wilcox(a, b, categ_col, sample_col, value_col, is_paired=False):
    """a vs b within each category: rank-sum test, or signed-rank test on the units present in both"""
    categs = set(a[categ_col].unique()).intersection(set(b[categ_col].unique()))

    res = {"comparison": [], "p": [], "fdr": []}
    for cat in categs:
        a_idx = (a[categ_col] == cat)
        b_idx = (b[categ_col] == cat)
        func = scipy.stats.ranksums

        if is_paired:
            a_sample = a.loc[a_idx, sample_col].values
            b_sample = b.loc[b_idx, sample_col].values
            overlap = set(a_sample).intersection(set(b_sample))
            if len(overlap) == 0:
                continue
            a_idx = a_idx & a[sample_col].isin(overlap)
            b_idx = b_idx & b[sample_col].isin(overlap)
            func = scipy.stats.wilcoxon

        res["comparison"].append(cat)
        res["p"].append(func(list(a.loc[a_idx, value_col]), list(b.loc[b_idx, value_col]))[1])

    res["fdr"] = multi.fdrcorrection(res["p"], alpha=0.05, method="indep", is_sorted=False)[1]
    return pd.DataFrame(res)


def categ_cross_wilcox(df, categ_col, value_col):
    """rank-sum test between every pair of categories"""
    categs = list(df[categ_col].unique())
    res = {"comparison": [], "p": [], "fdr": []}
    for x, y in list(combinations(categs, 2)):
        res["comparison"].append("{},{}".format(x, y))
        res["p"].append(scipy.stats.ranksums(list(df.loc[df[categ_col] == x, value_col]),
                                             list(df.loc[df[categ_col] == y, value_col]))[1])
    res["fdr"] = multi.fdrcorrection(res["p"], alpha=0.05, method="indep", is_sorted=False)[1]
    return pd.DataFrame(res)


def compute_at_once(df, malignant_score="Malignancy Score", mm_score="scPC-MM Score"):
    res = {}

    # disease stages, malignant and normal BM PCs
    bm_tumors = df[(df["annotation"] == "tumor") & (df["sample_type"] == "BM")].copy()
    res["pT|BM:tumor|disease_status"] = categ_cross_wilcox(bm_tumors, "disease_status", malignant_score)
    res["pMM|BM:tumor|disease_status"] = categ_cross_wilcox(bm_tumors, "disease_status", mm_score)
    bm_normals = df[(df["annotation"] == "normal") & (df["sample_type"] == "BM")].copy()
    res["pT|BM:normal|disease_status"] = categ_cross_wilcox(bm_normals, "disease_status", malignant_score)
    res["pMM|BM:normal|disease_status"] = categ_cross_wilcox(bm_normals, "disease_status", mm_score)

    # malignant vs normal BM PCs
    res["pT|BM|TvN"] = categ_pair_wilcox(bm_normals, bm_tumors, "disease_status", "sample", malignant_score)
    res["pMM|BM|TvN"] = categ_pair_wilcox(bm_normals, bm_tumors, "disease_status", "sample", mm_score)

    # disease stages, malignant and normal PB PCs
    pb_tumors = df[(df["annotation"] == "tumor") & (df["sample_type"] == "PB")]
    res["pT|PB:tumor|disease_status"] = categ_cross_wilcox(pb_tumors, "disease_status", malignant_score)
    res["pMM|PB:tumor|disease_status"] = categ_cross_wilcox(pb_tumors, "disease_status", mm_score)
    pb_normals = df[(df["annotation"] == "normal") & (df["sample_type"] == "PB")]
    res["pT|PB:normal|disease_status"] = categ_cross_wilcox(pb_normals, "disease_status", malignant_score)
    res["pMM|PB:normal|disease_status"] = categ_cross_wilcox(pb_normals, "disease_status", mm_score)

    # normal PCs, BM vs PB
    res["pT|normal|BMvPB"] = categ_pair_wilcox(bm_normals, pb_normals, "disease_status", "sample", malignant_score)
    res["pMM|normal|BMvPB"] = categ_pair_wilcox(bm_normals, pb_normals, "disease_status", "sample", mm_score)

    # malignant PCs, BM vs PB (paired)
    res["pT|tumor|BMvPB"] = categ_pair_wilcox(bm_tumors, pb_tumors, "disease_status", "sample", malignant_score,
                                              is_paired=True)
    res["pMM|tumor|BMvPB"] = categ_pair_wilcox(bm_tumors, pb_tumors, "disease_status", "sample", mm_score, is_paired=True)

    # malignant vs normal PB PCs
    res["pT|PB|TvN"] = categ_pair_wilcox(pb_normals, pb_tumors, "disease_status", "sample", malignant_score)
    res["pMM|PB|TvN"] = categ_pair_wilcox(pb_normals, pb_tumors, "disease_status", "sample", mm_score)
    return res


patient_stats = compute_at_once(patient_probs)

# ---- 4. cytogenetic group and mean signature scores per unit; save -------------------------------------------------------
patient_probs["cytogenetics"] = np.nan
for patient_group in patient_probs[patient_probs["annotation"] == "tumor"].index:
    df = pd.DataFrame(a.obs.loc[a.obs["sample_disease"] == patient_group, "cytogenetics"].value_counts())
    cyto = list(df.head(n=1).index)[0]
    patient_probs.loc[patient_group, "cytogenetics"] = cyto
patient_probs["cytogenetics"] = pd.Categorical(patient_probs["cytogenetics"],
                                               categories=["t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)", "t(12;14)",
                                                           "t(14;16)", "t(14;20)", "HRD", "Other"])

exr_columns = list(all_expr.columns[all_expr.columns.str.startswith("mean_expr:")])
auc_columns = list(all_expr.columns[all_expr.columns.str.startswith("auc:")])
patient_scores = a.obs[["sample_disease"] + exr_columns + auc_columns].groupby("sample_disease").mean()
tissue_scores = a.obs[["tissue_disease"] + exr_columns + auc_columns].groupby("tissue_disease").mean()

patient_probs = patient_probs.join(patient_scores)
patient_probs.to_pickle(data(*PROBABILITY, "patient_id_mean_probability.pkl"))
tissue_probs = tissue_probs.join(tissue_scores)
tissue_probs.to_pickle(data(*PROBABILITY, "tissue_id_mean_probability.pkl"))
with open(data(*PROBABILITY, "patient_id_stats.pkl"), "wb") as fo:
    pickle.dump(patient_stats, fo)
probs.to_pickle(data(*PROBABILITY, "cell-level-prediction-knn30.pkl"))
print("done")
