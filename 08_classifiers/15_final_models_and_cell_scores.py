#!/usr/bin/env python3
"""Final malignancy and MGUS-MM random forests trained on the training and test cells, and the malignancy and MST scores
of every PC with their means per sample group (Methods, classifier development; Fig. 4D,E).

Purpose:      Training cells of donors, MGUS and MM patients (training and test sets together, SMM removed),
              cohort-smoothed features. Malignancy tier: z-scored mean expression of the 16 signatures and senescence score
              + partner genes, 100 trees, max_depth 30, min_samples_split 20, min_samples_leaf 30, balanced_subsample.
              MGUS-MM tier: AUCell scores of the 16 signatures + CNV-arm scores quantile-normalized (seed 42), trained on
              the malignant cells, 200 trees, max_depth 25, min_samples_split 20, min_samples_leaf 10, balanced. The
              forests get no random_state: they draw from numpy's global random state, seeded with 42 once, malignancy tier
              first. Every PC (kNN-30 smoothed features, z-scored and quantile-normalized over all cells) then gets the
              malignancy score (probability of tumor) and the MST score (probability of MM), and the call normal, MGUS or
              MM. Means of the scores and of the classifier inputs per individual and per sample, by disease stage, tissue
              and PC status (and per sample and BCR isotype); the majority cytogenetic group of the malignant PCs of each
              unit; two-sided Wilcoxon rank-sum tests between disease stages and between malignant and normal PCs, and
              two-sided Wilcoxon signed-rank tests between paired BM and PB malignant PCs, Benjamini-Hochberg correction
              within each comparison set. Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl,
              sc-test-cohort-smoothed-expr_47samples.pkl: feature columns, annotation, disease_status, redcap_id.
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt, auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt.
              data/signature/score/cell-level-cohort-smoothed-knn30.pkl (03_knn_smoothing_all_cells.py): smoothed features
              of every cell.
              data/h5/agg/plasma-cell-GEX5-umap-250207.h5ad: obs redcap_id, catissue_id, sample_type, disease_status,
              annotation, isotype_status, cytogenetics.

Outputs:      data/classifier/sc-tumor-model-knn30-RF.pkl, sc-disease-model-knn30-RF.pkl: the two final forests;
              X_tn_all_input_data.pkl, X_ds_all_input_data.pkl: classifier inputs of every cell.
              data/classifier/probability/: cell-level-prediction-knn30.pkl (per cell: normal, tumor, tn_pred, MGUS, MM,
              ds_pred, pred); {redcap_id,catissue_id,catissue_isotype}_mean_X_{ds,tn}_scores.pkl (mean classifier inputs
              per individual, per sample, per sample and isotype); catissue_id_isotype_fraction.pkl (cells and fraction of
              each isotype per sample); catissue_isotype_mean_probability.pkl; redcap_id_mean_probability.csv and
              catissue_id_mean_probability.csv (mean scores per individual and per sample, majority cytogenetic group per
              individual); redcap_id_stats.pkl (tests: comparison, p, fdr, by comparison set).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels, scanpy, anndata, scikit-learn 1.0.2);
              config.py. Other scikit-learn versions grow different trees from the same random state.
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
from sklearn.ensemble import RandomForestClassifier
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
    g = pd.Categorical(list(df["redcap_id"]), categories=sorted(set(df["redcap_id"])))
    tn_feats = mean_gene_modules + mean_aging_sigs + cyto_genes
    ds_feats = auc_gene_modules + cnv_arms
    return y_tn, y_ds, g, X_tn, X_ds, tn_feats, ds_feats


# ---- 1. final forests, trained on the training and test cells -------------------------------------------------------------
tn_mex_gm16_age_cyto_params = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 30, "min_samples_split": 20,
                               "min_samples_leaf": 30, "bootstrap": True, "class_weight": "balanced_subsample",
                               "n_jobs": -1}
ds_auc_gm16_arm_params = {"n_estimators": 200, "max_features": "sqrt", "max_depth": 25, "min_samples_split": 20,
                          "min_samples_leaf": 10, "bootstrap": True, "class_weight": "balanced", "n_jobs": -1}

train_expr = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl"))
test_expr = pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-expr_47samples.pkl"))
expr = pd.concat([train_expr, test_expr])
y_tn, y_ds, _, X_tn, X_ds, tn_feats, ds_feats = format_data(expr)

model_tn = RandomForestClassifier(**tn_mex_gm16_age_cyto_params)
model_tn.fit(X_tn, y_tn)

model_ds = RandomForestClassifier(**ds_auc_gm16_arm_params)
X_ds_ = X_ds[y_tn == "tumor"]
y_ds_ = pd.Categorical(list(y_ds[y_tn == "tumor"]), categories=["MGUS", "MM"])
model_ds.fit(X_ds_, y_ds_)

with open(data("classifier", "sc-tumor-model-knn30-RF.pkl"), "wb") as ftn:
    pickle.dump(model_tn, ftn)
with open(data("classifier", "sc-disease-model-knn30-RF.pkl"), "wb") as fds:
    pickle.dump(model_ds, fds)
del train_expr, test_expr, expr

# ---- 2. scores of every cell ----------------------------------------------------------------------------------------------
all_expr = pd.read_pickle(data("signature", "score", "cell-level-cohort-smoothed-knn30.pkl"))
a = sc.read(data("h5", "agg", "plasma-cell-GEX5-umap-250207.h5ad"))
all_expr = all_expr.join(a.obs[["redcap_id", "catissue_id", "sample_type", "disease_status", "annotation"]])

_, _, _, X_tn_all, X_ds_all, _, _ = format_data(all_expr, remove_smm=False)
X_tn_all.to_pickle(data("classifier", "X_tn_all_input_data.pkl"))
X_ds_all.to_pickle(data("classifier", "X_ds_all_input_data.pkl"))

tn_probs = pd.DataFrame(model_tn.predict_proba(X_tn_all), columns=model_tn.classes_, index=all_expr.index)
tn_probs["tn_pred"] = model_tn.classes_[np.argmax(tn_probs.values, axis=1)]

ds_probs = pd.DataFrame(model_ds.predict_proba(X_ds_all), columns=model_ds.classes_, index=all_expr.index)
ds_probs["ds_pred"] = model_ds.classes_[np.argmax(ds_probs.values, axis=1)]

# call: normal, or the MGUS-MM call of the cells called malignant
probs = tn_probs.join(ds_probs)
probs["pred"] = list(tn_probs["tn_pred"])
probs.loc[probs["pred"] == "tumor", "pred"] = ds_probs.loc[probs["pred"] == "tumor", "ds_pred"]
a.obs = a.obs.join(probs)

# ---- 3. mean classifier inputs per individual, per sample and per sample and isotype ---------------------------------------
a.obs["sample_disease"] = (np.array(a.obs["disease_status"]) + ":" + np.array(a.obs["sample_type"]) + ":" +
                           np.array(a.obs["annotation"]) + ":" + np.array(a.obs["redcap_id"]))
a.obs["catissue_disease"] = (np.array(a.obs["disease_status"]) + ":" + np.array(a.obs["sample_type"]) + ":" +
                             np.array(a.obs["annotation"]) + ":" + np.array(a.obs["catissue_id"]))

a.obs.join(X_ds_all.astype(float))[["sample_disease"] + list(X_ds_all.columns)].groupby("sample_disease").mean().to_pickle(
    data(*PROBABILITY, "redcap_id_mean_X_ds_scores.pkl"))
a.obs.join(X_tn_all.astype(float))[["sample_disease"] + list(X_tn_all.columns)].groupby("sample_disease").mean().to_pickle(
    data(*PROBABILITY, "redcap_id_mean_X_tn_scores.pkl"))
a.obs.join(X_ds_all.astype(float))[["catissue_disease"] + list(X_ds_all.columns)].groupby("catissue_disease").mean().to_pickle(
    data(*PROBABILITY, "catissue_id_mean_X_ds_scores.pkl"))
a.obs.join(X_tn_all.astype(float))[["catissue_disease"] + list(X_tn_all.columns)].groupby("catissue_disease").mean().to_pickle(
    data(*PROBABILITY, "catissue_id_mean_X_tn_scores.pkl"))

# isotype composition per sample
catissue_isotype = pd.crosstab(a.obs["catissue_disease"], a.obs["isotype_status"])
isotype_names = list(catissue_isotype.columns)
isotype_frac = catissue_isotype.apply(lambda x: x[isotype_names] / sum(x[isotype_names]), axis=1)
isotype_frac.rename(columns={isoname: isoname + "_frac" for isoname in isotype_names}, inplace=True)
catissue_isotype.join(isotype_frac).to_pickle(data(*PROBABILITY, "catissue_id_isotype_fraction.pkl"))

# per sample and isotype
idx = ~a.obs["isotype_status"].isna()
a.obs["catissue_isotype"] = np.nan
a.obs.loc[idx, "catissue_isotype"] = np.array(a.obs.loc[idx, "catissue_disease"]) + "--" + np.array(a.obs.loc[idx, "isotype_status"])
a.obs.join(X_ds_all.astype(float))[["catissue_isotype"] + list(X_ds_all.columns)].groupby("catissue_isotype").mean().to_pickle(
    data(*PROBABILITY, "catissue_isotype_mean_X_ds_scores.pkl"))
a.obs.join(X_tn_all.astype(float))[["catissue_isotype"] + list(X_tn_all.columns)].groupby("catissue_isotype").mean().to_pickle(
    data(*PROBABILITY, "catissue_isotype_mean_X_tn_scores.pkl"))

scores = probs.rename(columns={"tumor": "Malignancy Score", "MM": "scPC-MM Score"})
a.obs = a.obs.join(scores[["Malignancy Score", "scPC-MM Score"]])
a.obs[["catissue_isotype", "Malignancy Score", "scPC-MM Score"]].groupby("catissue_isotype").mean().to_pickle(
    data(*PROBABILITY, "catissue_isotype_mean_probability.pkl"))

# ---- 4. mean scores per individual and per sample, and tests ---------------------------------------------------------------
redcap_probs = a.obs[["sample_disease", "tumor", "MM"]].groupby("sample_disease").mean()
catissue_probs = a.obs[["catissue_disease", "tumor", "MM"]].groupby("catissue_disease").mean()


def process_mean_probs(df):
    """split the group label into disease stage, tissue, PC status and individual (or sample)"""
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


def compute_at_once(df):
    res = {}

    # disease stages, malignant BM PCs
    bm_tumors = df[(df["annotation"] == "tumor") & (df["sample_type"] == "BM")].copy()
    res["pT|BM:tumor|disease_status"] = categ_cross_wilcox(bm_tumors, "disease_status", "tumor")
    res["pMM|BM:tumor|disease_status"] = categ_cross_wilcox(bm_tumors, "disease_status", "MM")

    # disease stages, normal BM PCs
    bm_normals = df[(df["annotation"] == "normal") & (df["sample_type"] == "BM")].copy()
    res["pT|BM:normal|disease_status"] = categ_cross_wilcox(bm_normals, "disease_status", "tumor")
    res["pMM|BM:normal|disease_status"] = categ_cross_wilcox(bm_normals, "disease_status", "MM")

    # malignant vs normal BM PCs
    res["pT|BM|TvN"] = categ_pair_wilcox(bm_normals, bm_tumors, "disease_status", "sample", "tumor")
    res["pMM|BM|TvN"] = categ_pair_wilcox(bm_normals, bm_tumors, "disease_status", "sample", "MM")

    # normal PCs, BM vs PB
    pb_normals = df[(df["annotation"] == "normal") & (df["sample_type"] == "PB")]
    res["pT|normal|BMvPB"] = categ_pair_wilcox(bm_normals, pb_normals, "disease_status", "sample", "tumor")
    res["pMM|normal|BMvPB"] = categ_pair_wilcox(bm_normals, pb_normals, "disease_status", "sample", "MM")

    # malignant PCs, BM vs PB (paired)
    pb_tumors = df[(df["annotation"] == "tumor") & (df["sample_type"] == "PB")]
    res["pT|tumor|BMvPB"] = categ_pair_wilcox(bm_tumors, pb_tumors, "disease_status", "sample", "tumor", is_paired=True)
    res["pMM|tumor|BMvPB"] = categ_pair_wilcox(bm_tumors, pb_tumors, "disease_status", "sample", "MM", is_paired=True)

    # malignant vs normal PB PCs
    res["pT|PB|TvN"] = categ_pair_wilcox(pb_normals, pb_tumors, "disease_status", "sample", "tumor")
    res["pMM|PB|TvN"] = categ_pair_wilcox(pb_normals, pb_tumors, "disease_status", "sample", "MM")
    return res


redcap_probs = process_mean_probs(redcap_probs)
catissue_probs = process_mean_probs(catissue_probs)
redcap_stats = compute_at_once(redcap_probs)

# majority cytogenetic group of the malignant PCs of each individual x disease stage x tissue
redcap_probs["cytogenetics"] = np.nan
for redcap_group in redcap_probs[redcap_probs["annotation"] == "tumor"].index:
    df = pd.DataFrame(a.obs.loc[a.obs["sample_disease"] == redcap_group, "cytogenetics"].value_counts())
    cyto = list(df.head(n=1).index)[0]
    redcap_probs.loc[redcap_group, "cytogenetics"] = cyto
redcap_probs["cytogenetics"] = pd.Categorical(redcap_probs["cytogenetics"],
                                              categories=["t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)", "t(12;14)",
                                                          "t(14;16)", "t(14;20)", "HRD", "Other"])

redcap_probs.to_csv(data(*PROBABILITY, "redcap_id_mean_probability.csv"))
catissue_probs.to_csv(data(*PROBABILITY, "catissue_id_mean_probability.csv"))
with open(data(*PROBABILITY, "redcap_id_stats.pkl"), "wb") as fo:
    pickle.dump(redcap_stats, fo)
probs.to_pickle(data(*PROBABILITY, "cell-level-prediction-knn30.pkl"))
print("done")
