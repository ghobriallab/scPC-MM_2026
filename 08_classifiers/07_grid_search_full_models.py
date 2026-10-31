#!/usr/bin/env python3
"""Grid search of the random forest parameters for five candidate models: two one-step three-class models and three
two-tier models (Methods, classifier development).

Purpose:      Training cells of donors, MGUS and MM patients (SMM removed), cohort-smoothed features. One-step models: one
              random forest for normal PC / MGUS malignant PC / MM malignant PC on the AUCell scores of the 16 signatures,
              without (multi_cohort-knn_auc-gm16) or with (multi_cohort-knn_auc-gm16-age) the senescence score. Two-tier
              models: a malignant vs normal forest, then an MGUS vs MM forest trained on the malignant training cells and
              applied to the cells the first one calls malignant; MGUS-MM tier = AUCell scores of the 16 signatures + CNV-arm
              scores (quantile-normalized, seed 42); malignancy tier = mean expression of the 16 signatures and senescence
              score + partner genes (z-scored) + CNV-arm scores (cohort-TN-mex-gm16-age-arm-cyto_DS-auc-gm16-arm), AUCell
              scores of the 16 signatures and senescence + CNV-arm scores + z-scored partner genes
              (cohort-TN-auc-gm16-age-arm-cyto_DS-auc-gm16-arm), or the first one without CNV-arm scores
              (cohort-TN-mex-gm16-age-cyto_DS-auc-gm16-arm). Grid: n_estimators 100-500, max_depth 10-35,
              min_samples_split 20-50, min_samples_leaf 10-50, class weights balanced or balanced_subsample (max_features
              sqrt, bootstrap): 1,200 settings, each scored by a 5-fold cross-validation grouped by individual
              (StratifiedGroupKFold, shuffled, seed 42): accuracy overall and per class (normal, MGUS, MM), mean and SD
              (numpy) over the folds. The forests get no random_state: they draw from numpy's global random state, which is
              seeded with 42 before each model's search. Run with PYTHONHASHSEED=0.
              Usage: python3 07_grid_search_full_models.py [model ...] (default: the five models in the order above).

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl: feature columns, annotation,
              disease_status, redcap_id (individual, used only to group the folds).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              auc_16_gene_modules.txt, auc_1_aging_signature.txt, cytogenetics_genes.txt, mean_expr_fc_driver_cnv_arms.txt.

Outputs:      data/classifier/performance/grid/<model>.csv: one row per setting (parameters, mean_acc, std_acc,
              MM_mean_acc, MM_std_acc, MGUS_mean_acc, MGUS_std_acc, normal_mean_acc, normal_std_acc).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn 1.0.2); config.py. Other scikit-learn versions
              grow different trees from the same random state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

import argparse
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

MODELS = ["multi_cohort-knn_auc-gm16", "multi_cohort-knn_auc-gm16-age", "cohort-TN-mex-gm16-age-arm-cyto_DS-auc-gm16-arm",
          "cohort-TN-auc-gm16-age-arm-cyto_DS-auc-gm16-arm", "cohort-TN-mex-gm16-age-cyto_DS-auc-gm16-arm"]
PARAM_GRID = {"n_estimators": [100, 200, 300, 400, 500],
              "max_features": ["sqrt"],
              "max_depth": [10, 15, 20, 25, 30, 35],
              "min_samples_split": [20, 30, 40, 50],
              "min_samples_leaf": [10, 20, 30, 40, 50],
              "bootstrap": [True],
              "class_weight": ["balanced", "balanced_subsample"],
              "n_jobs": [-1]}
PARAM_NAMES = ["n_estimators", "max_features", "max_depth", "min_samples_split", "min_samples_leaf", "bootstrap",
               "class_weight"]
CLASSES = ["MM", "MGUS", "normal"]


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


def format_data(expr, model):
    """labels, individual and feature matrices of one candidate model"""
    df = expr.copy()
    df = df[~df["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
    df["class"] = df["disease_status"]
    df.loc[df["annotation"] == "normal", "class"] = "normal"
    g = pd.Categorical(list(df["redcap_id"]), categories=sorted(set(df["redcap_id"])))

    mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
    mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
    auc_gene_modules = load_list("auc_16_gene_modules.txt")
    auc_aging_sigs = load_list("auc_1_aging_signature.txt")
    cyto_genes = load_list("cytogenetics_genes.txt")
    cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

    if model.startswith("multi_"):                              # one three-class forest
        X = df[auc_gene_modules + (auc_aging_sigs if model.endswith("-age") else [])]
        y = pd.Categorical(list(df["class"]), categories=["normal", "MGUS", "MM"])
        return y, g, X.values

    if model == "cohort-TN-mex-gm16-age-arm-cyto_DS-auc-gm16-arm":
        X_cnv_arms = quantile_transform(df[cnv_arms])
        X_tn = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes]).join(X_cnv_arms)
        X_ds = df[auc_gene_modules].join(X_cnv_arms)
    elif model == "cohort-TN-auc-gm16-age-arm-cyto_DS-auc-gm16-arm":
        X_cnv_arms = quantile_transform(df[cnv_arms])
        X_tn = df[auc_gene_modules + auc_aging_sigs].join(X_cnv_arms).join(compute_zscore(df[cyto_genes]))
        X_ds = df[auc_gene_modules].join(X_cnv_arms)
    else:                                                       # cohort-TN-mex-gm16-age-cyto_DS-auc-gm16-arm
        X_tn = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes])
        X_ds = df[auc_gene_modules].join(quantile_transform(df[cnv_arms]))
    y_tn = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
    y_ds = np.array(df["class"])
    return y_tn, y_ds, g, X_tn.values, X_ds.values


def fold_accuracy(label, pred, acc, acc_by_class):
    """accuracy overall and per class (share of the cells of the class that are called correctly) of one fold"""
    pi = pd.DataFrame({"label": label, "pred": pred})
    acc.append(sum(pi["label"] == pi["pred"]) / pi.shape[0])
    for c in ["normal", "MGUS", "MM"]:
        n_c = sum(pi["label"] == c)
        acc_by_class[c].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == c)) / max(n_c, 1))


def record(pred_res, params, acc, acc_by_class):
    for p in PARAM_NAMES:
        pred_res[p].append(params[p])
    pred_res["mean_acc"].append(np.mean(acc))
    pred_res["std_acc"].append(np.std(acc))
    for c in CLASSES:
        pred_res[c + "_mean_acc"].append(np.mean(acc_by_class[c]))
        pred_res[c + "_std_acc"].append(np.std(acc_by_class[c]))


def empty_results():
    return {k: [] for k in PARAM_NAMES + ["mean_acc", "std_acc"] + [c + s for c in CLASSES for s in ("_mean_acc", "_std_acc")]}


def grid_search_multilabel(y, g, X, seed=SEED):
    pred_res = empty_results()
    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    idx_set = list(cv.split(X, y, groups=g))
    for params in list(ParameterGrid(PARAM_GRID)):
        acc, acc_by_class = [], {"normal": [], "MGUS": [], "MM": []}
        for train_idx, test_idx in idx_set:
            model = RandomForestClassifier(**params)
            model.fit(X[train_idx], y[train_idx])
            fold_accuracy(y[test_idx], model.predict(X[test_idx]), acc, acc_by_class)
        record(pred_res, params, acc, acc_by_class)
    return pd.DataFrame(pred_res)


def grid_search_two_tier(y_tn, y_ds, g, X_tn, X_ds, seed=SEED):
    pred_res = empty_results()
    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    idx_set = list(cv.split(X_tn, y_tn, groups=g))
    for params in list(ParameterGrid(PARAM_GRID)):
        acc, acc_by_class = [], {"normal": [], "MGUS": [], "MM": []}
        for train_idx, test_idx in idx_set:
            # malignant vs normal forest
            model_tn = RandomForestClassifier(**params)
            Xt_tn = X_tn[train_idx].copy()
            yt_tn = y_tn[train_idx].copy()
            model_tn.fit(Xt_tn, yt_tn)

            # MGUS vs MM forest on the malignant training cells
            model_ds = RandomForestClassifier(**params)
            Xt_ds = X_ds[train_idx].copy()
            yt_ds = y_ds[train_idx].copy()
            Xt_ds = Xt_ds[yt_tn == "tumor"]
            yt_ds = pd.Categorical(list(yt_ds[yt_tn == "tumor"]), categories=["MGUS", "MM"])
            model_ds.fit(Xt_ds, yt_ds)

            # cells called malignant get the MGUS-MM call
            y_pred = model_tn.predict(X_tn[test_idx])
            y_pred_ds = model_ds.predict(X_ds[test_idx])
            y_pred[y_pred == "tumor"] = y_pred_ds[y_pred == "tumor"]

            # truth: normal, or the disease stage of malignant cells
            y_truth = np.array(y_tn[test_idx])
            y_truth_ds = np.array(y_ds[test_idx])
            y_truth[y_truth == "tumor"] = y_truth_ds[y_truth == "tumor"]
            fold_accuracy(y_truth, y_pred, acc, acc_by_class)
        record(pred_res, params, acc, acc_by_class)
    return pd.DataFrame(pred_res)


parser = argparse.ArgumentParser(description="grid search of the candidate models")
parser.add_argument("model", nargs="*", help="models to search (default: all five)")
args = parser.parse_args()
models = args.model or MODELS
if any(m not in MODELS for m in models):
    parser.error("models: " + ", ".join(MODELS))

train_expr = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl"))
out_dir = data("classifier", "performance", "grid")
out_dir.mkdir(parents=True, exist_ok=True)

for model in models:
    random.seed(SEED)                                           # each search starts from the same global random state
    np.random.seed(SEED)
    if model.startswith("multi_"):
        y, g, X = format_data(train_expr, model)
        pred_res = grid_search_multilabel(y, g, X)
    else:
        y_tn, y_ds, g, X_tn, X_ds = format_data(train_expr, model)
        pred_res = grid_search_two_tier(y_tn, y_ds, g, X_tn, X_ds)
    pred_res.to_csv(out_dir / (model + ".csv"))
    best = pred_res.sort_values("mean_acc", ascending=False).iloc[0]
    print("%s: best mean accuracy %.4f" % (model, best["mean_acc"]), flush=True)
print("done")
