#!/usr/bin/env python3
"""Grid search of the random forest parameters of each tier on its own: the MGUS-MM tier and three versions of the
malignancy tier (Methods, classifier development; Extended Data Fig. 6D).

Purpose:      Training cells of donors, MGUS and MM patients (SMM removed), cohort-smoothed features. MGUS-MM tier
              (part-cohort-DS-auc-gm16-arm): malignant PCs only, MGUS vs MM, AUCell scores of the 16 signatures + CNV-arm
              scores quantile-normalized (seed 42) on these cells. Malignancy tier, malignant vs normal PCs, all cells:
              AUCell scores of the 16 signatures and senescence + CNV-arm scores + z-scored partner genes
              (part-cohort-TN-auc-gm16-age-arm-cyto); z-scored mean expression of the 16 signatures and senescence and
              partner genes + CNV-arm scores (part-cohort-TN-mex-gm16-age-arm-cyto); the same without CNV-arm scores
              (part-cohort-TN-mex-gm16-age-cyto). Grid: n_estimators 100-500, max_depth 10-35, min_samples_split 20-50,
              min_samples_leaf 10-50, class weights balanced or balanced_subsample (max_features sqrt, bootstrap): 1,200
              settings, each scored by a 5-fold cross-validation grouped by individual (StratifiedGroupKFold, shuffled,
              seed 42): accuracy (and MM and MGUS accuracy for the MGUS-MM tier), mean and SD (numpy) over the folds. The
              forests get no random_state: they draw from numpy's global random state, which is seeded with 42 before each
              search. Settings kept for the final classifiers: MGUS-MM tier 200 trees, max_depth 25, min_samples_split 20,
              min_samples_leaf 10, balanced; malignancy tier (mex:gm16+age+cyto) 100 trees, max_depth 30,
              min_samples_split 20, min_samples_leaf 30, balanced_subsample. Run with PYTHONHASHSEED=0.
              Usage: python3 08_grid_search_tiers.py [model ...] (default: the four searches in the order above).

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl: feature columns, annotation,
              disease_status, redcap_id (individual, used only to group the folds).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              auc_16_gene_modules.txt, auc_1_aging_signature.txt, cytogenetics_genes.txt, mean_expr_fc_driver_cnv_arms.txt.

Outputs:      data/classifier/performance/grid/<model>.csv: one row per setting (parameters, mean_acc, std_acc; MGUS-MM
              tier also MM_mean_acc, MM_std_acc, MGUS_mean_acc, MGUS_std_acc).

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

MODELS = ["part-cohort-DS-auc-gm16-arm", "part-cohort-TN-auc-gm16-age-arm-cyto", "part-cohort-TN-mex-gm16-age-arm-cyto",
          "part-cohort-TN-mex-gm16-age-cyto"]
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
    """label, individual and feature matrix of one tier"""
    df = expr.copy()
    df = df[~df["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]

    auc_gene_modules = load_list("auc_16_gene_modules.txt")
    cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

    if model == "part-cohort-DS-auc-gm16-arm":                  # MGUS vs MM, malignant PCs only
        df["class"] = df["disease_status"]
        df = df[df["annotation"] == "tumor"]
        X = df[auc_gene_modules].join(quantile_transform(df[cnv_arms]))
        y = pd.Categorical(list(df["class"]), categories=["MGUS", "MM"])
    else:                                                       # malignant vs normal, all PCs
        mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
        mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
        auc_aging_sigs = load_list("auc_1_aging_signature.txt")
        cyto_genes = load_list("cytogenetics_genes.txt")
        if model == "part-cohort-TN-auc-gm16-age-arm-cyto":
            X = df[auc_gene_modules + auc_aging_sigs].join(quantile_transform(df[cnv_arms])).join(compute_zscore(df[cyto_genes]))
        elif model == "part-cohort-TN-mex-gm16-age-arm-cyto":
            X = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes]).join(quantile_transform(df[cnv_arms]))
        else:                                                   # part-cohort-TN-mex-gm16-age-cyto
            X = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes])
        y = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
    g = pd.Categorical(list(df["redcap_id"]), categories=sorted(set(df["redcap_id"])))
    return y, g, X.values


def grid_search(y, g, X, classes, seed=SEED):
    """mean and SD over the folds of the accuracy, overall and for each class in `classes`"""
    pred_res = {k: [] for k in PARAM_NAMES + ["mean_acc", "std_acc"] +
                [c + s for c in classes for s in ("_mean_acc", "_std_acc")]}
    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    idx_set = list(cv.split(X, y, groups=g))
    for params in list(ParameterGrid(PARAM_GRID)):
        acc, acc_by_class = [], {c: [] for c in classes}
        for train_idx, test_idx in idx_set:
            model = RandomForestClassifier(**params)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            pi = pd.DataFrame({"label": y[test_idx], "pred": y_pred})
            acc.append(sum(pi["label"] == pi["pred"]) / pi.shape[0])
            for c in classes:
                n_c = sum(pi["label"] == c)
                acc_by_class[c].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == c)) / max(n_c, 1))
        for p in PARAM_NAMES:
            pred_res[p].append(params[p])
        pred_res["mean_acc"].append(np.mean(acc))
        pred_res["std_acc"].append(np.std(acc))
        for c in classes:
            pred_res[c + "_mean_acc"].append(np.mean(acc_by_class[c]))
            pred_res[c + "_std_acc"].append(np.std(acc_by_class[c]))
    return pd.DataFrame(pred_res)


parser = argparse.ArgumentParser(description="grid search of each classifier tier")
parser.add_argument("model", nargs="*", help="searches to run (default: all four)")
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
    y, g, X = format_data(train_expr, model)
    classes = ["MM", "MGUS"] if model == "part-cohort-DS-auc-gm16-arm" else []
    pred_res = grid_search(y, g, X, classes)
    pred_res.to_csv(out_dir / (model + ".csv"))
    best = pred_res.sort_values("mean_acc", ascending=False).iloc[0]
    print("%s: best mean accuracy %.4f" % (model, best["mean_acc"]), flush=True)
print("done")
