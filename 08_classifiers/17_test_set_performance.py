#!/usr/bin/env python3
"""Performance of the two-tier classifier trained on the training set and applied to the test set, with kNN smoothing
over 30 and 50 neighbors (Fig. 4B,C; Results and Methods, classifier performance); also writes the unsmoothed per-cell
feature table of all PCs.

Purpose:      Training cells of donors, MGUS and MM patients (SMM removed). Malignancy tier: z-scored mean expression of
              the 16 signatures and senescence score + partner genes (100 trees, max_depth 30, min_samples_split 20,
              min_samples_leaf 30, balanced_subsample); MGUS-MM tier: AUCell scores of the 16 signatures + CNV-arm scores
              quantile-normalized (seed 42) (200 trees, max_depth 25, min_samples_split 20, min_samples_leaf 10, balanced),
              trained on the malignant training cells; the test cells called malignant get the MGUS-MM call. The features
              of each set are z-scored and quantile-normalized within that set. Accuracy overall and per class (normal,
              MGUS, MM) on the training cells (fit) and on the test cells, with the cohort-smoothed tables at k = 30 (the
              training table without k in its name, the test table with k = 30) and at k = 50. The forests get no
              random_state: they draw from numpy's global random state, seeded with 42 once, k = 30 first. Run with
              PYTHONHASHSEED=0.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl,
              sc-test-cohort-smoothed-knn30-expr_47samples.pkl, sc-train-cohort-smoothed-knn50-expr_134samples.pkl,
              sc-test-cohort-smoothed-knn50-expr_47samples.pkl: feature columns, annotation, disease_status, redcap_id.
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt, auc_16_gene_modules.txt, auc_1_aging_signature.txt, mean_expr_fc_driver_cnv_arms.txt.
              data/signature/score/cell-level-copy-number-full-gene-expr.pkl, cell-level-cytogenetics-gene-expr.pkl,
              cell-level-senescence-expr-auc.pkl, cell-level-module-gene-expr-auc.pkl: unsmoothed per-cell features.
              data/h5/agg/plasma-cell-GEX5-counts-250207.h5ad: cell names of all PCs (obs).

Outputs:      results/classifiers/final_model_test_performance.tsv: k, set (training fit or test), cells, accuracy and
              accuracy per class.
              data/signature/score/agg-cell-level-features-raw.pkl: unsmoothed features (mean expression and AUCell of the
              16 signatures, SenMayo, partner genes, CNV-arm scores) of all PCs.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata, scikit-learn 1.0.2); config.py. Other
              scikit-learn versions grow different trees from the same random state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data, result_dir

import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)

DATASETS = ("classifier", "datasets")
OUT = result_dir("classifiers")


def load_list(name):
    return [x.rstrip() for x in open(data(*DATASETS, "feature_columns", name))]


def compute_zscore(df):
    ss = StandardScaler()
    ss.fit(df)
    return pd.DataFrame(ss.transform(df), index=df.index, columns=df.columns)


def quantile_transform(df):
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(df)
    return pd.DataFrame(qt.transform(df), index=df.index, columns=df.columns)


def format_data(expr):
    df = expr.copy()
    df = df[~df["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
    df["class"] = df["disease_status"]
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
    return y_tn, y_ds, g, X_tn.values, X_ds.values, tn_feats, ds_feats


def predict_flags(model_tn, model_ds, X_tn, X_ds, y_tn, y_ds):
    """call normal, or the MGUS-MM call of the cells called malignant; truth: normal, or the stage of malignant cells"""
    y_pred = model_tn.predict(X_tn)
    y_pred_ds = model_ds.predict(X_ds)
    y_pred[y_pred == "tumor"] = y_pred_ds[y_pred == "tumor"]

    y_truth = np.array(y_tn)
    y_truth_ds = np.array(y_ds)
    y_truth[y_truth == "tumor"] = y_truth_ds[y_truth == "tumor"]
    return y_truth, y_pred


def compute_performance(y_truth, y_pred):
    """accuracy overall and per class (share of the cells of the class called correctly)"""
    pi = pd.DataFrame({"label": y_truth, "pred": y_pred})
    accuracy = sum(pi["label"] == pi["pred"]) / pi.shape[0]
    n_normal = sum(pi["label"] == "normal")
    acc_normal = sum((pi["label"] == pi["pred"]) & (pi["label"] == "normal")) / max(n_normal, 1)
    n_MGUS = sum(pi["label"] == "MGUS")
    acc_MGUS = sum((pi["label"] == pi["pred"]) & (pi["label"] == "MGUS")) / max(n_MGUS, 1)
    n_MM = sum(pi["label"] == "MM")
    acc_MM = sum((pi["label"] == pi["pred"]) & (pi["label"] == "MM")) / max(n_MM, 1)
    return {"cells": pi.shape[0], "accuracy": accuracy, "acc_normal": acc_normal, "acc_MGUS": acc_MGUS, "acc_MM": acc_MM}


tn_mex_gm16_age_cyto_params = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 30, "min_samples_split": 20,
                               "min_samples_leaf": 30, "bootstrap": True, "class_weight": "balanced_subsample",
                               "n_jobs": -1}
ds_auc_gm16_arm_params = {"n_estimators": 200, "max_features": "sqrt", "max_depth": 25, "min_samples_split": 20,
                          "min_samples_leaf": 10, "bootstrap": True, "class_weight": "balanced", "n_jobs": -1}

# ---- 1. training on the training set, evaluation on the training and test sets, k = 30 then k = 50 ------------------------
rows = []
for k, train_name, test_name in ((30, "sc-train-cohort-smoothed-expr_134samples.pkl",
                                  "sc-test-cohort-smoothed-knn30-expr_47samples.pkl"),
                                 (50, "sc-train-cohort-smoothed-knn50-expr_134samples.pkl",
                                  "sc-test-cohort-smoothed-knn50-expr_47samples.pkl")):
    train_expr = pd.read_pickle(data(*DATASETS, train_name))
    y_tn_train, y_ds_train, _, X_tn_train, X_ds_train, tn_feats, ds_feats = format_data(train_expr)

    # malignancy tier
    model_tn = RandomForestClassifier(**tn_mex_gm16_age_cyto_params)
    model_tn.fit(X_tn_train, y_tn_train)

    # MGUS-MM tier on the malignant training cells
    model_ds = RandomForestClassifier(**ds_auc_gm16_arm_params)
    X_ds_train_ = X_ds_train[y_tn_train == "tumor"]
    y_ds_train_ = pd.Categorical(list(y_ds_train[y_tn_train == "tumor"]), categories=["MGUS", "MM"])
    model_ds.fit(X_ds_train_, y_ds_train_)

    y_truth, y_pred = predict_flags(model_tn, model_ds, X_tn_train, X_ds_train, y_tn_train, y_ds_train)
    rows.append(dict(k=k, set="training (fit)", **compute_performance(y_truth, y_pred)))

    test_expr = pd.read_pickle(data(*DATASETS, test_name))
    y_tn_test, y_ds_test, _, X_tn_test, X_ds_test, _, _ = format_data(test_expr)
    y_truth, y_pred = predict_flags(model_tn, model_ds, X_tn_test, X_ds_test, y_tn_test, y_ds_test)
    rows.append(dict(k=k, set="test", **compute_performance(y_truth, y_pred)))

perf = pd.DataFrame(rows)
perf.to_csv(os.path.join(OUT, "final_model_test_performance.tsv"), sep="\t", index=False)
print(perf.to_string(index=False))

# ---- 2. unsmoothed per-cell features of all PCs ----------------------------------------------------------------------------
mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
cyto_genes = load_list("cytogenetics_genes.txt")
auc_gene_modules = load_list("auc_16_gene_modules.txt")
auc_aging_sigs = load_list("auc_1_aging_signature.txt")
cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

cnv_scores = pd.read_pickle(data("signature", "score", "cell-level-copy-number-full-gene-expr.pkl"))[cnv_arms]
cyto_expr = pd.read_pickle(data("signature", "score", "cell-level-cytogenetics-gene-expr.pkl"))[cyto_genes]
aging_expr = pd.read_pickle(data("signature", "score", "cell-level-senescence-expr-auc.pkl"))[mean_aging_sigs + auc_aging_sigs]
module_expr = pd.read_pickle(data("signature", "score", "cell-level-module-gene-expr-auc.pkl"))[mean_gene_modules + auc_gene_modules]

a = sc.read(data("h5", "agg", "plasma-cell-GEX5-counts-250207.h5ad"), backed="r")
expr = module_expr.join(aging_expr).join(cyto_expr).join(cnv_scores)
expr.loc[a.obs.index].to_pickle(data("signature", "score", "agg-cell-level-features-raw.pkl"))
print("done")
