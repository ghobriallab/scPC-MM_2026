#!/usr/bin/env python3
"""Cross-validated performance of the final two-tier classifier for k = 0 (no smoothing) to 100 neighbors of the kNN
smoothing (Extended Data Fig. 6C; Methods, classifier development).

Purpose:      Training cells of donors, MGUS and MM patients (SMM removed). Malignancy tier: z-scored mean expression of
              the 16 signatures and senescence score + partner genes (100 trees, max_depth 30, min_samples_split 20,
              min_samples_leaf 30, balanced_subsample); MGUS-MM tier: AUCell scores of the 16 signatures + CNV-arm scores
              quantile-normalized (seed 42) (200 trees, max_depth 25, min_samples_split 20, min_samples_leaf 10, balanced),
              trained on the malignant training cells and applied to the cells called malignant. For each k, a 5-fold
              cross-validation grouped by individual (StratifiedGroupKFold, shuffled, seed 42): accuracy overall and per
              class (normal, MGUS, MM); precision, recall and F1 of the malignancy call (malignant = positive) and of the
              MGUS-MM call in malignant cells (MM = positive); macro and micro precision, recall and F1 over the three
              classes; mean and SD (pandas, n - 1) over the folds. The forests get no random_state: they draw from numpy's
              global random state, seeded with 42 once, so the order of the runs matters: the cohort-smoothed training
              table without k in its name is evaluated first (result printed only), then k = 0, 10, ..., 100 in this
              order. Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl, sc-train-expr_134samples.pkl (k = 0),
              sc-train-cohort-smoothed-knn<k>-expr_134samples.pkl (02_knn_smoothing_train_test.py): feature columns,
              annotation, disease_status, redcap_id (individual, used only to group the folds).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt, auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt.

Outputs:      data/classifier/kNN_param_accuracy.csv: per k, mean and SD of every metric.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn 1.0.2); config.py. Other scikit-learn versions
              grow different trees from the same random state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)

DATASETS = ("classifier", "datasets")
# tuned parameters of the two tiers
TN_PARAMS = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 30, "min_samples_split": 20,
             "min_samples_leaf": 30, "bootstrap": True, "class_weight": "balanced_subsample", "n_jobs": -1}
DS_PARAMS = {"n_estimators": 200, "max_features": "sqrt", "max_depth": 25, "min_samples_split": 20,
             "min_samples_leaf": 10, "bootstrap": True, "class_weight": "balanced", "n_jobs": -1}
# output columns: (mean column, SD column, per-fold metric)
SUMMARY = ([("mean_acc", "std_acc", "accuracy"), ("MM_mean_acc", "MM_std_acc", "acc_MM"),
            ("MGUS_mean_acc", "MGUS_std_acc", "acc_MGUS"), ("normal_mean_acc", "normal_std_acc", "acc_normal")] +
           [(m + "_mean", m + "_std", m) for m in ("tn_precision", "tn_recall", "tn_f1", "ds_precision", "ds_recall",
                                                   "ds_f1", "macro_precision", "macro_recall", "macro_f1",
                                                   "micro_precision", "micro_recall", "micro_f1")])


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
    return y_tn, y_ds, g, X_tn.values, X_ds.values


def check_performance(y_tn, y_ds, g, X_tn, X_ds, tn_params, ds_params, seed=SEED):
    """per-fold metrics of the two-tier classifier"""
    res = {"fold": [], "accuracy": [], "acc_MM": [], "acc_MGUS": [], "acc_normal": [],
           "tn_precision": [], "tn_recall": [], "tn_f1": [],
           "ds_precision": [], "ds_recall": [], "ds_f1": [],
           "macro_precision": [], "macro_recall": [], "macro_f1": [],
           "micro_precision": [], "micro_recall": [], "micro_f1": []}

    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(cv.split(X_tn, y_tn, groups=g)):
        # malignancy tier
        model_tn = RandomForestClassifier(**tn_params)
        Xt_tn = X_tn[train_idx].copy()
        yt_tn = y_tn[train_idx].copy()
        model_tn.fit(Xt_tn, yt_tn)

        # MGUS-MM tier on the malignant training cells
        model_ds = RandomForestClassifier(**ds_params)
        Xt_ds = X_ds[train_idx].copy()
        yt_ds = y_ds[train_idx].copy()
        Xt_ds = Xt_ds[yt_tn == "tumor"]
        yt_ds = pd.Categorical(list(yt_ds[yt_tn == "tumor"]), categories=["MGUS", "MM"])
        model_ds.fit(Xt_ds, yt_ds)

        # cells called malignant get the MGUS-MM call
        y_pred = model_tn.predict(X_tn[test_idx])
        y_pred_ds = model_ds.predict(X_ds[test_idx])
        y_pred[y_pred == "tumor"] = y_pred_ds[y_pred == "tumor"]

        y_truth = np.array(y_tn[test_idx])
        y_truth_ds = np.array(y_ds[test_idx])

        # malignancy call (malignant = 1) and MGUS-MM call in the malignant test cells (MM = 1)
        tn_precision, tn_recall, tn_f1, _ = precision_recall_fscore_support(
            [0 if p == "normal" else 1 for p in y_tn[test_idx]], [0 if p == "normal" else 1 for p in y_pred],
            average="binary")
        res["tn_precision"].append(tn_precision)
        res["tn_recall"].append(tn_recall)
        res["tn_f1"].append(tn_f1)

        ds_precision, ds_recall, ds_f1, _ = precision_recall_fscore_support(
            [1 if p == "MM" else 0 for p in y_truth_ds[y_truth == "tumor"]],
            [1 if p == "MM" else 0 for p in y_pred_ds[y_truth == "tumor"]], average="binary")
        res["ds_precision"].append(ds_precision)
        res["ds_recall"].append(ds_recall)
        res["ds_f1"].append(ds_f1)

        # three classes: normal, or the disease stage of malignant cells
        y_truth[y_truth == "tumor"] = y_truth_ds[y_truth == "tumor"]
        pi = pd.DataFrame({"label": y_truth, "pred": y_pred})

        y_truth = pd.Categorical(y_truth, categories=["normal", "MGUS", "MM"])
        y_pred = pd.Categorical(y_pred, categories=["normal", "MGUS", "MM"])
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_truth, y_pred, average="macro")
        res["macro_precision"].append(macro_precision)
        res["macro_recall"].append(macro_recall)
        res["macro_f1"].append(macro_f1)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_truth, y_pred, average="micro")
        res["micro_precision"].append(micro_precision)
        res["micro_recall"].append(micro_recall)
        res["micro_f1"].append(micro_f1)

        res["fold"].append(i)
        res["accuracy"].append(sum(pi["label"] == pi["pred"]) / pi.shape[0])
        n_normal = sum(pi["label"] == "normal")
        res["acc_normal"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "normal")) / max(n_normal, 1))
        n_MGUS = sum(pi["label"] == "MGUS")
        res["acc_MGUS"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "MGUS")) / max(n_MGUS, 1))
        n_MM = sum(pi["label"] == "MM")
        res["acc_MM"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "MM")) / max(n_MM, 1))
    return pd.DataFrame(res)


# ---- smoothed training table without k in its name (evaluated first, see Purpose) ---------------------------------------
train_expr = pd.read_pickle(data(*DATASETS, "sc-train-cohort-smoothed-expr_134samples.pkl"))
res = check_performance(*format_data(train_expr), TN_PARAMS, DS_PARAMS, seed=SEED)
print("smoothed training table: accuracy %.4f (%.4f)" % (res["accuracy"].mean(), res["accuracy"].std()), flush=True)

# ---- k = 0 (unsmoothed) and k = 10, ..., 100 ------------------------------------------------------------------------------
k_param_res = {"k": []}
k_param_res.update({col: [] for mean_col, sd_col, _ in SUMMARY for col in (mean_col, sd_col)})
for k in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    if k == 0:
        path = data(*DATASETS, "sc-train-expr_134samples.pkl")
    else:
        path = data(*DATASETS, "sc-train-cohort-smoothed-knn{}-expr_134samples.pkl".format(k))

    expr = pd.read_pickle(path)
    res = check_performance(*format_data(expr), TN_PARAMS, DS_PARAMS, seed=SEED)

    k_param_res["k"].append(k)
    for mean_col, sd_col, metric in SUMMARY:
        k_param_res[mean_col].append(res[metric].mean())
        k_param_res[sd_col].append(res[metric].std())
    print("k = %d: accuracy %.4f (%.4f)" % (k, res["accuracy"].mean(), res["accuracy"].std()), flush=True)

pd.DataFrame(k_param_res).to_csv(data("classifier", "kNN_param_accuracy.csv"))
print("done")
