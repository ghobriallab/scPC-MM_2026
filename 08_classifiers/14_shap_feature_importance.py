#!/usr/bin/env python3
"""Mean absolute SHAP value of each feature of the two classifier tiers over the cross-validation folds, and
cross-validated performance of feature subsets chosen from it (feature ranking used by the feature ablation,
Extended Data Fig. 6E).

Purpose:      Collects the SHAP values of the held-out cells of the five folds (12_shap_values_malignancy.py,
              13_shap_values_mst.py; class tumor for the malignancy tier, class MM for the MGUS-MM tier), takes the mean
              absolute SHAP value of each feature over all held-out cells, and ranks the features with their cumulative
              share of the total. Then the nine translocation partner genes (malignancy tier) and the five CNV-arm scores
              (MGUS-MM tier) are each merged into one group entry carrying the summed value, and two-tier classifiers are
              cross-validated on feature subsets taken from these group rankings: features with mean |SHAP| > 0.01, and the
              top features up to 90% or 95% of the cumulative share, with or without the next feature, for both tiers and
              mixed. Subset features are matched to the classifier columns by name. Same training cells, features,
              parameters and folds as 10_knn_neighbors_accuracy.py (StratifiedGroupKFold, shuffled, seed 42); the forests
              get no random_state and draw from numpy's global random state, seeded with 42 once, subsets in the order of
              the output table. Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/feature_importance/shap/model-tn-shap-value-5cv-part<fold>.pkl,
              model-ds-shap-value-5cv-part<fold>.pkl, fold 0-4.
              data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl: feature columns, annotation,
              disease_status, redcap_id (individual, used only to group the folds).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt, auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt.

Outputs:      data/classifier/feature_importance/model-tn-shap-feature-importance.csv,
              model-ds-shap-feature-importance.csv: feature, mean_shap, cumulative_importance.
              data/classifier/feature_importance/shap_feature_selection_results.csv: per subset, mean and SD of every metric
              of 10_knn_neighbors_accuracy.py.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn 1.0.2); config.py. Other scikit-learn versions
              grow different trees from the same random state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

import pickle
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

IMPORTANCE = ("classifier", "feature_importance")
TN_PARAMS = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 30, "min_samples_split": 20,
             "min_samples_leaf": 30, "bootstrap": True, "class_weight": "balanced_subsample", "n_jobs": -1}
DS_PARAMS = {"n_estimators": 200, "max_features": "sqrt", "max_depth": 25, "min_samples_split": 20,
             "min_samples_leaf": 10, "bootstrap": True, "class_weight": "balanced", "n_jobs": -1}
SUMMARY = ([("mean_acc", "std_acc", "accuracy"), ("MM_mean_acc", "MM_std_acc", "acc_MM"),
            ("MGUS_mean_acc", "MGUS_std_acc", "acc_MGUS"), ("normal_mean_acc", "normal_std_acc", "acc_normal")] +
           [(m + "_mean", m + "_std", m) for m in ("tn_precision", "tn_recall", "tn_f1", "ds_precision", "ds_recall",
                                                   "ds_f1", "macro_precision", "macro_recall", "macro_f1",
                                                   "micro_precision", "micro_recall", "micro_f1")])


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
    return y_tn, y_ds, g, X_tn, X_ds


def load_pickled_data(path):
    with open(path, "rb") as fo:
        return pickle.load(fo)


def check_performance(y_tn, y_ds, g, X_tn, X_ds, tn_columns, ds_columns, tn_features, ds_features,
                      tn_params, ds_params, seed=SEED):
    """per-fold metrics of the two-tier classifier restricted to the given features of each tier"""
    res = {"fold": [], "accuracy": [], "acc_MM": [], "acc_MGUS": [], "acc_normal": [],
           "tn_precision": [], "tn_recall": [], "tn_f1": [],
           "ds_precision": [], "ds_recall": [], "ds_f1": [],
           "macro_precision": [], "macro_recall": [], "macro_f1": [],
           "micro_precision": [], "micro_recall": [], "micro_f1": []}

    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)

    tn_feat_idx = tn_columns.isin(tn_features)
    ds_feat_idx = ds_columns.isin(ds_features)

    for i, (train_idx, test_idx) in enumerate(cv.split(X_tn, y_tn, groups=g)):
        # malignancy tier
        model_tn = RandomForestClassifier(**tn_params)
        Xt_tn = X_tn[train_idx].copy()
        yt_tn = y_tn[train_idx].copy()
        model_tn.fit(Xt_tn[:, tn_feat_idx], yt_tn)

        # MGUS-MM tier on the malignant training cells
        model_ds = RandomForestClassifier(**ds_params)
        Xt_ds = X_ds[train_idx].copy()
        yt_ds = y_ds[train_idx].copy()
        Xt_ds = Xt_ds[yt_tn == "tumor"]
        yt_ds = pd.Categorical(list(yt_ds[yt_tn == "tumor"]), categories=["MGUS", "MM"])
        model_ds.fit(Xt_ds[:, ds_feat_idx], yt_ds)

        # cells called malignant get the MGUS-MM call
        X_test_tn = X_tn[test_idx].copy()
        X_test_ds = X_ds[test_idx].copy()
        y_pred = model_tn.predict(X_test_tn[:, tn_feat_idx])
        y_pred_ds = model_ds.predict(X_test_ds[:, ds_feat_idx])
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


def get_cumsum_features(shap_df, cutoff=0.95, offset=1):
    """top features whose cumulative share stays under the cut-off, plus `offset` more"""
    mask = shap_df["cumulative_importance"] < cutoff
    return shap_df.head(n=mask.sum() + offset).index


# ---- training data and classifier columns ---------------------------------------------------------------------------------
train_expr = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl"))
y_tn, y_ds, g, X_tn, X_ds = format_data(train_expr)

# ---- mean absolute SHAP value per feature over the held-out cells of the five folds ---------------------------------------
shap_dir = data(*IMPORTANCE, "shap")
tn_shap_outs = sorted(str(p) for p in shap_dir.glob("model-tn-*.pkl"))
ds_shap_outs = sorted(str(p) for p in shap_dir.glob("model-ds-*.pkl"))

tn_shap = []
ds_shap = []
for i, _ in enumerate(tn_shap_outs):
    tn = load_pickled_data(tn_shap_outs[i])
    ds = load_pickled_data(ds_shap_outs[i])
    tn_shap.append(tn[1])                                       # class tumor
    ds_shap.append(ds[1])                                       # class MM
tn_shap = np.concatenate(tn_shap)
ds_shap = np.concatenate(ds_shap)

tn_shap = pd.DataFrame({"feature": X_tn.columns, "mean_shap": np.abs(tn_shap).mean(axis=0)})
ds_shap = pd.DataFrame({"feature": X_ds.columns, "mean_shap": np.abs(ds_shap).mean(axis=0)})
tn_shap = tn_shap.set_index("feature").sort_values("mean_shap", ascending=False)
ds_shap = ds_shap.set_index("feature").sort_values("mean_shap", ascending=False)
tn_shap["cumulative_importance"] = tn_shap["mean_shap"].cumsum() / tn_shap["mean_shap"].sum()
ds_shap["cumulative_importance"] = ds_shap["mean_shap"].cumsum() / ds_shap["mean_shap"].sum()

tn_shap.to_csv(data(*IMPORTANCE, "model-tn-shap-feature-importance.csv"))
ds_shap.to_csv(data(*IMPORTANCE, "model-ds-shap-feature-importance.csv"))

# ---- group entries: the nine partner genes (malignancy tier) and the five CNV-arm scores (MGUS-MM tier) ------------------
tn_shap = pd.read_csv(data(*IMPORTANCE, "model-tn-shap-feature-importance.csv"), index_col=0)
cyto_features = ["CCND1", "CCND2", "MAF", "MYC", "FGFR3", "NSD2", "MAFB", "CCND3", "MAFA"]
tn_shap.loc[tn_shap.index.isin(cyto_features), "mean_shap"] = tn_shap.loc[tn_shap.index.isin(cyto_features), "mean_shap"].sum()
tn_shap.index = [i if i not in cyto_features else "cytogenetics" for i in tn_shap.index]
tn_shap = tn_shap[["mean_shap"]].drop_duplicates().sort_values("mean_shap", ascending=False)
tn_shap["cumulative_importance"] = tn_shap["mean_shap"].cumsum() / tn_shap["mean_shap"].sum()

ds_shap = pd.read_csv(data(*IMPORTANCE, "model-ds-shap-feature-importance.csv"), index_col=0)
cnv_features = list(ds_shap.index[ds_shap.index.str.endswith("full_log_fc")])
ds_shap.loc[ds_shap.index.isin(cnv_features), "mean_shap"] = ds_shap.loc[ds_shap.index.isin(cnv_features), "mean_shap"].sum()
ds_shap.index = [i if i not in cnv_features else "cnv_arm_feature" for i in ds_shap.index]
ds_shap = ds_shap[["mean_shap"]].drop_duplicates().sort_values("mean_shap", ascending=False)
ds_shap["cumulative_importance"] = ds_shap["mean_shap"].cumsum() / ds_shap["mean_shap"].sum()

# ---- two-tier classifier on SHAP-based feature subsets -----------------------------------------------------------------
# (subset name, features of the malignancy tier, features of the MGUS-MM tier)
SUBSETS = [("mean_001", tn_shap.loc[tn_shap["mean_shap"] > 0.01].index, ds_shap.loc[ds_shap["mean_shap"] > 0.01].index)]
for name, (tn_cut, tn_off), (ds_cut, ds_off) in [("cumsum_90+1", (0.90, 1), (0.90, 1)),
                                                 ("cumsum_90+0", (0.90, 0), (0.90, 0)),
                                                 ("cumsum_95+1", (0.95, 1), (0.95, 1)),
                                                 ("cumsum_95+0", (0.95, 0), (0.95, 0)),
                                                 ("tn90+1_ds95+1", (0.90, 1), (0.95, 1)),
                                                 ("tn90+1_ds95+0", (0.90, 1), (0.95, 0)),
                                                 ("tn90+0_ds95+1", (0.90, 0), (0.95, 1)),
                                                 ("tn90+0_ds95+0", (0.90, 0), (0.95, 0)),
                                                 ("tn95+1_ds90+1", (0.95, 1), (0.90, 1)),
                                                 ("tn95+0_ds90+1", (0.95, 0), (0.90, 1)),
                                                 ("tn95+1_ds90+0", (0.95, 1), (0.90, 0)),
                                                 ("tn95+0_ds90+0", (0.95, 0), (0.90, 0))]:
    SUBSETS.append((name, get_cumsum_features(tn_shap, cutoff=tn_cut, offset=tn_off),
                    get_cumsum_features(ds_shap, cutoff=ds_cut, offset=ds_off)))

res_summary = {"shap": []}
res_summary.update({col: [] for mean_col, sd_col, _ in SUMMARY for col in (mean_col, sd_col)})
for name, tn_features, ds_features in SUBSETS:
    res = check_performance(y_tn, y_ds, g, X_tn.values, X_ds.values, X_tn.columns, X_ds.columns,
                            tn_features, ds_features, TN_PARAMS, DS_PARAMS, seed=SEED)
    res_summary["shap"].append(name)
    for mean_col, sd_col, metric in SUMMARY:
        res_summary[mean_col].append(res[metric].mean())
        res_summary[sd_col].append(res[metric].std())
    print("%s: accuracy %.4f (%.4f)" % (name, res["accuracy"].mean(), res["accuracy"].std()), flush=True)

pd.DataFrame(res_summary).to_csv(data(*IMPORTANCE, "shap_feature_selection_results.csv"), index=False)
print("done")
