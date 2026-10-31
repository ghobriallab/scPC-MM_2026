#!/usr/bin/env python3
"""Cross-validated accuracy of the two-tier classifier with the tuned parameters of each tier, for the three candidate
malignancy tiers (Methods, classifier development).

Purpose:      Training cells of donors, MGUS and MM patients (SMM removed), cohort-smoothed features; malignancy tier
              (malignant vs normal), then MGUS-MM tier (AUCell scores of the 16 signatures + quantile-normalized CNV-arm
              scores, seed 42) trained on the malignant training cells and applied to the cells called malignant. Four
              combinations, each with the parameters selected by 08_grid_search_tiers.py: malignancy tier
              auc:gm16+age+arm+cyto, mex:gm16+age+arm+cyto or mex:gm16+age+cyto, and mex:gm16+age+cyto with the same
              parameters for both tiers (100 trees, max_depth 20, min_samples_split 20, min_samples_leaf 10, balanced).
              5-fold cross-validation grouped by individual (StratifiedGroupKFold, shuffled, seed 42): accuracy overall and
              per class (normal, MGUS, MM). The forests get no random_state: they draw from numpy's global random state,
              seeded with 42 once, and the combinations run in the order above. The selected combination is
              mex:gm16+age+cyto (malignancy) with auc:gm16+arm (MGUS-MM). Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl: feature columns, annotation,
              disease_status, redcap_id (individual, used only to group the folds).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              auc_16_gene_modules.txt, auc_1_aging_signature.txt, cytogenetics_genes.txt, mean_expr_fc_driver_cnv_arms.txt.

Outputs:      results/classifiers/tuned_two_tier_cv_folds.tsv: combination, fold, accuracy, acc_MM, acc_MGUS, acc_normal.
              results/classifiers/tuned_two_tier_cv_summary.tsv: mean and SD (pandas, n - 1) over the folds.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn 1.0.2); config.py. Other scikit-learn versions
              grow different trees from the same random state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data, result_dir

import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)
OUT = result_dir("classifiers")


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


def check_performance(y_tn, y_ds, g, X_tn, X_ds, tn_params, ds_params, seed=SEED):
    """per-fold accuracy (overall, MM, MGUS, normal) of the two-tier classifier"""
    res = {"fold": [], "accuracy": [], "acc_MM": [], "acc_MGUS": [], "acc_normal": []}
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

        # truth: normal, or the disease stage of malignant cells
        y_truth = np.array(y_tn[test_idx])
        y_truth_ds = np.array(y_ds[test_idx])
        y_truth[y_truth == "tumor"] = y_truth_ds[y_truth == "tumor"]

        pi = pd.DataFrame({"label": y_truth, "pred": y_pred})
        res["fold"].append(i)
        res["accuracy"].append(sum(pi["label"] == pi["pred"]) / pi.shape[0])
        n_normal = sum(pi["label"] == "normal")
        res["acc_normal"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "normal")) / max(n_normal, 1))
        n_MGUS = sum(pi["label"] == "MGUS")
        res["acc_MGUS"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "MGUS")) / max(n_MGUS, 1))
        n_MM = sum(pi["label"] == "MM")
        res["acc_MM"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "MM")) / max(n_MM, 1))
    return pd.DataFrame(res)


# ---- training cells and labels -------------------------------------------------------------------------------------------
train_expr = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl"))
df = train_expr.copy()
df = df[~df["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
df["class"] = df["disease_status"]
df.loc[df["annotation"] == "normal", "class"] = "normal"

y_tn = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
y_ds = np.array(df["class"])
g = pd.Categorical(list(df["redcap_id"]), categories=sorted(set(df["redcap_id"])))

# ---- feature matrices --------------------------------------------------------------------------------------------------
mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
auc_gene_modules = load_list("auc_16_gene_modules.txt")
auc_aging_sigs = load_list("auc_1_aging_signature.txt")
cyto_genes = load_list("cytogenetics_genes.txt")
cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

X_cnv_arms = quantile_transform(df[cnv_arms])

X_tn_auc_gm16_age_arm_cyto = df[auc_gene_modules + auc_aging_sigs].join(X_cnv_arms).join(compute_zscore(df[cyto_genes])).values
X_tn_mex_gm16_age_arm_cyto = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes]).join(X_cnv_arms).values
X_tn_mex_gm16_age_cyto = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes]).values
X_ds_auc_gm16_arm = df[auc_gene_modules].join(X_cnv_arms).values

# ---- tuned parameters (08_grid_search_tiers.py) ----------------------------------------------------------------------
tn_auc_gm16_age_arm_cyto_params = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 20, "min_samples_split": 50,
                                   "min_samples_leaf": 20, "bootstrap": True, "class_weight": "balanced_subsample",
                                   "n_jobs": -1}
tn_mex_gm16_age_arm_cyto_params = {"n_estimators": 200, "max_features": "sqrt", "max_depth": 20, "min_samples_split": 50,
                                   "min_samples_leaf": 20, "bootstrap": True, "class_weight": "balanced", "n_jobs": -1}
tn_mex_gm16_age_cyto_params = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 30, "min_samples_split": 20,
                               "min_samples_leaf": 30, "bootstrap": True, "class_weight": "balanced_subsample",
                               "n_jobs": -1}
ds_auc_gm16_arm_params = {"n_estimators": 200, "max_features": "sqrt", "max_depth": 25, "min_samples_split": 20,
                          "min_samples_leaf": 10, "bootstrap": True, "class_weight": "balanced", "n_jobs": -1}
uniform_params = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 20, "min_samples_split": 20,
                  "min_samples_leaf": 10, "bootstrap": True, "class_weight": "balanced", "n_jobs": -1}

COMBINATIONS = [
    ("TN auc:gm16+age+arm+cyto | DS auc:gm16+arm", X_tn_auc_gm16_age_arm_cyto, tn_auc_gm16_age_arm_cyto_params,
     ds_auc_gm16_arm_params),
    ("TN mex:gm16+age+arm+cyto | DS auc:gm16+arm", X_tn_mex_gm16_age_arm_cyto, tn_mex_gm16_age_arm_cyto_params,
     ds_auc_gm16_arm_params),
    ("TN mex:gm16+age+cyto | DS auc:gm16+arm", X_tn_mex_gm16_age_cyto, tn_mex_gm16_age_cyto_params, ds_auc_gm16_arm_params),
    ("TN mex:gm16+age+cyto | DS auc:gm16+arm, same parameters", X_tn_mex_gm16_age_cyto, uniform_params, uniform_params),
]

folds = []
for name, X_tn, tn_params, ds_params in COMBINATIONS:
    res = check_performance(y_tn, y_ds, g, X_tn, X_ds_auc_gm16_arm, tn_params, ds_params, seed=SEED)
    res.insert(0, "combination", name)
    folds.append(res)
folds = pd.concat(folds)
folds.to_csv(os.path.join(OUT, "tuned_two_tier_cv_folds.tsv"), sep="\t", index=False)

metrics = ["accuracy", "acc_MM", "acc_MGUS", "acc_normal"]
summary = folds.groupby("combination", sort=False)[metrics].agg(["mean", "std"])
summary.columns = ["%s_%s" % (m, s) for m, s in summary.columns]
summary.reset_index().to_csv(os.path.join(OUT, "tuned_two_tier_cv_summary.tsv"), sep="\t", index=False)
print(summary.round(6).to_string())
