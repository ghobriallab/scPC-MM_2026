#!/usr/bin/env python3
"""Training and held-out cells of the five cross-validation folds of the two-tier classifier, for the SHAP feature ranking
(the ranking used by the feature ablation, Supplementary Fig. 12E).

Purpose:      Training cells of donors, MGUS and MM patients (SMM removed), cohort-smoothed features as in the final
              classifier: malignancy tier = z-scored mean expression of the 16 signatures and senescence score + partner
              genes; MGUS-MM tier = AUCell scores of the 16 signatures + CNV-arm scores quantile-normalized (seed 42).
              Folds: 5-fold cross-validation grouped by individual (StratifiedGroupKFold on the malignancy label, shuffled,
              seed 42). Each fold package holds the training cells of both tiers (the MGUS-MM tier keeps only the
              malignant training cells, labels MGUS/MM) and the held-out cells (all held-out cells for the malignancy tier,
              the malignant ones for the MGUS-MM tier). The SHAP values are computed per fold by
              12_shap_values_malignancy.py and 13_shap_values_mst.py. Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl: feature columns, annotation,
              disease_status, redcap_id (individual, used only to group the folds).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt, auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt.

Outputs:      data/classifier/feature_importance/shap/training-package-5cv-part<fold>.pkl, fold 0-4: dictionary with
              train_idx, test_idx, tn_feature, ds_feature, X_train_tn, y_train_tn, X_train_ds, y_train_ds, X_test_tn,
              X_test_ds.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn 1.0.2); config.py.
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)

SHAP_DIR = data("classifier", "feature_importance", "shap")


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


def create_training_bundle(y_tn, y_ds, g, X_tn, X_ds, tn_features, ds_features, seed=SEED):
    """one pickled package per cross-validation fold"""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(cv.split(X_tn, y_tn, groups=g)):
        # training cells: all for the malignancy tier, malignant ones for the MGUS-MM tier
        Xt_tn = X_tn[train_idx].copy()
        yt_tn = y_tn[train_idx].copy()
        Xt_ds = X_ds[train_idx].copy()
        yt_ds = y_ds[train_idx].copy()
        Xt_ds = Xt_ds[yt_tn == "tumor"]
        yt_ds = pd.Categorical(list(yt_ds[yt_tn == "tumor"]), categories=["MGUS", "MM"])

        # held-out cells: all for the malignancy tier, malignant ones for the MGUS-MM tier
        Xs_tn = X_tn[test_idx].copy()
        Xs_ds = X_ds[test_idx].copy()
        Xs_ds = Xs_ds[y_tn[test_idx] == "tumor"]

        fold_data = {"train_idx": train_idx,
                     "test_idx": test_idx,
                     "tn_feature": tn_features,
                     "ds_feature": ds_features,
                     "X_train_tn": Xt_tn,
                     "y_train_tn": yt_tn,
                     "X_train_ds": Xt_ds,
                     "y_train_ds": yt_ds,
                     "X_test_tn": Xs_tn,
                     "X_test_ds": Xs_ds}
        with open(SHAP_DIR / "training-package-5cv-part{}.pkl".format(i), "wb") as fo:
            pickle.dump(fold_data, fo)


train_expr = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl"))
y_tn, y_ds, g, X_tn, X_ds = format_data(train_expr)
create_training_bundle(y_tn, y_ds, g, X_tn.values, X_ds.values, X_tn.columns, X_ds.columns, seed=SEED)
print("done")
