#!/usr/bin/env python3
"""Cross-validated accuracy of malignant versus normal PC classifiers for 32 feature sets, three versions of the training
data and two model types (Extended Data Fig. 6A; Methods, classifier development).

Purpose:      Training cells of all disease stages and donors (01_training_test_split.py); label = malignant (tumor) or
              normal PC. Features: mean-expression scores and partner genes z-scored, CNV-arm scores quantile-normalized
              to a normal distribution (seed 42), AUCell scores as they are. The 32 feature sets combine the 16 signatures
              (gm16) or the 16 signatures and the six stage-wise DE gene sets (gm22), scored by AUCell (auc) or mean
              expression (mex), with or without the senescence score (age), the CNV-arm scores (arm) and the partner genes
              (cyto). Training data: unsmoothed (single-cell), smoothed over neighbors in the whole set (cohort-knn) and
              within each individual (sample-knn). Models: random forest (100 trees, min_samples_leaf 20, balanced class
              weights, seed 42) and L2 logistic regression (balanced class weights, max_iter 100). Accuracy over a 5-fold
              cross-validation grouped by individual (StratifiedGroupKFold, shuffled, seed 42): mean and SD over the folds.
              Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/datasets/sc-train-expr_134samples.pkl, sc-train-cohort-smoothed-expr_134samples.pkl,
              sc-train-patient-smoothed-expr_134samples.pkl: feature columns, annotation (label), redcap_id (individual,
              used only to group the folds).
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_6_de_gene_sets.txt,
              mean_expr_1_aging_signature.txt, auc_16_gene_modules.txt, auc_6_de_gene_sets.txt, auc_1_aging_signature.txt,
              cytogenetics_genes.txt, mean_expr_fc_driver_cnv_arms.txt.

Outputs:      data/classifier/performance/init_model/parts/tumor-normal-classifier.csv: classifier, train_set, feature_set,
              mean_acc, std_acc.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn 1.0.2); config.py. Other scikit-learn versions
              grow different trees from the same seed.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)

DATASETS = ("classifier", "datasets")


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
    """label, individual and the 32 feature matrices of one training table"""
    df = expr.copy()

    mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
    mean_de_genes = load_list("mean_expr_6_de_gene_sets.txt")
    mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
    auc_gene_modules = load_list("auc_16_gene_modules.txt")
    auc_de_genes = load_list("auc_6_de_gene_sets.txt")
    auc_aging_sigs = load_list("auc_1_aging_signature.txt")
    cyto_genes = load_list("cytogenetics_genes.txt")
    cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

    # z-scored expression-based features
    mat = compute_zscore(df[mean_gene_modules + mean_de_genes + mean_aging_sigs + cyto_genes])
    # CNV-arm fold changes mapped to a normal distribution
    mat = mat.join(quantile_transform(df[cnv_arms]))
    # AUCell scores as they are
    mat = mat.join(df[auc_gene_modules + auc_de_genes + auc_aging_sigs])

    y = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
    g = pd.Categorical(list(df["redcap_id"]), categories=sorted(set(df["redcap_id"])))
    X = {
        # 16 signatures, AUCell
        "auc:gm16":              mat[auc_gene_modules],
        "auc:gm16+age":          mat[auc_gene_modules + auc_aging_sigs],
        "auc:gm16+arm":          mat[auc_gene_modules + cnv_arms],
        "auc:gm16+cyto":         mat[auc_gene_modules + cyto_genes],
        "auc:gm16+age+cyto":     mat[auc_gene_modules + auc_aging_sigs + cyto_genes],
        "auc:gm16+arm+cyto":     mat[auc_gene_modules + cnv_arms + cyto_genes],
        "auc:gm16+age+arm":      mat[auc_gene_modules + auc_aging_sigs + cnv_arms],
        "auc:gm16+age+arm+cyto": mat[auc_gene_modules + auc_aging_sigs + cnv_arms + cyto_genes],
        # 16 signatures, mean expression
        "mex:gm16":              mat[mean_gene_modules],
        "mex:gm16+age":          mat[mean_gene_modules + mean_aging_sigs],
        "mex:gm16+arm":          mat[mean_gene_modules + cnv_arms],
        "mex:gm16+cyto":         mat[mean_gene_modules + cyto_genes],
        "mex:gm16+age+cyto":     mat[mean_gene_modules + mean_aging_sigs + cyto_genes],
        "mex:gm16+arm+cyto":     mat[mean_gene_modules + cnv_arms + cyto_genes],
        "mex:gm16+age+arm":      mat[mean_gene_modules + mean_aging_sigs + cnv_arms],
        "mex:gm16+age+arm+cyto": mat[mean_gene_modules + mean_aging_sigs + cnv_arms + cyto_genes],
        # 16 signatures + 6 DE gene sets, AUCell
        "auc:gm22":              mat[auc_gene_modules + auc_de_genes],
        "auc:gm22_age":          mat[auc_gene_modules + auc_de_genes + auc_aging_sigs],
        "auc:gm22+arm":          mat[auc_gene_modules + auc_de_genes + cnv_arms],
        "auc:gm22+cyto":         mat[auc_gene_modules + auc_de_genes + cyto_genes],
        "auc:gm22+age+cyto":     mat[auc_gene_modules + auc_de_genes + auc_aging_sigs + cyto_genes],
        "auc:gm22+arm+cyto":     mat[auc_gene_modules + auc_de_genes + cnv_arms + cyto_genes],
        "auc:gm22+age+arm":      mat[auc_gene_modules + auc_de_genes + auc_aging_sigs + cnv_arms],
        "auc:gm22+age+arm+cyto": mat[auc_gene_modules + auc_de_genes + auc_aging_sigs + cnv_arms + cyto_genes],
        # 16 signatures + 6 DE gene sets, mean expression
        "mex:gm22":              mat[mean_gene_modules + mean_de_genes],
        "mex:gm22+age":          mat[mean_gene_modules + mean_de_genes + mean_aging_sigs],
        "mex:gm22+arm":          mat[mean_gene_modules + mean_de_genes + cnv_arms],
        "mex:gm22+cyto":         mat[mean_gene_modules + mean_de_genes + cyto_genes],
        "mex:gm22+age+cyto":     mat[mean_gene_modules + mean_de_genes + mean_aging_sigs + cyto_genes],
        "mex:gm22+arm+cyto":     mat[mean_gene_modules + mean_de_genes + cnv_arms + cyto_genes],
        "mex:gm22+age+arm":      mat[mean_gene_modules + mean_de_genes + mean_aging_sigs + cnv_arms],
        "mex:gm22+age+arm+cyto": mat[mean_gene_modules + mean_de_genes + mean_aging_sigs + cnv_arms + cyto_genes],
    }
    return y, g, X


def train_model(X, y, g, classifier_func, params, seed=SEED):
    """accuracy per fold of a 5-fold cross-validation grouped by individual"""
    pred_res = {"fold": [], "accuracy": []}
    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=g)):
        model = classifier_func(**params)
        model.fit(X[train_idx].copy(), y[train_idx].copy())
        y_test_pred = model.predict(X[test_idx])
        pi = pd.DataFrame({"label": y[test_idx], "pred": y_test_pred})
        pred_res["fold"].append(i)
        pred_res["accuracy"].append(sum(pi["label"] == pi["pred"]) / pi.shape[0])
    return pd.DataFrame(pred_res).set_index("fold")


# ---- data: unsmoothed, cohort-smoothed and patient-smoothed training cells -------------------------------------------------
train_expr = pd.read_pickle(data(*DATASETS, "sc-train-expr_134samples.pkl"))
train_expr_grp_knn = pd.read_pickle(data(*DATASETS, "sc-train-cohort-smoothed-expr_134samples.pkl"))
train_expr_pat_knn = pd.read_pickle(data(*DATASETS, "sc-train-patient-smoothed-expr_134samples.pkl"))

y, g, X = format_data(train_expr)
_, _, X_grp_knn = format_data(train_expr_grp_knn)
_, _, X_pat_knn = format_data(train_expr_pat_knn)

random_forest_params = {
    "n_estimators": 100,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}
logistic_params = {
    "penalty": "l2",
    "class_weight": "balanced",
    "max_iter": 100,
    "n_jobs": -1,
}

# ---- 5-fold cross-validation of every model x training data x feature set ----------------------------------------------
results = {"classifier": [], "train_set": [], "feature_set": [], "mean_acc": [], "std_acc": []}
for classifier, classifier_func, params in (("RandomForest", RandomForestClassifier, random_forest_params),
                                            ("LogisticRegression", LogisticRegression, logistic_params)):
    for train_set, X_set in (("single-cell", X), ("cohort-knn", X_grp_knn), ("sample-knn", X_pat_knn)):
        for key in X_set.keys():
            res = train_model(X_set[key].values, y, g, classifier_func, params, seed=SEED)
            print("{} | {} | {}: accuracy {:.5} ({:.5})".format(classifier, train_set, key, res["accuracy"].mean(),
                                                                 res["accuracy"].std()), flush=True)
            results["classifier"].append(classifier)
            results["train_set"].append(train_set)
            results["feature_set"].append(key)
            results["mean_acc"].append(res["accuracy"].mean())
            results["std_acc"].append(res["accuracy"].std())

out = data("classifier", "performance", "init_model", "parts", "tumor-normal-classifier.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(out, index=False)
print("done")
