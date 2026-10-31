#!/usr/bin/env python3
"""Cross-validated accuracy of the two-tier classifier (malignant vs normal, then MGUS vs MM) for every combination of
the feature sets of the two tiers (Supplementary Fig. 12B; Methods, classifier development).

Purpose:      Training cells of donors, MGUS and MM patients (SMM removed), cohort-smoothed and patient-smoothed versions.
              Tier 1: random forest for malignant vs normal PCs; tier 2: random forest for MGUS vs MM trained on the
              malignant training cells; cells called malignant by tier 1 get the tier-2 call, giving three classes (normal,
              MGUS, MM). Feature sets as in 04_tumor_normal_feature_sets.py (z-scored mean-expression scores and partner
              genes, quantile-normalized CNV-arm scores, AUCell scores as they are), named <smoothing>|<feature set>. Every
              pair of gm16 feature sets (32 x 32, both smoothings) and every pair of gm22 feature sets (32 x 32). Forests:
              100 trees, min_samples_leaf 20, balanced_subsample class weights, seed 42. 5-fold cross-validation grouped by
              individual (StratifiedGroupKFold on the tier-1 label, shuffled, seed 42): accuracy overall and per class
              (normal, MGUS, MM), mean and SD over the folds. Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl,
              sc-train-patient-smoothed-expr_134samples.pkl: feature columns, annotation, disease_status, redcap_id
              (individual, used only to group the folds).
              data/classifier/datasets/feature_columns/: the eight feature-column lists of 04_tumor_normal_feature_sets.py.

Outputs:      data/classifier/performance/init_model/full/cross-combined-mgus-mm-classifier.csv: classifier, tn_feature,
              ds_feature, mean_acc, std_acc, MM_mean_acc, MM_std_acc, MGUS_mean_acc, MGUS_std_acc, normal_mean_acc,
              normal_std_acc.

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


def format_data(expr, remove_smm=False, prefix=""):
    """labels, individual, feature table (columns named <prefix><feature>) and the 32 feature sets"""
    df = expr.copy()

    if remove_smm:
        df = df[~df["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]

    df["class"] = df["disease_status"]
    df.loc[df["annotation"] == "normal", "class"] = "normal"

    mean_gene_modules = load_list("mean_expr_16_gene_modules.txt")
    mean_de_genes = load_list("mean_expr_6_de_gene_sets.txt")
    mean_aging_sigs = load_list("mean_expr_1_aging_signature.txt")
    auc_gene_modules = load_list("auc_16_gene_modules.txt")
    auc_de_genes = load_list("auc_6_de_gene_sets.txt")
    auc_aging_sigs = load_list("auc_1_aging_signature.txt")
    cyto_genes = load_list("cytogenetics_genes.txt")
    cnv_arms = load_list("mean_expr_fc_driver_cnv_arms.txt")

    # z-scored expression-based features, normal-mapped CNV-arm scores, AUCell scores as they are
    mat = compute_zscore(df[mean_gene_modules + mean_de_genes + mean_aging_sigs + cyto_genes])
    mat = mat.join(quantile_transform(df[cnv_arms]))
    mat = mat.join(df[auc_gene_modules + auc_de_genes + auc_aging_sigs])

    if prefix != "":
        scope = (auc_gene_modules + auc_de_genes + auc_aging_sigs + mean_gene_modules + mean_de_genes + mean_aging_sigs +
                 cnv_arms + cyto_genes)
        mat.rename(columns={c: prefix + c for c in scope}, inplace=True)
        mean_gene_modules = [prefix + x for x in mean_gene_modules]
        mean_de_genes = [prefix + x for x in mean_de_genes]
        mean_aging_sigs = [prefix + x for x in mean_aging_sigs]
        auc_gene_modules = [prefix + x for x in auc_gene_modules]
        auc_de_genes = [prefix + x for x in auc_de_genes]
        auc_aging_sigs = [prefix + x for x in auc_aging_sigs]
        cyto_genes = [prefix + x for x in cyto_genes]
        cnv_arms = [prefix + x for x in cnv_arms]

    y_tn = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
    y_ds = np.array(df["class"])
    g = pd.Categorical(list(df["redcap_id"]), categories=sorted(set(df["redcap_id"])))

    feats = {
        # 16 signatures, AUCell
        prefix + "auc:gm16":              auc_gene_modules,
        prefix + "auc:gm16+age":          auc_gene_modules + auc_aging_sigs,
        prefix + "auc:gm16+arm":          auc_gene_modules + cnv_arms,
        prefix + "auc:gm16+cyto":         auc_gene_modules + cyto_genes,
        prefix + "auc:gm16+age+cyto":     auc_gene_modules + auc_aging_sigs + cyto_genes,
        prefix + "auc:gm16+arm+cyto":     auc_gene_modules + cnv_arms + cyto_genes,
        prefix + "auc:gm16+age+arm":      auc_gene_modules + auc_aging_sigs + cnv_arms,
        prefix + "auc:gm16+age+arm+cyto": auc_gene_modules + auc_aging_sigs + cnv_arms + cyto_genes,
        # 16 signatures, mean expression
        prefix + "mex:gm16":              mean_gene_modules,
        prefix + "mex:gm16+age":          mean_gene_modules + mean_aging_sigs,
        prefix + "mex:gm16+arm":          mean_gene_modules + cnv_arms,
        prefix + "mex:gm16+cyto":         mean_gene_modules + cyto_genes,
        prefix + "mex:gm16+age+cyto":     mean_gene_modules + mean_aging_sigs + cyto_genes,
        prefix + "mex:gm16+arm+cyto":     mean_gene_modules + cnv_arms + cyto_genes,
        prefix + "mex:gm16+age+arm":      mean_gene_modules + mean_aging_sigs + cnv_arms,
        prefix + "mex:gm16+age+arm+cyto": mean_gene_modules + mean_aging_sigs + cnv_arms + cyto_genes,
        # 16 signatures + 6 DE gene sets, AUCell
        prefix + "auc:gm22":              auc_gene_modules + auc_de_genes,
        prefix + "auc:gm22+age":          auc_gene_modules + auc_de_genes + auc_aging_sigs,
        prefix + "auc:gm22+arm":          auc_gene_modules + auc_de_genes + cnv_arms,
        prefix + "auc:gm22+cyto":         auc_gene_modules + auc_de_genes + cyto_genes,
        prefix + "auc:gm22+age+cyto":     auc_gene_modules + auc_de_genes + auc_aging_sigs + cyto_genes,
        prefix + "auc:gm22+arm+cyto":     auc_gene_modules + auc_de_genes + cnv_arms + cyto_genes,
        prefix + "auc:gm22+age+arm":      auc_gene_modules + auc_de_genes + auc_aging_sigs + cnv_arms,
        prefix + "auc:gm22+age+arm+cyto": auc_gene_modules + auc_de_genes + auc_aging_sigs + cnv_arms + cyto_genes,
        # 16 signatures + 6 DE gene sets, mean expression
        prefix + "mex:gm22":              mean_gene_modules + mean_de_genes,
        prefix + "mex:gm22+age":          mean_gene_modules + mean_de_genes + mean_aging_sigs,
        prefix + "mex:gm22+arm":          mean_gene_modules + mean_de_genes + cnv_arms,
        prefix + "mex:gm22+cyto":         mean_gene_modules + mean_de_genes + cyto_genes,
        prefix + "mex:gm22+age+cyto":     mean_gene_modules + mean_de_genes + mean_aging_sigs + cyto_genes,
        prefix + "mex:gm22+arm+cyto":     mean_gene_modules + mean_de_genes + cnv_arms + cyto_genes,
        prefix + "mex:gm22+age+arm":      mean_gene_modules + mean_de_genes + mean_aging_sigs + cnv_arms,
        prefix + "mex:gm22+age+arm+cyto": mean_gene_modules + mean_de_genes + mean_aging_sigs + cnv_arms + cyto_genes,
    }
    return y_tn, y_ds, g, mat, feats


def train_cross_model(X, tn_feats, ds_feats, y_tn, y_ds, g, classifier_func, params, tumor_categories, seed=SEED):
    """per-fold accuracy (overall, normal, MGUS, MM) of the two-tier classifier"""
    pred_res = {"fold": [], "accuracy": [], "acc_normal": [], "acc_MGUS": [], "acc_MM": []}

    cv = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)

    X_tn = X[tn_feats].values
    X_ds = X[ds_feats].values

    for i, (train_idx, test_idx) in enumerate(cv.split(X_tn, y_tn, groups=g)):
        model_tn = classifier_func(**params)
        model_ds = classifier_func(**params)

        # tier 1: all training cells
        Xt_tn = X_tn[train_idx].copy()
        yt_tn = y_tn[train_idx].copy()

        # tier 2: malignant training cells
        Xt_ds = X_ds[train_idx].copy()
        yt_ds = y_ds[train_idx].copy()
        Xt_ds = Xt_ds[yt_tn == "tumor"]
        yt_ds = pd.Categorical(list(yt_ds[yt_tn == "tumor"]), categories=tumor_categories)

        model_tn.fit(Xt_tn, yt_tn)
        model_ds.fit(Xt_ds, yt_ds)

        # cells called malignant by tier 1 get the tier-2 call
        y_test_pred = model_tn.predict(X_tn[test_idx])
        y_test_pred_ds = model_ds.predict(X_ds[test_idx])
        y_test_pred[y_test_pred == "tumor"] = y_test_pred_ds[y_test_pred == "tumor"]

        # truth: normal, or the disease stage of malignant cells
        y_test = np.array(y_tn[test_idx])
        y_test_ds = np.array(y_ds[test_idx])
        y_test[y_test == "tumor"] = y_test_ds[y_test == "tumor"]

        pi = pd.DataFrame({"label": y_test, "pred": y_test_pred})
        pred_res["fold"].append(i)
        pred_res["accuracy"].append(sum(pi["label"] == pi["pred"]) / pi.shape[0])

        n_normal = sum(pi["label"] == "normal")
        n_MGUS = sum(pi["label"] == "MGUS")
        n_MM = sum(pi["label"] == "MM")
        pred_res["acc_normal"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "normal")) / max(n_normal, 1))
        pred_res["acc_MGUS"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "MGUS")) / max(n_MGUS, 1))
        pred_res["acc_MM"].append(sum((pi["label"] == pi["pred"]) & (pi["label"] == "MM")) / max(n_MM, 1))
    return pd.DataFrame(pred_res).set_index("fold")


# ---- data: cohort-smoothed and patient-smoothed training cells, SMM removed --------------------------------------------
train_expr_grp_knn = pd.read_pickle(data(*DATASETS, "sc-train-cohort-smoothed-expr_134samples.pkl"))
train_expr_pat_knn = pd.read_pickle(data(*DATASETS, "sc-train-patient-smoothed-expr_134samples.pkl"))

y_tn, y_ds, g, mat, feats = format_data(train_expr_grp_knn, remove_smm=True, prefix="cohort-knn|")
_, _, _, mat_pat, feats_pat = format_data(train_expr_pat_knn, remove_smm=True, prefix="sample-knn|")

X = mat.join(mat_pat)
feats.update(feats_pat)
gm16_feats = [x for x in feats.keys() if "gm16" in x]
gm22_feats = [x for x in feats.keys() if "gm22" in x]

random_forest_params = {
    "n_estimators": 100,
    "min_samples_leaf": 20,
    "class_weight": "balanced_subsample",
    "random_state": SEED,
    "n_jobs": -1,
}

# ---- every pair of tier-1 and tier-2 feature sets, gm16 then gm22 -----------------------------------------------------
results = {"classifier": [], "tn_feature": [], "ds_feature": [], "mean_acc": [], "std_acc": [],
           "MM_mean_acc": [], "MM_std_acc": [], "MGUS_mean_acc": [], "MGUS_std_acc": [],
           "normal_mean_acc": [], "normal_std_acc": []}
for feature_sets in (gm16_feats, gm22_feats):
    for feats_tn in feature_sets:
        for feats_ds in feature_sets:
            res = train_cross_model(X, feats[feats_tn], feats[feats_ds], y_tn, y_ds, g,
                                    RandomForestClassifier, random_forest_params, ["MGUS", "MM"], seed=SEED)
            print("tn={} | ds={}: accuracy {:.5} ({:.5})".format(feats_tn, feats_ds, res["accuracy"].mean(),
                                                                 res["accuracy"].std()), flush=True)
            results["classifier"].append("RandomForest")
            results["tn_feature"].append(feats_tn)
            results["ds_feature"].append(feats_ds)
            results["mean_acc"].append(res["accuracy"].mean())
            results["std_acc"].append(res["accuracy"].std())
            results["MM_mean_acc"].append(res["acc_MM"].mean())
            results["MM_std_acc"].append(res["acc_MM"].std())
            results["MGUS_mean_acc"].append(res["acc_MGUS"].mean())
            results["MGUS_std_acc"].append(res["acc_MGUS"].std())
            results["normal_mean_acc"].append(res["acc_normal"].mean())
            results["normal_std_acc"].append(res["acc_normal"].std())

out = data("classifier", "performance", "init_model", "full", "cross-combined-mgus-mm-classifier.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(out, index=False)
print("done")
