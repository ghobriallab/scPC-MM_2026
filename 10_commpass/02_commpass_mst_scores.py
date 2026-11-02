#!/usr/bin/env python3
"""Malignancy and MST scores of the CoMMpass baseline BM samples from the final random forest classifiers (Fig. 5D;
Extended Data Fig. 9A,B).

Purpose:      The final malignancy and MGUS-MM classifiers (kNN-smoothed features, k = 30) score each sample from its
              signature, CNV-arm and partner-gene scores, in log1p TPM (exr) and DESeq2 VST (vst) units, with the
              features scaled over the CoMMpass samples alone (raw) or together with the cells of donors and of patients
              with MGUS or MM of the classifier training and test sets (norm). Malignancy features (mean expression of the
              16 signatures and SenMayo, partner genes) are z-scored; MGUS-MM features are the AUCell scores of the 16
              signatures as they are and the CNV-arm scores quantile-transformed to a normal distribution (random_state
              42). Malignant score = probability of the tumor class; MM score (MST score) = probability of MM; pred = the
              malignancy call, replaced by the MGUS-MM call for samples called malignant. Seeds 42 (Python random,
              numpy). Run with PYTHONHASHSEED=0.

Inputs:       data/classifier/sc-tumor-model-knn30-RF.pkl and sc-disease-model-knn30-RF.pkl: final random forests
              (classes normal, tumor and MGUS, MM).
              data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl and
              sc-test-cohort-smoothed-expr_47samples.pkl: cohort cells (disease_status, annotation, features).
              data/signature/external/ledergor_natmedicine2018_features-knn30.pkl: names of the feature columns.
              data/classifier/datasets/feature_columns/: mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt, auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt.
              data/commpass/MMRF_CoMMpass_IA22_BM_1_VST_sig_score.tsv and MMRF_CoMMpass_IA22_BM_1_LogExpr_sig_score.tsv
              (01_commpass_expression_and_scores.py; rows = PUBLIC_ID).

Outputs:      data/commpass/mm_classifier_results_250313.csv: per patient (PUBLIC_ID) and version (exr_raw, exr_norm,
              vst_raw, vst_norm): Malignant_score, MM_score, tn_pred, ds_pred, pred.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn); config.py.
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
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(SEED)
np.random.seed(SEED)
FEATURE_COLUMNS = data("classifier", "datasets", "feature_columns")


def load_external_list(valid_columns, input_file):
    """names of one feature list that are columns of the data, in the column order of the data"""
    columns = [x.rstrip() for x in open(input_file)]
    return list(valid_columns.intersection(columns))


def compute_zscore(df):
    ss = StandardScaler()
    ss.fit(df)
    return pd.DataFrame(ss.transform(df), index=df.index, columns=df.columns)


def quantile_transform(df, seed=SEED):
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(df)
    return pd.DataFrame(qt.transform(df), index=df.index, columns=df.columns)


def format_external_data(expr, ref_expr=pd.DataFrame(), seed=SEED):
    """classifier inputs of the samples, scaled alone or together with the reference cells"""
    df = expr.copy()
    df["class"] = list(df["disease_status"])
    df.loc[df["annotation"] == "normal", "class"] = "normal"

    if not ref_expr.empty:
        df = pd.concat([expr, ref_expr])

    # feature columns of each group
    mean_gene_modules = load_external_list(df.columns, FEATURE_COLUMNS / "mean_expr_16_gene_modules.txt")
    mean_aging_sigs = load_external_list(df.columns, FEATURE_COLUMNS / "mean_expr_1_aging_signature.txt")
    cyto_genes = load_external_list(df.columns, FEATURE_COLUMNS / "cytogenetics_genes.txt")
    auc_gene_modules = load_external_list(df.columns, FEATURE_COLUMNS / "auc_16_gene_modules.txt")
    cnv_arms = load_external_list(df.columns, FEATURE_COLUMNS / "mean_expr_fc_driver_cnv_arms.txt")

    X_tn = compute_zscore(df[mean_gene_modules + mean_aging_sigs + cyto_genes])
    X_ds = df[auc_gene_modules].join(quantile_transform(df[cnv_arms], seed))

    if not ref_expr.empty:
        X_tn = X_tn.loc[expr.index]
        X_ds = X_ds.loc[expr.index]

    y_tn = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
    y_ds = np.array(df["class"])

    tn_feats = mean_gene_modules + mean_aging_sigs + cyto_genes
    ds_feats = auc_gene_modules + cnv_arms

    return y_tn, y_ds, X_tn.values, X_ds.values, tn_feats, ds_feats


def predict_commpass(model_tn, model_ds, table, ref_expr=pd.DataFrame()):
    """malignancy and MGUS-MM probabilities and calls of each sample"""
    res = table.copy()

    y_tn, y_ds, X_tn, X_ds, _, _ = format_external_data(res, ref_expr)

    tn_probs = pd.DataFrame(model_tn.predict_proba(X_tn), columns=model_tn.classes_, index=res.index)
    tn_probs["tn_pred"] = model_tn.classes_[np.argmax(tn_probs.values, axis=1)]

    ds_probs = pd.DataFrame(model_ds.predict_proba(X_ds), columns=model_ds.classes_, index=res.index)
    ds_probs["ds_pred"] = model_ds.classes_[np.argmax(ds_probs.values, axis=1)]

    probs = tn_probs.join(ds_probs)
    probs["pred"] = list(tn_probs["tn_pred"])
    probs.loc[probs["pred"] == "tumor", "pred"] = ds_probs.loc[probs["pred"] == "tumor", "ds_pred"]

    res = res.join(probs)
    res.rename(columns={"MM": "MM score", "tumor": "Malignant score"}, inplace=True)
    return res


# ---- 1. final classifiers and the reference cells -------------------------------------------------------------------------
with open(data("classifier", "sc-tumor-model-knn30-RF.pkl"), "rb") as fi:
    model_tn = pickle.load(fi)
with open(data("classifier", "sc-disease-model-knn30-RF.pkl"), "rb") as fi:
    model_ds = pickle.load(fi)

# training and test cells of donors and of patients with MGUS or MM
expr = pd.concat([pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl")),
                  pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-expr_47samples.pkl"))])
expr = expr[~expr["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
expr["class"] = expr["disease_status"]
expr.loc[expr["annotation"] == "normal", "class"] = "normal"

# feature columns as in the tables of the public datasets
columns = pd.read_pickle(data("signature", "external", "ledergor_natmedicine2018_features-knn30.pkl")).columns
expr = expr[["class", "annotation", "disease_status"] + list(columns)]

# ---- 2. scores of the CoMMpass samples ------------------------------------------------------------------------------------
commpass_vst = pd.read_csv(data("commpass", "MMRF_CoMMpass_IA22_BM_1_VST_sig_score.tsv"), sep="\t", index_col=0)
commpass_exr = pd.read_csv(data("commpass", "MMRF_CoMMpass_IA22_BM_1_LogExpr_sig_score.tsv"), sep="\t", index_col=0)
commpass_vst["annotation"] = "tumor"
commpass_exr["annotation"] = "tumor"
commpass_vst["disease_status"] = "MM"
commpass_exr["disease_status"] = "MM"

commpass_vst_raw = predict_commpass(model_tn, model_ds, commpass_vst)
commpass_vst_norm = predict_commpass(model_tn, model_ds, commpass_vst, ref_expr=expr)
commpass_exr_raw = predict_commpass(model_tn, model_ds, commpass_exr)
commpass_exr_norm = predict_commpass(model_tn, model_ds, commpass_exr, ref_expr=expr)

# ---- 3. one table with the four versions -----------------------------------------------------------------------------------
base_cols = ["Malignant score", "MM score", "tn_pred", "ds_pred", "pred"]
exr_raw = commpass_exr_raw[base_cols].rename(columns={c: "exr_raw:" + c.replace(" ", "_") for c in base_cols})
exr_norm = commpass_exr_norm[base_cols].rename(columns={c: "exr_norm:" + c.replace(" ", "_") for c in base_cols})
vst_raw = commpass_vst_raw[base_cols].rename(columns={c: "vst_raw:" + c.replace(" ", "_") for c in base_cols})
vst_norm = commpass_vst_norm[base_cols].rename(columns={c: "vst_norm:" + c.replace(" ", "_") for c in base_cols})

results = exr_raw.join(exr_norm).join(vst_raw).join(vst_norm)
results.to_csv(data("commpass", "mm_classifier_results_250313.csv"))
