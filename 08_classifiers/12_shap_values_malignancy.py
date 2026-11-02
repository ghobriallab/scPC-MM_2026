#!/usr/bin/env python3
"""SHAP values of the malignancy random forest for one cross-validation fold (feature ranking used by the feature
ablation, Extended Data Fig. 6E).

Purpose:      Fits the malignant vs normal forest with the tuned parameters (100 trees, max_features sqrt, max_depth 30,
              min_samples_split 20, min_samples_leaf 30, bootstrap, balanced_subsample class weights) on the training cells
              of the fold and computes the SHAP values of its held-out cells with shap.TreeExplainer. The forest gets no
              random_state: it draws from numpy's global random state, seeded with 42 at the start. The script restarts
              itself with PYTHONHASHSEED=0 when that variable is not set.
              Usage: python3 12_shap_values_malignancy.py <fold>, once for each fold 0-4 written by 11_shap_training_folds.py
              (the runs are independent and can run in parallel).

Inputs:       data/classifier/feature_importance/shap/training-package-5cv-part<fold>.pkl: X_train_tn, y_train_tn, X_test_tn.

Outputs:      data/classifier/feature_importance/shap/model-tn-shap-value-5cv-part<fold>.pkl: SHAP values of the held-out
              cells, one array per class (normal, tumor).

Dependencies: Python 3.8 with requirements.txt (numpy, scikit-learn 1.0.2, shap); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

if not os.getenv("PYTHONHASHSEED"):                                     # restart with PYTHONHASHSEED=0
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import pickle
import random

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

random.seed(SEED)
np.random.seed(SEED)

# malignancy tier: cohort-smoothed mex:gm16+age+cyto, tuned parameters
TN_PARAMS = {"n_estimators": 100,
             "max_features": "sqrt",
             "max_depth": 30,
             "min_samples_split": 20,
             "min_samples_leaf": 30,
             "bootstrap": True,
             "class_weight": "balanced_subsample",
             "n_jobs": -1}


def compute_shap(fold, tn_params=TN_PARAMS):
    shap_dir = data("classifier", "feature_importance", "shap")
    with open(shap_dir / "training-package-5cv-part{}.pkl".format(fold), "rb") as fi:
        fold_data = pickle.load(fi)

    model_tn = RandomForestClassifier(**tn_params)
    model_tn.fit(fold_data["X_train_tn"], fold_data["y_train_tn"])
    explainer = shap.TreeExplainer(model_tn)
    shap_values = explainer.shap_values(fold_data["X_test_tn"])

    with open(shap_dir / "model-tn-shap-value-5cv-part{}.pkl".format(fold), "wb") as fo:
        pickle.dump(shap_values, fo)


parser = argparse.ArgumentParser(description="SHAP values of the malignancy forest on one cross-validation fold")
parser.add_argument("fold", type=int, choices=range(5), help="fold number, 0-4")
args = parser.parse_args()
compute_shap(args.fold)
