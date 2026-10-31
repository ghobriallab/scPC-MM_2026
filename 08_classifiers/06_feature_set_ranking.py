#!/usr/bin/env python3
"""Ranking of the feature sets of the malignancy and MGUS-MM tiers by cross-validated accuracy, the basis of the feature
choice of the final classifiers (Methods, classifier development; Supplementary Fig. 12A,B).

Purpose:      Only feature sets built from the 16 signatures (gm16) are ranked: the six stage-wise DE gene sets (gm22)
              were selected with the MGUS, SMM and MM labels. Malignancy tier: gm16 feature sets of
              04_tumor_normal_feature_sets.py (both model types and the three versions of the training data) sorted by mean
              accuracy. MGUS-MM tier: combinations of 05_two_tier_feature_sets.py whose tier-1 feature set is gm16, with the
              ratio of MGUS to MM accuracy, sorted by mean accuracy; the three best combinations with a mean accuracy above
              0.85 and an MGUS/MM ratio above 0.75, and above 0.70, are printed. The final classifiers use the
              cohort-smoothed feature sets mex:gm16+age+cyto (malignancy tier) and auc:gm16+arm (MGUS-MM tier). Run with
              PYTHONHASHSEED=0.

Inputs:       data/classifier/performance/init_model/parts/tumor-normal-classifier.csv (04_tumor_normal_feature_sets.py).
              data/classifier/performance/init_model/full/cross-combined-mgus-mm-classifier.csv
              (05_two_tier_feature_sets.py).

Outputs:      results/classifiers/feature_set_ranking_malignancy.tsv: gm16 feature sets by mean accuracy.
              results/classifiers/feature_set_ranking_mgus_mm.tsv: combinations with a gm16 tier-1 feature set, with the
              MGUS/MM accuracy ratio, by mean accuracy.

Dependencies: Python 3.8 with requirements.txt (pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import pandas as pd

OUT = result_dir("classifiers")
pd.set_option("display.width", 250)

malignancy = pd.read_csv(data("classifier", "performance", "init_model", "parts", "tumor-normal-classifier.csv"))
cross_2d = pd.read_csv(data("classifier", "performance", "init_model", "full", "cross-combined-mgus-mm-classifier.csv"))

# ---- malignancy tier: gm16 feature sets by mean accuracy -----------------------------------------------------------------
tn_gm16 = malignancy[malignancy["feature_set"].apply(lambda x: "gm16" in x)].sort_values("mean_acc", ascending=False)
tn_gm16.to_csv(os.path.join(OUT, "feature_set_ranking_malignancy.tsv"), sep="\t", index=False)
print("malignancy tier, gm16 feature sets (top 10):\n" + tn_gm16.head(n=10).to_string(index=False))

# ---- MGUS-MM tier: combinations with a gm16 tier-1 feature set ---------------------------------------------------------
cross_2d["MGUS/MM"] = cross_2d["MGUS_mean_acc"] / cross_2d["MM_mean_acc"]
cross_2d_gm16 = cross_2d[cross_2d["tn_feature"].apply(lambda x: "gm16" in x)].sort_values("mean_acc", ascending=False)
cross_2d_gm16.to_csv(os.path.join(OUT, "feature_set_ranking_mgus_mm.tsv"), sep="\t", index=False)
print("\ntwo-tier combinations, gm16 (top 10):\n" + cross_2d_gm16.head(n=10).to_string(index=False))
print("\nMGUS/MM accuracy ratio:\n" + cross_2d_gm16["MGUS/MM"].describe().to_string())
for ratio in (0.75, 0.70):
    top = cross_2d_gm16[(cross_2d_gm16["MGUS/MM"] > ratio) & (cross_2d_gm16["mean_acc"] > 0.85)].head(n=3)
    print("\nMGUS/MM > %.2f and mean accuracy > 0.85 (top 3):\n%s" % (ratio, top.to_string(index=False)))
