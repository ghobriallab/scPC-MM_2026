#!/usr/bin/env python3
"""Feature ablation of the malignancy and MGUS-MM random forest classifiers (Extended Data Fig. 6E; Results and Methods).

Purpose:      Both tiers are refit with their tuned parameters (seed 42) on the training patients, with the full model
              refit in the same run as the baseline. MGUS-MM tier (the MST score; 21 features = AUCell scores of the 16
              signatures + 5 quantile-normalized CNV-arm scores, kNN-30 smoothed; malignant PCs of MGUS and MM patients):
              models without each feature, and without the lowest-ranked features that together hold at most 10% of the
              total mean |SHAP| (the saved SHAP table behind Fig. 4C). Performance = AUROC of the patient MST (mean over
              the patient's malignant PCs), MGUS vs MM: out-of-fold scores of the training patients (5-fold
              cross-validation grouped by patient, same folds for every model) and the test patients (model fit on all
              training patients); paired stratified bootstrap of the difference from the full model (2,000 resamples of
              patients). Malignancy tier (26 features = mean expression of the 16 signatures and the SenMayo senescence
              set, z-scored with the training cells, + expression of 9 translocation partner genes; all PCs of MGUS, MM
              and healthy donors): models without each signature, without the 9 translocation genes together, and
              without the lowest-ranked SHAP features (at most 10%). Performance = cell-level AUROC, malignant vs normal
              PCs, out-of-fold (5 folds grouped by individual) and test set; bootstrap of the difference resampling
              individuals (1,000 resamples). Runtime about 25 min on 16 cores.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-knn30-expr_134samples.pkl,
              sc-test-cohort-smoothed-knn30-expr_47samples.pkl: per-cell features (auc:<signature>, mean_expr:<signature>,
              mean_expr:SenMayo, mean_expr:<arm>_full_log_fc, translocation genes), redcap_id (individual, used only to
              group), annotation (tumor/normal), disease_status.
              data/classifier/datasets/feature_columns/*.txt: feature lists of both tiers.
              data/classifier/feature_importance/model-ds-shap-feature-importance.csv,
              model-tn-shap-feature-importance.csv: feature, mean_shap.

Outputs:      results/classifiers/feature_ablation_mst.tsv: MGUS-MM tier, per model, AUROCs with CIs and difference.
              results/classifiers/feature_ablation_malignancy.tsv: malignancy tier, per model, AUROCs, accuracy, difference.
              results/classifiers/feature_ablation_log.txt: run log.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn); config.py. The reported AUROCs need
              scikit-learn 1.0.2 (requirements.txt): other versions grow different trees from the same seed (for example,
              a full-model out-of-fold AUROC of 0.855 instead of 0.861).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

OUT = result_dir("classifiers")
FH = open(os.path.join(OUT, "feature_ablation_log.txt"), "w")
RS = 42
rng = np.random.default_rng(RS)
# tuned random-forest parameters of the two tiers (Methods), seed 42
DS_PARAMS = dict(n_estimators=200, max_features="sqrt", max_depth=25, min_samples_split=20, min_samples_leaf=10,
                 bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=RS)
TN_PARAMS = dict(n_estimators=100, max_features="sqrt", max_depth=30, min_samples_split=20, min_samples_leaf=30,
                 bootstrap=True, class_weight="balanced_subsample", n_jobs=-1, random_state=RS)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def load_list(name):
    return [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", name))]


AUC16, ARMS = load_list("auc_16_gene_modules.txt"), load_list("mean_expr_fc_driver_cnv_arms.txt")
MEX = load_list("mean_expr_16_gene_modules.txt") + load_list("mean_expr_1_aging_signature.txt")
CYTO = load_list("cytogenetics_genes.txt")
tr = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-knn30-expr_134samples.pkl"))
te = pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-knn30-expr_47samples.pkl"))
tr, te = (d[d["disease_status"].isin(["MGUS", "MM", "Donor"])].copy() for d in (tr, te))   # SMM not used for training
shap = {k: pd.read_csv(data("classifier", "feature_importance", "model-%s-shap-feature-importance.csv" % k),
                       index_col=0)["mean_shap"]
        for k in ("ds", "tn")}


def low_shap(s, cols, share=0.10):
    """lowest-ranked features whose mean |SHAP| sums to at most `share` of the total"""
    s = s.reindex(cols).sort_values()
    return list(s.index[(s.cumsum() / s.sum()) <= share])


def short(f):
    return f.split(":")[-1].replace("_full_log_fc", " arm")


def bootstrap_diff(stat, groups_by_stratum, base, alt, n):
    """stat(idx, scores) on resampled units; returns CI of base, alt and of alt - base (same resamples)"""
    b = []
    for _ in range(n):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups_by_stratum])
        b.append((stat(i, base), stat(i, alt)))
    b = np.array(b)
    return np.percentile(b, [2.5, 97.5], axis=0), np.percentile(b[:, 1] - b[:, 0], [2.5, 97.5])


# ---- 1. MGUS-MM tier ------------------------------------------------------------------------------------------------------
def format_ds(df):
    """Features of the MGUS-MM tier: AUCell features as they are, CNV-arm scores quantile-normalized"""
    qt = QuantileTransformer(output_distribution="normal", random_state=RS)
    return df[AUC16].join(pd.DataFrame(qt.fit_transform(df[ARMS]), index=df.index, columns=ARMS))


mtr, mte = tr[tr["annotation"] == "tumor"], te[te["annotation"] == "tumor"]
mtr, mte = mtr[mtr["disease_status"] != "Donor"], mte[mte["disease_status"] != "Donor"]
Xtr, Xte = format_ds(mtr), format_ds(mte)
y = (mtr["disease_status"] == "MM").values.astype(int)
g = mtr["redcap_id"].astype(str).values
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Xtr, y, g))
FULL = AUC16 + ARMS
low = low_shap(shap["ds"], FULL)
MODELS = {"all 21 features (paper)": FULL}
MODELS.update({"without " + short(f): [c for c in FULL if c != f] for f in FULL})
MODELS["without the %d lowest-SHAP features (%s)" % (len(low), ", ".join(short(f) for f in low))] = [c for c in FULL if c not in low]
log("MGUS-MM tier: %d training cells (%d patients), %d test cells (%d patients); lowest-SHAP features: %s (%.1f%% of total)"
    % (len(mtr), len(set(g)), len(mte), mte["redcap_id"].nunique(), ", ".join(short(f) for f in low),
       100 * shap["ds"][low].sum() / shap["ds"][FULL].sum()))


def patient_mean(df, p):
    s = pd.DataFrame({"patient": df["redcap_id"].astype(str).values, "stage": df["disease_status"].astype(str).values, "p": p})
    return s.groupby(["patient", "stage"])["p"].mean()


oof, test = {}, {}
for name, cols in MODELS.items():
    p = np.zeros(len(mtr))
    for a, b in folds:
        p[b] = RandomForestClassifier(**DS_PARAMS).fit(Xtr[cols].values[a], y[a]).predict_proba(Xtr[cols].values[b])[:, 1]
    oof[name] = patient_mean(mtr, p)
    m = RandomForestClassifier(**DS_PARAMS).fit(Xtr[cols].values, y)
    test[name] = patient_mean(mte, m.predict_proba(Xte[cols].values)[:, 1])
    log("  %s: out-of-fold AUROC %.3f" % (name, roc_auc_score(oof[name].index.get_level_values("stage") == "MM", oof[name])))


def auroc_units(i, s, yy):
    return roc_auc_score(yy[i], s[i])


rows = []
base = list(MODELS)[0]
for name in MODELS:
    row = dict(model=name, n_features=len(MODELS[name]))
    for key, sc in (("oof", oof), ("test", test)):
        s0, s1 = sc[base], sc[name].reindex(sc[base].index)
        yy = (s0.index.get_level_values("stage") == "MM").astype(int)
        strata = [np.where(yy == 0)[0], np.where(yy == 1)[0]]
        a0, a1 = roc_auc_score(yy, s0), roc_auc_score(yy, s1)
        ci, dci = bootstrap_diff(lambda i, s: auroc_units(i, s, yy), strata, s0.values, s1.values, 2000)
        row.update({key + "_n": "%d MGUS / %d MM" % ((yy == 0).sum(), yy.sum()), key + "_AUROC": a1, key + "_lo": ci[0, 1],
                    key + "_hi": ci[1, 1], key + "_difference": a1 - a0, key + "_diff_lo": dci[0], key + "_diff_hi": dci[1]})
    rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "feature_ablation_mst.tsv"), sep="\t", index=False)
log("\nMGUS-MM tier ablation:\n" + R.drop(columns=["oof_n", "test_n"]).round(3).to_string(index=False))

# ---- 2. malignancy tier ---------------------------------------------------------------------------------------------------
TN = MEX + CYTO
ss = StandardScaler().fit(tr[MEX])                                       # mean expression centered on the training cells
Ttr = pd.DataFrame(ss.transform(tr[MEX]), index=tr.index, columns=MEX).join(tr[CYTO])
Tte = pd.DataFrame(ss.transform(te[MEX]), index=te.index, columns=MEX).join(te[CYTO])
yt, yte = (tr["annotation"] == "tumor").values.astype(int), (te["annotation"] == "tumor").values.astype(int)
gt, gte = tr["redcap_id"].astype(str).values, te["redcap_id"].astype(str).values
tfolds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Ttr, yt, gt))
low_t = low_shap(shap["tn"], TN)
TMODELS = {"all 26 features (paper)": TN}
TMODELS.update({"without " + short(f): [c for c in TN if c != f] for f in MEX})
TMODELS["without the 9 translocation genes"] = MEX
TMODELS["without the %d lowest-SHAP features (%s)" % (len(low_t), ", ".join(short(f) for f in low_t))] = [c for c in TN if c not in low_t]
log("\nmalignancy tier: %d training cells (%d individuals), %d test cells; lowest-SHAP features: %s (%.1f%% of total)"
    % (len(tr), len(set(gt)), len(te), ", ".join(short(f) for f in low_t), 100 * shap["tn"][low_t].sum() / shap["tn"][TN].sum()))
toof, ttest = {}, {}
for name, cols in TMODELS.items():
    p = np.zeros(len(tr))
    for a, b in tfolds:
        p[b] = RandomForestClassifier(**TN_PARAMS).fit(Ttr[cols].values[a], yt[a]).predict_proba(Ttr[cols].values[b])[:, 1]
    toof[name] = p
    ttest[name] = RandomForestClassifier(**TN_PARAMS).fit(Ttr[cols].values, yt).predict_proba(Tte[cols].values)[:, 1]
    log("  %s: out-of-fold AUROC %.4f" % (name, roc_auc_score(yt, p)))


def cell_boot(sc, yy, gg, n=1000):
    """cell-level AUROC of every model on the same resamples of individuals"""
    units = np.unique(gg)
    idx = [np.where(gg == u)[0].astype(np.int32) for u in units]
    out = {name: [] for name in sc}
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(len(units), len(units))])
        for name, s in sc.items():
            out[name].append(roc_auc_score(yy[j], s[j]))
    return len(units), {k: np.array(v) for k, v in out.items()}


rows = []
tbase = list(TMODELS)[0]
boots = {"oof": cell_boot(toof, yt, gt), "test": cell_boot(ttest, yte, gte)}
for name in TMODELS:
    row = dict(model=name, n_features=len(TMODELS[name]))
    for key, sc, yy in (("oof", toof, yt), ("test", ttest, yte)):
        n_units, b = boots[key]
        a0, a1 = roc_auc_score(yy, sc[tbase]), roc_auc_score(yy, sc[name])
        d = b[name] - b[tbase]
        row.update({key + "_individuals": n_units, key + "_AUROC": a1, key + "_lo": np.percentile(b[name], 2.5),
                    key + "_hi": np.percentile(b[name], 97.5), key + "_accuracy": ((sc[name] > 0.5).astype(int) == yy).mean(),
                    key + "_difference": a1 - a0, key + "_diff_lo": np.percentile(d, 2.5), key + "_diff_hi": np.percentile(d, 97.5)})
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "feature_ablation_malignancy.tsv"), sep="\t", index=False)
log("\nmalignancy tier ablation:\n" + T.round(4).to_string(index=False))
log("done")
FH.close()
