#!/usr/bin/env python3
"""MGUS-MM classifier refit with and without the published NF-kB index as an extra feature (Results and Methods, MGUS-MM
classifier).

Purpose:      The MGUS-MM tier is refit as in the paper (AUCell scores of the 16 signatures + quantile-normalized CNV-arm
              scores, kNN-30 smoothed; malignant PCs of MGUS and MM patients; SMM excluded from training; tuned
              random-forest parameters, seed 42), once without and once with the NF-kB index (AUCell score of the 11
              genes of Annunziata et al., Cancer Cell 2007) as a 22nd feature. Both are compared on out-of-fold scores of
              the training patients (5-fold cross-validation grouped by patient), on the test patients (model trained on
              all training patients) and on SMM patients (never used for training; Spearman's rho with the IMWG 2/20/20
              risk group, risk.py). Patient score = mean MST of the patient's malignant PCs; AUROC MGUS vs MM with a stratified
              bootstrap 95% CI; paired bootstrap over patients for the out-of-fold AUROC difference (2,000 resamples);
              Spearman's rho between the SMM scores of the two models.

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-knn30-expr_134samples.pkl,
              sc-test-cohort-smoothed-knn30-expr_47samples.pkl: per-cell features (auc:<signature>,
              mean_expr:<arm>_full_log_fc, auc:NFkB_index), redcap_id (patient, used only to group), annotation,
              disease_status.
              data/classifier/datasets/feature_columns/auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt.
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id, patient_id (Pt-###; IMWG risk groups,
              risk.py).

Outputs:      results/classifiers/nfkb_index_model_comparison.tsv: per model, AUROCs with CIs, SMM rho, AUROC difference.
              results/classifiers/nfkb_index_model_log.txt: run log.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn); config.py, risk.py. The reported
              AUROCs need scikit-learn 1.0.2 (requirements.txt): other versions grow different trees from the same seed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer

OUT = result_dir("classifiers")
FH = open(os.path.join(OUT, "nfkb_index_model_log.txt"), "w")
RS = 42
rng = np.random.default_rng(RS)
SMM = ["LRSMM", "IRSMM", "HRSMM"]
NFKB = "auc:NFkB_index"
# tuned parameters of the MGUS-MM tier (Methods)
DS_PARAMS = dict(n_estimators=200, max_features="sqrt", max_depth=25, min_samples_split=20, min_samples_leaf=10,
                 bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=RS)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def load_list(name):
    return [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", name))]


AUC16, ARMS = load_list("auc_16_gene_modules.txt"), load_list("mean_expr_fc_driver_cnv_arms.txt")


def format_ds(df):
    """Features of the MGUS-MM tier: AUCell features as they are, CNV-arm scores quantile-normalized."""
    qt = QuantileTransformer(output_distribution="normal", random_state=RS)
    arms = pd.DataFrame(qt.fit_transform(df[ARMS]), index=df.index, columns=ARMS)
    return df[AUC16 + [NFKB]].join(arms)


RED2PT = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), usecols=["redcap_id", "patient_id"]).dropna() \
    .astype(str).drop_duplicates().set_index("redcap_id")["patient_id"]


def load(path):
    df = pd.read_pickle(path)
    df = df[df["annotation"] == "tumor"].copy()           # the MGUS-MM tier is trained and applied on malignant PCs
    df["disease_status"] = stages(df["redcap_id"].astype(str).map(RED2PT), ["BM"] * len(df), df["disease_status"])  # IMWG
    return df, format_ds(df)


tr, Xtr = load(data("classifier", "datasets", "sc-train-cohort-smoothed-knn30-expr_134samples.pkl"))
te, Xte = load(data("classifier", "datasets", "sc-test-cohort-smoothed-knn30-expr_47samples.pkl"))
FEATS = {"MST model (paper)": AUC16 + ARMS, "MST model + NF-kB index": AUC16 + ARMS + [NFKB]}
fit = tr["disease_status"].isin(["MGUS", "MM"]).values
y_cell = (tr["disease_status"] == "MM").values.astype(int)
groups = tr["redcap_id"].astype(str).values
log("training malignant PCs (MGUS/MM): %d cells, %d patients; test patients MGUS %d, MM %d"
    % (fit.sum(), len(set(groups[fit])), te.loc[te["disease_status"] == "MGUS", "redcap_id"].nunique(),
       te.loc[te["disease_status"] == "MM", "redcap_id"].nunique()))


def patient_scores(df, p):
    s = pd.DataFrame({"patient": df["redcap_id"].astype(str).values, "stage": df["disease_status"].astype(str).values, "p": p})
    return s.groupby(["patient", "stage"])["p"].mean().reset_index()


def auroc_ci(y, s, n=2000):
    y, s = np.asarray(y), np.asarray(s)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = [roc_auc_score(y[i], s[i]) for i in (np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))] for _ in range(n))]
    return roc_auc_score(y, s), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# out-of-fold scores of the training patients (same folds for both models)
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Xtr[fit], y_cell[fit], groups[fit]))
oof, test, smm, imp = {}, {}, {}, {}
for name, cols in FEATS.items():
    p = np.zeros(fit.sum())
    for a, b in folds:
        m = RandomForestClassifier(**DS_PARAMS).fit(Xtr[cols].values[fit][a], y_cell[fit][a])
        p[b] = m.predict_proba(Xtr[cols].values[fit][b])[:, 1]
    oof[name] = patient_scores(tr[fit], p)
    # model on all training patients: test patients and SMM patients (train + test files, never trained on)
    m = RandomForestClassifier(**DS_PARAMS).fit(Xtr[cols].values[fit], y_cell[fit])
    imp[name] = pd.Series(m.feature_importances_, index=cols).rank(ascending=False)
    tsel = te["disease_status"].isin(["MGUS", "MM"]).values
    test[name] = patient_scores(te[tsel], m.predict_proba(Xte[cols].values[tsel])[:, 1])
    s_tr, s_te = ~fit & tr["disease_status"].isin(SMM).values, te["disease_status"].isin(SMM).values
    smm[name] = pd.concat([patient_scores(tr[s_tr], m.predict_proba(Xtr[cols].values[s_tr])[:, 1]),
                           patient_scores(te[s_te], m.predict_proba(Xte[cols].values[s_te])[:, 1])])

rows = []
names = list(FEATS)
for name in names:
    o, t, s = oof[name], test[name], smm[name]
    a_o = auroc_ci((o["stage"] == "MM").astype(int), o["p"])
    a_t = auroc_ci((t["stage"] == "MM").astype(int), t["p"])
    rho, p_rho = stats.spearmanr(s["p"], s["stage"].map({g: i for i, g in enumerate(SMM)}))
    rows.append(dict(model=name, oof_patients="%d MGUS / %d MM" % ((o["stage"] == "MGUS").sum(), (o["stage"] == "MM").sum()),
                     oof_AUROC=a_o[0], oof_lo=a_o[1], oof_hi=a_o[2],
                     test_patients="%d MGUS / %d MM" % ((t["stage"] == "MGUS").sum(), (t["stage"] == "MM").sum()),
                     test_AUROC=a_t[0], test_lo=a_t[1], test_hi=a_t[2], SMM_patient_stage_units=len(s),
                     SMM_rho_risk_group=rho, SMM_P=p_rho, NFkB_importance_rank=imp[name].get(NFKB, np.nan)))
R = pd.DataFrame(rows)

# paired bootstrap over training patients for the difference in out-of-fold AUROC
o0, o1 = oof[names[0]].set_index(["patient", "stage"]), oof[names[1]].set_index(["patient", "stage"])
yy = (o0.index.get_level_values("stage") == "MM").astype(int)
i1, i0 = np.where(yy == 1)[0], np.where(yy == 0)[0]
diff = []
for _ in range(2000):
    i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
    diff.append(roc_auc_score(yy[i], o1["p"].values[i]) - roc_auc_score(yy[i], o0["p"].values[i]))
R["oof_AUROC_difference_vs_paper"] = [np.nan, roc_auc_score(yy, o1["p"]) - roc_auc_score(yy, o0["p"])]
R["difference_lo"] = [np.nan, np.percentile(diff, 2.5)]
R["difference_hi"] = [np.nan, np.percentile(diff, 97.5)]
s0, s1 = smm[names[0]].set_index(["patient", "stage"])["p"], smm[names[1]].set_index(["patient", "stage"])["p"]
R["SMM_scores_rho_with_paper_model"] = [np.nan, stats.spearmanr(s0, s1.reindex(s0.index))[0]]
R.to_csv(os.path.join(OUT, "nfkb_index_model_comparison.tsv"), sep="\t", index=False)
log(R.round(3).T.to_string())

FH.close()
