#!/usr/bin/env python3
"""Malignant MGUS PCs that the MGUS-MM classifier calls MM: test-set recall, out-of-sample scores and within-patient
comparisons (Fig. 4B,C; Extended Data Fig. 7A,B; Results and Methods).

Purpose:      1. Test-set cell-level recall of Fig. 4B,C: both random forests fit on the training cells (tuned parameters,
              seed 42) and applied to the test cells (malignancy tier: normal and malignant PCs; MGUS-MM tier: malignant
              PCs that the malignancy tier called malignant). 2. Out-of-sample scores for the malignant PCs of the 16
              MGUS and 36 MM patients of the classifier: 5-fold cross-validation grouped by patient for the training
              patients (same folds and models as 26_feature_ablation.py) and the model fit on all training patients for
              the test patients; malignancy score likewise (5 folds grouped by individual). A cell is MM-like when its
              MST score is above 0.5 (the classifier's call). 3. Per MGUS patient: percentage of MM-like cells and how
              concentrated the MM-like cells are among patients. 4. Within each MGUS patient with at least 20 MM-like and
              20 MGUS-like cells: median difference (MM-like minus MGUS-like) of the malignancy score, the 21 MGUS-MM
              features, UMI count, mitochondrial fraction and cell-cycle scores, in units of each variable's SD over all
              malignant cells; two-sided Wilcoxon signed-rank test across patients, Benjamini-Hochberg-adjusted; CNV:
              percentage of cells with a Numbat CNV call (posterior >= 0.9 for any segment); each cell also carries its
              Numbat clone (used by 09_stage_and_progression/11_mm_like_cells_out_of_sample.py). 5. Per MGUS patient:
              percentage of MM-like cells against BM PC infiltration, M-spike, FLC ratio, the number of malignant PCs and
              later progression (two-sided Spearman correlation and Wilcoxon rank-sum tests).

Inputs:       data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl (training cells of the test-set
              recall), sc-train-cohort-smoothed-knn30-expr_134samples.pkl, sc-test-cohort-smoothed-knn30-expr_47samples.pkl:
              per-cell features, redcap_id, annotation, disease_status, total_counts, pct_counts_mt, G2M_score, S_score.
              data/classifier/datasets/feature_columns/*.txt: feature lists of both tiers.
              data/cnv/plasma-cell-cell-level-cnv-prob-250306.pkl: per cell and CNV segment, p_cnv (Numbat posterior).
              data/cnv/plasma-cell-cell-level-filtered-clone-250306.pkl: per malignant PC, clone_id and numbat_annotation.
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id, patient_id.
              data/paper/cohort-clinical-metadata.csv: patient_id, sample_type, disease_status, BMBx, Mspike, FLC,
              progression_stage.
              data/paper/cytogenetics-status.csv: patient_id, cytogenetics (main malignant population).

Outputs:      results/classifiers/misclassified_mgus_test_recall.tsv: test-set cell-level recall per tier and true class.
              results/classifiers/misclassified_mgus_cells.pkl.gz: per malignant cell, patient, stage, split, scores, features.
              results/classifiers/misclassified_mgus_patients.tsv: per MGUS patient, % MM-like cells and clinical values.
              results/classifiers/misclassified_mgus_within_patient.tsv: MM-like minus MGUS-like differences, P and q.
              results/classifiers/misclassified_mgus_patient_associations.tsv: % MM-like cells vs clinical variables.
              results/classifiers/misclassified_mgus_log.txt: run log. All outputs carry Pt-### labels only.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn, statsmodels); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from statsmodels.stats.multitest import multipletests

OUT = result_dir("classifiers")
FH = open(os.path.join(OUT, "misclassified_mgus_log.txt"), "w")
RS = 42
DS_PARAMS = dict(n_estimators=200, max_features="sqrt", max_depth=25, min_samples_split=20, min_samples_leaf=10,
                 bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=RS)
TN_PARAMS = dict(n_estimators=100, max_features="sqrt", max_depth=30, min_samples_split=20, min_samples_leaf=30,
                 bootstrap=True, class_weight="balanced_subsample", n_jobs=-1, random_state=RS)
MIN_CELLS = 20


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
DS, TN = AUC16 + ARMS, MEX + CYTO
QC = ["total_counts", "pct_counts_mt", "G2M_score", "S_score"]


def short(f):
    return f.split(":")[-1].replace("_full_log_fc", " arm")


# ---- 1. test-set recall of Fig. 4B,C ------------------------------------------------------------------------------------------
def format_data(expr):
    """SMM removed; malignancy features z-scored and CNV-arm scores quantile-normalized within the dataset"""
    df = expr[~expr["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])].copy()
    df["class"] = np.where(df["annotation"] == "normal", "normal", df["disease_status"])
    X_tn = pd.DataFrame(StandardScaler().fit_transform(df[TN]), index=df.index, columns=TN)
    qt = QuantileTransformer(output_distribution="normal", random_state=RS)
    X_ds = df[AUC16].join(pd.DataFrame(qt.fit_transform(df[ARMS]), index=df.index, columns=ARMS))
    return df, X_tn, X_ds


d_tr, Xtn_tr, Xds_tr = format_data(pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl")))
d_te, Xtn_te, Xds_te = format_data(pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-knn30-expr_47samples.pkl")))
m_tn = RandomForestClassifier(**TN_PARAMS).fit(Xtn_tr.values, d_tr["annotation"].values)
tum = (d_tr["annotation"] == "tumor").values
m_ds = RandomForestClassifier(**DS_PARAMS).fit(Xds_tr.values[tum], d_tr["class"].values[tum])
p_tn = m_tn.predict(Xtn_te.values)
p_ds = m_ds.predict(Xds_te.values)
pred = np.where(p_tn == "tumor", p_ds, "normal")
truth = d_te["class"].values
rows = []
cm = confusion_matrix(np.where(truth == "normal", "normal", "tumor"), np.where(pred == "normal", "normal", "tumor"), labels=["normal", "tumor"])
for i, lab in enumerate(["normal", "tumor"]):
    rows.append(dict(tier="malignancy", true_class=lab, n_cells=cm[i].sum(), recall=cm[i, i] / cm[i].sum()))
rows.append(dict(tier="malignancy", true_class="all", n_cells=cm.sum(), recall=np.trace(cm) / cm.sum()))
k = truth != "normal"
cm2 = confusion_matrix(truth[k], pred[k], labels=["MGUS", "MM"])          # as in Fig. 4C: cells called normal are left out
for i, lab in enumerate(["MGUS", "MM"]):
    rows.append(dict(tier="MGUS-MM (cells called malignant)", true_class=lab, n_cells=cm2[i].sum(), recall=cm2[i, i] / cm2[i].sum()))
    rows.append(dict(tier="MGUS-MM (all malignant cells)", true_class=lab, n_cells=(truth == lab).sum(),
                     recall=np.mean(pred[truth == lab] == lab)))
CM = pd.DataFrame(rows)
CM.to_csv(os.path.join(OUT, "misclassified_mgus_test_recall.tsv"), sep="\t", index=False)
log("1. Test-set cell-level recall (Fig. 4B,C: paper 98.5% normal, 62.6% MGUS, 88.0% MM)")
log(CM.round(4).to_string(index=False))
log("test-set MGUS patients: %d; MM patients: %d" % (d_te.loc[d_te["class"] == "MGUS", "redcap_id"].nunique(),
                                                    d_te.loc[d_te["class"] == "MM", "redcap_id"].nunique()))

# ---- 2. out-of-sample scores of all malignant MGUS and MM cells ---------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
red2pt = meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"]
tr = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-knn30-expr_134samples.pkl"))
te = pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-knn30-expr_47samples.pkl"))
tr, te = (d[d["disease_status"].isin(["MGUS", "MM", "Donor"])].copy() for d in (tr, te))
for d in (tr, te):
    d["patient"] = d["redcap_id"].map(red2pt)
    assert d["patient"].notna().all()

# MST: as in 26_feature_ablation.py (CNV-arm scores quantile-normalized within each set; same folds)
mtr, mte = tr[(tr["annotation"] == "tumor") & (tr["disease_status"] != "Donor")], te[(te["annotation"] == "tumor") & (te["disease_status"] != "Donor")]


def format_ds(df):
    qt = QuantileTransformer(output_distribution="normal", random_state=RS)
    return df[AUC16].join(pd.DataFrame(qt.fit_transform(df[ARMS]), index=df.index, columns=ARMS))


Xtr, Xte = format_ds(mtr), format_ds(mte)
y = (mtr["disease_status"] == "MM").values.astype(int)
g = mtr["redcap_id"].astype(str).values
mst_tr = np.zeros(len(mtr))
for a, b in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Xtr, y, g):
    mst_tr[b] = RandomForestClassifier(**DS_PARAMS).fit(Xtr.values[a], y[a]).predict_proba(Xtr.values[b])[:, 1]
mst_te = RandomForestClassifier(**DS_PARAMS).fit(Xtr.values, y).predict_proba(Xte.values)[:, 1]

# malignancy score: as in 26_feature_ablation.py (mean expression z-scored with the training cells; folds grouped by individual)
ss = StandardScaler().fit(tr[MEX])
Ttr = pd.DataFrame(ss.transform(tr[MEX]), index=tr.index, columns=MEX).join(tr[CYTO])
Tte = pd.DataFrame(ss.transform(te[MEX]), index=te.index, columns=MEX).join(te[CYTO])
yt = (tr["annotation"] == "tumor").values.astype(int)
gt = tr["redcap_id"].astype(str).values
mal_tr = pd.Series(np.zeros(len(tr)), index=tr.index)
for a, b in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Ttr, yt, gt):
    mal_tr.iloc[b] = RandomForestClassifier(**TN_PARAMS).fit(Ttr.values[a], yt[a]).predict_proba(Ttr.values[b])[:, 1]
mal_te = pd.Series(RandomForestClassifier(**TN_PARAMS).fit(Ttr.values, yt).predict_proba(Tte.values)[:, 1], index=te.index)

cells = pd.concat([
    mtr[["patient", "disease_status"] + QC].assign(split="cross-validation", mst=mst_tr, malignancy=mal_tr.loc[mtr.index].values).join(Xtr),
    mte[["patient", "disease_status"] + QC].assign(split="test", mst=mst_te, malignancy=mal_te.loc[mte.index].values).join(Xte)])
cells["mm_like"] = cells["mst"] > 0.5
pm = cells.groupby(["patient", "disease_status", "split"])["mst"].mean().reset_index()
log("\n2. out-of-sample patient AUROC (check against the feature ablation, 0.861 / 0.917): cross-validation %.3f, test %.3f" % tuple(
    roc_auc_score(q["disease_status"] == "MM", q["mst"]) for _, q in pm.groupby("split")))
log("malignant cells: %s" % cells.groupby(["disease_status", "split"]).size().to_dict())

# CNV calls (Numbat): a cell is CNV-positive when p_cnv >= 0.9 for any segment; cells absent from the table have no call
cnv = pd.read_pickle(data("cnv", "plasma-cell-cell-level-cnv-prob-250306.pkl"))
pos = cnv.groupby(level=0)["p_cnv"].max() >= 0.9
cells["numbat_cell"] = cells.index.isin(pos.index)
cells["cnv_positive"] = cells.index.map(pos).fillna(False).astype(bool)
log("malignant cells found in the Numbat table: %.3f" % cells["numbat_cell"].mean())
# Numbat clone of each cell: clone number within its patient (1 = the diploid clone); "none" = no tumor clone assigned
cl = pd.read_pickle(data("cnv", "plasma-cell-cell-level-filtered-clone-250306.pkl"))[["clone_id", "numbat_annotation"]]
cl = cl[cl["numbat_annotation"] == "tumor"]["clone_id"].astype(str).str.rsplit(":", n=1).str[-1]
cells["clone"] = cells.index.map(cl).fillna("none").astype(str)
log("malignant cells with a Numbat tumor clone: %.3f" % (cells["clone"] != "none").mean())
cells.reset_index(drop=True).to_pickle(os.path.join(OUT, "misclassified_mgus_cells.pkl.gz"))       # index dropped: no cell barcodes

# ---- 3. MGUS patients: how many cells are MM-like, and where -------------------------------------------------------------------
mg = cells[cells["disease_status"] == "MGUS"]
P = mg.groupby(["patient", "split"]).agg(n_cells=("mst", "size"), pct_mm_like=("mm_like", lambda v: 100 * v.mean()),
                                         median_mst=("mst", "median"), mean_mst=("mst", "mean"),
                                         n_mm_like=("mm_like", "sum")).reset_index().sort_values("pct_mm_like", ascending=False)
P["share_of_all_mm_like_mgus_cells"] = 100 * P["n_mm_like"] / P["n_mm_like"].sum()
mmp = cells[cells["disease_status"] == "MM"].groupby("patient")["mm_like"].mean() * 100
log("\n3. MGUS patients (%d): %% MM-like cells per patient, median %.1f (range %.1f-%.1f); patients with > 50%%: %d"
    % (len(P), P["pct_mm_like"].median(), P["pct_mm_like"].min(), P["pct_mm_like"].max(), (P["pct_mm_like"] > 50).sum()))
log("all MGUS cells MM-like: %.1f%%; cross-validation %.1f%%, test %.1f%%" % (100 * mg["mm_like"].mean(),
    100 * mg.loc[mg["split"] == "cross-validation", "mm_like"].mean(), 100 * mg.loc[mg["split"] == "test", "mm_like"].mean()))
log("share of MM-like MGUS cells from the top 1 / top 3 patients: %.1f%% / %.1f%%"
    % (P["share_of_all_mm_like_mgus_cells"].iloc[0], P["share_of_all_mm_like_mgus_cells"].iloc[:3].sum()))
log("MM patients (%d): %% MM-like cells per patient, median %.1f (range %.1f-%.1f)" % (len(mmp), mmp.median(), mmp.min(), mmp.max()))
log(P.round(2).to_string(index=False))

# ---- 4. MM-like vs MGUS-like cells within MGUS patients ---------------------------------------------------------------------------
cells["log10_umi"] = np.log10(cells["total_counts"])
VARS = ["malignancy"] + DS + ["log10_umi", "pct_counts_mt", "G2M_score", "S_score"]
sd = cells[VARS].std()
both = [p for p, q in mg.groupby("patient") if q["mm_like"].sum() >= MIN_CELLS and (~q["mm_like"]).sum() >= MIN_CELLS]
mg = cells[(cells["disease_status"] == "MGUS") & cells["patient"].isin(both)]
rows = []
for v in VARS + ["cnv_positive"]:
    diff, mm_vals, mg_vals = [], [], []
    for p, q in mg.groupby("patient"):
        a, b = q.loc[q["mm_like"], v].astype(float), q.loc[~q["mm_like"], v].astype(float)
        if v == "cnv_positive":
            if not q["numbat_cell"].any():
                continue
            diff.append(100 * (a.mean() - b.mean()))
        else:
            diff.append((a.median() - b.median()) / sd[v])
        mm_vals.append(a.mean() if v == "cnv_positive" else a.median())
        mg_vals.append(b.mean() if v == "cnv_positive" else b.median())
    diff = np.array(diff)
    p = stats.wilcoxon(diff).pvalue if len(diff) >= 5 and np.any(diff != 0) else np.nan
    mm_pat = cells.loc[cells["disease_status"] == "MM"].groupby("patient")[v].median().median() if v != "cnv_positive" else \
        100 * cells.loc[(cells["disease_status"] == "MM") & cells["numbat_cell"].groupby(cells["patient"]).transform("any"), v].mean()
    rows.append(dict(variable=short(v), n_patients=len(diff), median_difference=np.median(diff),
                     n_patients_higher_in_mm_like=int((diff > 0).sum()), p=p,
                     median_mm_like=np.median(mm_vals), median_mgus_like=np.median(mg_vals), mm_patients_median=mm_pat))
W = pd.DataFrame(rows)
ok = W["p"].notna()
W.loc[ok, "q"] = multipletests(W.loc[ok, "p"], method="fdr_bh")[1]
W.to_csv(os.path.join(OUT, "misclassified_mgus_within_patient.tsv"), sep="\t", index=False)
log("\n4. MM-like minus MGUS-like cells within %d MGUS patients with >= %d of each (median difference in SD units;"
    " CNV: percentage points, patients with Numbat calls)" % (len(both), MIN_CELLS))
log(W.round(4).to_string(index=False))

# ---- 5. patient-level associations ----------------------------------------------------------------------------------------------
clin = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
clin = clin[(clin["disease_status"] == "MGUS") & (clin["sample_type"] == "BM")].drop_duplicates("patient_id").set_index("patient_id")
P2 = P.set_index("patient").join(clin[["BMBx", "Mspike", "FLC", "progression_stage"]])
P2["BMBx"] = pd.to_numeric(P2["BMBx"].astype(str).str.replace("<", "").str.replace("%", ""), errors="coerce")
P2["progressed"] = P2["progression_stage"].notna()
cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"), index_col=0)
P2["cytogenetics"] = cyto[cyto.index.str.endswith("tumor:0")].set_index("patient_id")["cytogenetics"].reindex(P2.index)
log("cytogenetics of MGUS patients with > 50%% vs <= 50%% MM-like cells: %s vs %s" % (
    P2.loc[P2["pct_mm_like"] > 50, "cytogenetics"].value_counts().to_dict(), P2.loc[P2["pct_mm_like"] <= 50, "cytogenetics"].value_counts().to_dict()))
rows = []
for v in ["BMBx", "Mspike", "FLC", "n_cells"]:
    x = P2[["pct_mm_like", v]].dropna()
    r, p = stats.spearmanr(x["pct_mm_like"], x[v]) if len(x) >= 5 and x[v].nunique() > 1 else (np.nan, np.nan)
    rows.append(dict(variable=v, test="Spearman", n=len(x), statistic=r, p=p))
a, b = P2.loc[P2["progressed"], "pct_mm_like"], P2.loc[~P2["progressed"], "pct_mm_like"]
rows.append(dict(variable="progressed (n=%d) vs not (n=%d), median %% MM-like %.1f vs %.1f" % (len(a), len(b), a.median(), b.median()),
                 test="Wilcoxon rank-sum", n=len(a) + len(b), statistic=np.nan,
                 p=stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else np.nan))
A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "misclassified_mgus_patient_associations.tsv"), sep="\t", index=False)
P2.reset_index().drop(columns=["progression_stage"]).to_csv(os.path.join(OUT, "misclassified_mgus_patients.tsv"), sep="\t", index=False)
log("\n5. %% MM-like cells per MGUS patient vs clinical features (BMBx distinct values: %s)" % sorted(P2["BMBx"].dropna().unique()))
log(A.round(4).to_string(index=False))
log("done")
