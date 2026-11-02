#!/usr/bin/env python3
"""Malignancy scores of the PCs of three external single-cell studies compared with the malignant and normal PC labels of
the original authors (Extended Data Fig. 7C; Results, external validation).

Purpose:      The final malignancy random forest scores the cells of Ledergor et al. 2018, Boiarsky et al. 2022 and Dang
              et al. 2023. Its 26 features (mean expression of the 16 signatures, the SenMayo senescence set, then CCND1,
              CCND2, CCND3, MAF, MAFA, MAFB, MYC, NSD2, FGFR3) are z-scored together with the MGUS, MM and donor cells of
              the classifier training and test sets and passed to the model by name, in the training order. As a check
              of the feature order, scores with the features in the column order of the external tables are computed
              alongside (columns *_by_position). Four Dang et al. samples are excluded at the request of the study
              authors; cells labeled malignant in healthy donors (Ledergor et al.: the labels follow cell clusters) are
              excluded, since a healthy donor has no malignant PCs. Unit = sample x cell label; unit score = mean
              malignancy score of its cells. Per study and pooled (within-study pairs only): AUROC of malignant vs normal
              units with a stratified bootstrap 95% CI (2,000 resamples), cell-level AUROC with a 95% CI from resampling
              samples (1,000), sensitivity and specificity of the cell calls (malignancy score > 0.5), and the AUROC
              within each sample with at least 20 malignant and 20 normal PCs. Seed 42.

Inputs:       data/classifier/sc-tumor-model-knn30-RF.pkl: malignancy random forest (classes normal, tumor).
              data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl and
              sc-test-cohort-smoothed-expr_47samples.pkl: cohort reference cells (disease_status, annotation, features).
              data/classifier/datasets/feature_columns/mean_expr_16_gene_modules.txt, mean_expr_1_aging_signature.txt,
              cytogenetics_genes.txt: names and training order of the malignancy features.
              data/signature/external/<study>_features-knn30.pkl: external cells, same feature columns.
              data/h5/external/<study>/scaled_umap.h5ad: obs sample_id, disease_status, annotation (tumor, normal).

Outputs:      results/classifiers/external_malignancy_units.tsv: mean malignancy score and number of cells per unit.
              results/classifiers/external_malignancy_stats.tsv: per study and pooled, unit and cell AUROCs, sensitivity.
              results/classifiers/external_malignancy_log.txt: run log.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn, h5py); config.py.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RS = 42
rng = np.random.default_rng(RS)
OUT = result_dir("classifiers")
FH = open(os.path.join(OUT, "external_malignancy_log.txt"), "w")
STUDIES = [  # name in the figure, feature file prefix, h5ad folder, samples to drop
    ("Ledergor et al.", "ledergor_natmedicine2018", "CA3Data_Ledergor2018", []),
    ("Boiarsky et al.", "boiarsky_natcommun2022", "Boiarsky_NatCommun2022", []),
    ("Dang et al.", "dang_cancercell2023", "Dang_CancerCell2023", ["MGUS02s", "MGUS18", "SMM02s", "SMM05s"]),
]
MIN_WITHIN = 20


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def load_list(f):
    return [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", f))]


def read_obs(path, cols):
    """obs columns of an anndata >= 0.8 h5ad, read with h5py (anndata 0.7.8 cannot read these files)."""
    dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
    with h5py.File(path, "r") as f:
        g = f["obs"]
        idx = dec(g[g.attrs.get("_index", "_index")][...])
        out = {}
        for c in cols:
            if c not in g:
                continue
            it = g[c]
            if isinstance(it, h5py.Group) and "codes" in it:
                cats, codes = dec(it["categories"][...]), it["codes"][...]
                out[c] = [cats[i] if i >= 0 else np.nan for i in codes]
            else:
                out[c] = dec(it[...])
    return pd.DataFrame(out, index=idx)


# ---- 1. malignancy scores of external cells ---------------------------------------------------------------------------------
model = pickle.load(open(data("classifier", "sc-tumor-model-knn30-RF.pkl"), "rb"))
TN = load_list("mean_expr_16_gene_modules.txt") + load_list("mean_expr_1_aging_signature.txt") + load_list("cytogenetics_genes.txt")
assert model.n_features_in_ == len(TN) == 26 and list(model.classes_) == ["normal", "tumor"]
log("malignancy model: %d features, training order: %s" % (len(TN), ", ".join(f.replace("mean_expr:", "") for f in TN)))

ref = pd.concat([pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl")),
                 pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-expr_47samples.pkl"))])
ref = ref[~ref["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])][TN]
log("reference cells (MGUS, MM, donors of the training and test sets):", len(ref))

units, cells_all = [], []
for study, feat, h5, drop in STUDIES:
    obs = read_obs(str(data("h5", "external", h5, "scaled_umap.h5ad")), ["sample_id", "disease_status", "annotation"])
    ext = pd.read_pickle(data("signature", "external", "%s_features-knn30.pkl" % feat))
    assert list(ext.index) == list(obs.index)
    if drop:
        obs = obs[~obs["sample_id"].isin(drop)]
        ext = ext.loc[obs.index]
    df = pd.concat([ext[TN], ref])                                    # z-scored together with the cohort reference cells
    Z = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=TN).loc[ext.index]
    score = model.predict_proba(Z[TN].values)[:, 1]                   # features by name, training order
    pos = [c for c in ext.columns if c in TN]                         # by position: the external table's own column order
    by_pos = [c for c in pos if c.startswith("mean_expr:") and c != "mean_expr:SenMayo"] + ["mean_expr:SenMayo"] + \
             [c for c in pos if not c.startswith("mean_expr:")]       # (signatures, SenMayo, then genes)
    score_pos = model.predict_proba(Z[by_pos].values)[:, 1]
    log("\n%s: %d cells; translocation genes in the external table: %s" % (study, len(obs), [c for c in by_pos if ":" not in c]))
    c = obs.assign(malignancy=score, malignancy_by_position=score_pos, study=study)
    c = c[c["annotation"].isin(["tumor", "normal"])]
    bad = (c["disease_status"] == "Donor") & (c["annotation"] == "tumor")
    log("  cells labeled malignant in healthy donors, excluded: %d (%d samples)" % (bad.sum(), c.loc[bad, "sample_id"].nunique()))
    if bad.any():                                                     # for comparison: sample AUROC with these cells
        k = c.groupby(["sample_id", "annotation"])["malignancy"].mean().reset_index()
        log("  sample AUROC with these cells included: %.3f" % roc_auc_score(k["annotation"] == "tumor", k["malignancy"]))
    c = c[~bad]
    log("  cells with an author label: %d (malignant %d, normal %d); by stage: %s" % (
        len(c), (c["annotation"] == "tumor").sum(), (c["annotation"] == "normal").sum(),
        c.groupby(["disease_status", "annotation"]).size().to_dict()))
    cells_all.append(c)
    u = c.groupby(["sample_id", "disease_status", "annotation"]).agg(
        n_cells=("malignancy", "size"), malignancy=("malignancy", "mean"),
        malignancy_by_position=("malignancy_by_position", "mean")).reset_index()
    u.insert(0, "study", study)
    units.append(u)
cells_all = pd.concat(cells_all)
units = pd.concat(units, ignore_index=True)
units.to_csv(os.path.join(OUT, "external_malignancy_units.tsv"), sep="\t", index=False)


# ---- 2. performance ---------------------------------------------------------------------------------------------------------
def pooled_auroc(groups):
    """AUROC of malignant vs normal units, pooled over studies with within-study pairs only (as in 22_external_mst_scores.py)."""
    u = sum(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]) * len(a) * len(b) for a, b in groups)
    return u / sum(len(a) * len(b) for a, b in groups)


def unit_auroc_ci(groups, n=2000):
    boot = [pooled_auroc([(rng.choice(a, len(a)), rng.choice(b, len(b))) for a, b in groups]) for _ in range(n)]
    return pooled_auroc(groups), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def cell_auroc_ci(c, col, n=1000):
    """Cell-level AUROC; 95% CI from resampling samples (cells of a sample are not independent)."""
    y, s = (c["annotation"] == "tumor").values, c[col].values
    idx = [np.where(c["sample_id"].values == k)[0] for k in c["sample_id"].unique()]
    boot = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(len(idx), len(idx))])
        if 0 < y[j].sum() < len(j):
            boot.append(roc_auc_score(y[j], s[j]))
    return roc_auc_score(y, s), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


rows = []
for study in [s[0] for s in STUDIES] + ["Pooled"]:
    names = [study] if study != "Pooled" else [s[0] for s in STUDIES]
    c = cells_all[cells_all["study"].isin(names)]
    r = dict(study=study, n_samples=c["sample_id"].nunique(),
             n_malignant_cells=int((c["annotation"] == "tumor").sum()), n_normal_cells=int((c["annotation"] == "normal").sum()))
    # both score versions draw their bootstrap resamples from the same generator, in this order
    for col, tag in [("malignancy", ""), ("malignancy_by_position", "by_position_")]:
        x = units
        g = [(x.loc[(x["study"] == s) & (x["annotation"] == "normal"), col].values,
              x.loc[(x["study"] == s) & (x["annotation"] == "tumor"), col].values) for s in names]
        g = [(a, b) for a, b in g if len(a) and len(b)]
        r[tag + "n_malignant_units"], r[tag + "n_normal_units"] = sum(len(b) for a, b in g), sum(len(a) for a, b in g)
        r[tag + "unit_auroc"], r[tag + "unit_lo"], r[tag + "unit_hi"] = unit_auroc_ci(g)
        if study != "Pooled":
            r[tag + "cell_auroc"], r[tag + "cell_lo"], r[tag + "cell_hi"] = cell_auroc_ci(c, col)
        called = c[col] > 0.5
        r[tag + "sensitivity"] = called[c["annotation"] == "tumor"].mean()
        r[tag + "specificity"] = (~called[c["annotation"] == "normal"]).mean()
        for st in ["MGUS", "SMM", "MM"]:
            r[tag + "sensitivity_" + st] = called[(c["annotation"] == "tumor") & (c["disease_status"] == st)].mean()
        w = []
        for _, q in c.groupby("sample_id"):
            k = q["annotation"].value_counts()
            if k.get("tumor", 0) >= MIN_WITHIN and k.get("normal", 0) >= MIN_WITHIN:
                w.append(roc_auc_score(q["annotation"] == "tumor", q[col]))
        r[tag + "within_n_samples"], r[tag + "within_median"] = len(w), np.median(w) if w else np.nan
        r[tag + "within_min"], r[tag + "within_n_above_0.9"] = (np.min(w), int(np.sum(np.array(w) > 0.9))) if w else (np.nan, 0)
    rows.append(r)
    log("%s: samples %d; units malignant %d, normal %d; unit AUROC %.3f (%.3f-%.3f) [by position %.3f]; %s"
        "sensitivity %.3f, specificity %.3f [by position %.3f, %.3f]; within-sample AUROC median %.3f (min %.3f; %d samples, %d > 0.9)"
        % (study, r["n_samples"], r["n_malignant_units"], r["n_normal_units"], r["unit_auroc"], r["unit_lo"], r["unit_hi"],
           r["by_position_unit_auroc"],
           "" if study == "Pooled" else "cell AUROC %.3f (%.3f-%.3f) [by position %.3f]; " % (r["cell_auroc"], r["cell_lo"], r["cell_hi"], r["by_position_cell_auroc"]),
           r["sensitivity"], r["specificity"], r["by_position_sensitivity"], r["by_position_specificity"],
           r["within_median"], r["within_min"], r["within_n_samples"], r["within_n_above_0.9"]))
    log("  sensitivity by stage: MGUS %.3f, SMM %.3f, MM %.3f" % (r["sensitivity_MGUS"], r["sensitivity_SMM"], r["sensitivity_MM"]))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "external_malignancy_stats.tsv"), sep="\t", index=False)
log("\naccuracy of the cell calls with the features passed by position: %s" % {
    s: round(((cells_all.loc[cells_all["study"] == s, "malignancy_by_position"] > 0.5) ==
              (cells_all.loc[cells_all["study"] == s, "annotation"] == "tumor")).mean(), 3) for s in cells_all["study"].unique()})
log("done")
