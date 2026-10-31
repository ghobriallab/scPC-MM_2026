#!/usr/bin/env python3
"""MST scores of the PCs of three external single-cell studies and the separation of disease stages, with effect sizes
and 95% CIs (Fig. 5C; Results, external validation).

Purpose:      The final MGUS-MM random forest scores the cells of Ledergor et al. 2018, Boiarsky et al. 2022 and Dang et
              al. 2023. External features (kNN-smoothed, k = 30) are scaled together with the MGUS, MM and donor cells of
              the classifier training and test sets (16 signature AUCell scores as they are, CNV-arm scores
              quantile-transformed) and passed in the training order. Four Dang et al. samples are excluded at the
              request of the study authors. Unit = sample x disease stage x cell annotation; unit score = mean MST score
              of its cells. Per study and pooled over within-study pairs, for malignant and normal PCs: n and median per
              stage; AUROC with a stratified bootstrap 95% CI (2,000 resamples) and two-sided Wilcoxon rank-sum P for
              MGUS vs MM, MGUS vs SMM and SMM vs MM; Spearman's rho of MST with stage (MGUS 1, SMM 2, MM 3) with a
              bootstrap 95% CI and P. Seed 42.

Inputs:       data/classifier/sc-disease-model-knn30-RF.pkl: MGUS-MM random forest (classes MGUS, MM).
              data/classifier/datasets/sc-train-cohort-smoothed-expr_134samples.pkl and
              sc-test-cohort-smoothed-expr_47samples.pkl: cohort cells (disease_status, annotation, feature columns).
              data/classifier/datasets/feature_columns/auc_16_gene_modules.txt, mean_expr_fc_driver_cnv_arms.txt: names
              of the MST features (auc:<signature>, CNV-arm log fold changes).
              data/signature/external/<study>_features-knn30.pkl: external cells, same feature columns.
              data/h5/external/<study>/scaled_umap.h5ad: obs sample_id, disease_status, annotation (malignant "tumor"
              or "normal", as labeled by the original studies); Dang et al. also has patient.

Outputs:      results/classifiers/external_mst_units.tsv: mean MST score and number of cells per unit.
              results/classifiers/external_mst_stats.tsv: per study (and pooled) and cell type, n, medians, AUROCs, P, rho.
              results/classifiers/external_mst_log.txt: run log.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn, h5py); config.py.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

RS = 42
NBOOT = 2000
rng = np.random.default_rng(RS)
OUT = result_dir("classifiers")
FH = open(os.path.join(OUT, "external_mst_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")
    FH.flush()


STUDIES = [  # name in the figure, feature file prefix, h5ad folder, samples to drop
    ("Ledergor et al.", "ledergor_natmedicine2018", "CA3Data_Ledergor2018", []),
    ("Boiarsky et al.", "boiarsky_natcommun2022", "Boiarsky_NatCommun2022", []),
    ("Dang et al.", "dang_cancercell2023", "Dang_CancerCell2023", ["MGUS02s", "MGUS18", "SMM02s", "SMM05s"]),
]
STAGES = ["MGUS", "SMM", "MM"]
PAIRS = [("MGUS", "MM"), ("MGUS", "SMM"), ("SMM", "MM")]


# ---- 1. MST scores of the external cells ---------------------------------------------------------------
def feature_list(columns, fname):
    """Feature names present in the data, in the data's column order."""
    names = [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", fname))]
    return list(columns.intersection(names))


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


model = pickle.load(open(data("classifier", "sc-disease-model-knn30-RF.pkl"), "rb"))
log("MGUS-MM model classes:", list(model.classes_))
for f in [("sc-disease-model-knn30-RF.pkl",), ("datasets", "sc-train-cohort-smoothed-expr_134samples.pkl"),
          ("datasets", "sc-test-cohort-smoothed-expr_47samples.pkl")]:
    log("  %s modified %s" % ("/".join(f), pd.Timestamp(os.path.getmtime(data("classifier", *f)), unit="s").strftime("%Y-%m-%d")))

# Reference cells: training and test cells of MGUS, MM and donors (SMM patients are not used to train the model)
ref = pd.concat([pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl")),
                 pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-expr_47samples.pkl"))])
ref = ref[~ref["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
ref_cols = pd.read_pickle(data("signature", "external", "ledergor_natmedicine2018_features-knn30.pkl")).columns
ref = ref[["annotation", "disease_status"] + list(ref_cols)]
log("reference cells:", len(ref))

canonical = [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", "auc_16_gene_modules.txt"))] + \
            [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", "mean_expr_fc_driver_cnv_arms.txt"))]
units = []
for study, feat, h5, drop in STUDIES:
    obs = read_obs(str(data("h5", "external", h5, "scaled_umap.h5ad")), ["sample_id", "patient", "disease_status", "annotation"])
    ext = pd.read_pickle(data("signature", "external", "%s_features-knn30.pkl" % feat))
    log("\n%s: %d cells; feature rows in obs order: %s" % (study, len(obs), list(ext.index) == list(obs.index)))
    if drop:
        n_dropped = obs.loc[obs["sample_id"].isin(drop), "sample_id"].nunique()
        obs = obs[~obs["sample_id"].isin(drop)]
        ext = ext.loc[obs.index]
        log("  removed samples excluded at the request of the study authors:", n_dropped, "-> cells", len(obs))
    ext = ext.join(obs[["annotation", "disease_status"]])
    # external and reference cells are transformed together
    df = pd.concat([ext, ref])
    assert df.index.is_unique
    auc, arms = feature_list(df.columns, "auc_16_gene_modules.txt"), feature_list(df.columns, "mean_expr_fc_driver_cnv_arms.txt")
    assert auc + arms == canonical, "feature order differs from training order"
    qt = QuantileTransformer(output_distribution="normal", random_state=RS)
    X = df[auc].join(pd.DataFrame(qt.fit_transform(df[arms]), index=df.index, columns=arms)).loc[ext.index]
    mst = model.predict_proba(X.values)[:, list(model.classes_).index("MM")]
    cells = obs.assign(mst=mst)
    cells = cells[cells["annotation"].isin(["tumor", "normal"]) & cells["disease_status"].isin(STAGES)]
    u = cells.groupby(["sample_id", "disease_status", "annotation"]).agg(n_cells=("mst", "size"), mst=("mst", "mean")).reset_index()
    u.insert(0, "study", study)
    units.append(u)
    for ann in ["tumor", "normal"]:
        x = u[u["annotation"] == ann]
        log("  %s units per stage: %s" % (ann, x["disease_status"].value_counts().reindex(STAGES).to_dict()))
    if "patient" in cells.columns:
        t = cells[cells["annotation"] == "tumor"].drop_duplicates(["sample_id", "patient", "disease_status"])
        log("  tumor units: %d samples from %d patients; patients with > 1 unit: %d"
            % (t["sample_id"].nunique(), t["patient"].nunique(), (t.groupby("patient").size() > 1).sum()))
units = pd.concat(units, ignore_index=True)
units.to_csv(os.path.join(OUT, "external_mst_units.tsv"), sep="\t", index=False)


# ---- 2. Effect sizes and confidence intervals ----------------------------------------------------------
def auroc(groups):
    """AUROC of the second group vs the first, pooled over studies: within-study pairs only.
    groups: list of (scores of the lower stage, scores of the higher stage), one per study.
    Equals the Mann-Whitney U statistic divided by the number of pairs (the Wilcoxon rank-sum effect size)."""
    u = sum(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]) * len(a) * len(b) for a, b in groups)
    return u / sum(len(a) * len(b) for a, b in groups)


def auroc_ci(groups):
    """Stratified bootstrap: units resampled within each study and stage."""
    boot = [auroc([(rng.choice(a, len(a)), rng.choice(b, len(b))) for a, b in groups]) for _ in range(NBOOT)]
    return auroc(groups), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def rho_ci(x):
    """Spearman's rho of MST with stage (MGUS 1, SMM 2, MM 3); units resampled within stage."""
    code = x["disease_status"].map({s: i + 1 for i, s in enumerate(STAGES)}).values
    r, p = stats.spearmanr(code, x["mst"].values)
    idx = [np.where(code == k)[0] for k in sorted(set(code))]
    boot = []
    for _ in range(NBOOT):
        i = np.concatenate([rng.choice(j, len(j)) for j in idx])
        boot.append(stats.spearmanr(code[i], x["mst"].values[i])[0])
    return r, np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5), p


rows = []
for ann in ["tumor", "normal"]:
    for study in [s[0] for s in STUDIES] + ["Pooled"]:
        x = units[units["annotation"] == ann]
        sub = [x[x["study"] == s] for s in ([study] if study != "Pooled" else [s[0] for s in STUDIES])]
        r = {"study": study, "annotation": ann}
        for s in STAGES:
            v = pd.concat([d.loc[d["disease_status"] == s, "mst"] for d in sub])
            r["n_" + s], r["median_" + s] = len(v), v.median()
        for lo, hi in PAIRS:
            g = [(d.loc[d["disease_status"] == lo, "mst"].values, d.loc[d["disease_status"] == hi, "mst"].values) for d in sub]
            g = [(a, b) for a, b in g if len(a) and len(b)]
            k = "%s_vs_%s" % (lo, hi)
            r["auroc_" + k], r["auroc_lo_" + k], r["auroc_hi_" + k] = auroc_ci(g)
            if study != "Pooled":
                r["wilcoxon_p_" + k] = stats.mannwhitneyu(g[0][1], g[0][0], alternative="two-sided").pvalue
        if study != "Pooled":
            r["rho"], r["rho_lo"], r["rho_hi"], r["rho_p"] = rho_ci(sub[0])
        rows.append(r)
        log("%s %s: n %s; medians %s; MGUS vs MM AUROC %.2f (%.2f-%.2f)%s%s" % (
            ann, study, [r["n_" + s] for s in STAGES], [round(r["median_" + s], 3) for s in STAGES],
            r["auroc_MGUS_vs_MM"], r["auroc_lo_MGUS_vs_MM"], r["auroc_hi_MGUS_vs_MM"],
            "" if study == "Pooled" else ", Wilcoxon P = %.2g" % r["wilcoxon_p_MGUS_vs_MM"],
            "" if study == "Pooled" else "; rho %.2f (%.2f-%.2f), P = %.2g" % (r["rho"], r["rho_lo"], r["rho_hi"], r["rho_p"])))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "external_mst_stats.tsv"), sep="\t", index=False)
log("done")
