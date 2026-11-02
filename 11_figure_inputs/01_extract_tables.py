#!/usr/bin/env python3
"""Plot-ready, de-identified tables for the figures, from the processed objects (small inputs).

Purpose:      Each part repeats the data steps behind one group of panels and writes only what the figures draw;
              identifiers are the de-identified Pt-### / HD-## labels (sample, record and library IDs are never written).
              SMM risk groups of the tables written here follow the IMWG 2/20/20 model (risk.py).
              Parts (run all, or name some: python3 01_extract_tables.py wgs shap):
                tables     per-patient score and signature tables as CSV (Figs. 4, 5; Supplementary Figs. 3–5 and 7 and Extended Data Fig. 8)
                wgs        MST score against the WGS MM-like score (least squares), LASSO of the MST score on the WGS
                           alterations (LassoCV, 5 folds, random_state 42) and the classifier features correlated with the
                           residuals (Spearman, Benjamini-Hochberg) (Fig. 5E-G; Extended Data Fig. 9C,D)
                confusion  test-set confusion counts of the two random forest classifiers (Fig. 4B,C)
                shap       SHAP values and feature values of the training and test cells (Fig. 4B,C)
                hotspot    Hotspot local-correlation Z-scores and the signature genes (Fig. 3A,B)
                tuning     cross-validation accuracy of the feature sets, smoothing strategies and k (Extended Data Fig. 6A-C)
                colors     patient colors (Fig. 1C; Extended Data Fig. 1C)
                cohort     tracks, isotypes and PC counts of the 226 individuals (Fig. 1B); progression track from
                           risk.progression_reached(); 2/20/20 criterion values per individual and disease stage
                clones     CNVs of the Numbat clones of BM tumors by disease stage, Ward order within stage; a patient's
                           clone sets of two samples that share tissue and IMWG risk group are kept once
                           (Extended Data Fig. 1C)

Inputs:       data/paper: cohort-clinical-metadata.csv, wgs-mm-like-score-final.csv, wgs-mutations-final.csv,
              anndata/plasma-cell-GEX5-umap.h5ad (obs), classifier/ (patient_id_mean_probability.pkl,
              tissue_id_mean_probability.pkl, tissue_id_mean_X_ds_scores.pkl), signatures/per-sample-mean-expr-score.pkl,
              cnvs/ (clone-phylogeny-data.pkl, cytoband-genome-cnv-map.pkl), colors/patient_colors.csv;
              data/classifier: datasets/ (training and test sets, feature_columns/), the SHAP explanations of the two final
              models, performance/ and kNN_param_accuracy.csv; data/signature/hotspot: consensus-modules_local-correlation-
              z.pkl, consensus-modules_genes.csv; data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl;
              data/metadata/tumor-cohort-metadata-final-240325.csv, updated_progression.csv (through risk.py).

Outputs:      results/figure_inputs: patient_id_mean_probability.csv.gz, per-sample-mean-expr-score.csv.gz, wgs_*.tsv,
              confusion_counts.tsv, shap_{malignancy,mgus_mm}.npz, hotspot_z.npz, hotspot_genes.tsv, tuning_*.tsv,
              colors_patient_colors.tsv, cohort_{tracks,isotypes,counts}.tsv, clones_<stage>.npz, clones_rows.tsv,
              clones_bands.tsv, extract_tables_log.txt.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels, scikit-learn 1.0.2, h5py, seaborn,
              natsort, shap to read the explanations); config.py, risk.py.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import progression_reached, stages

OUT = str(result_dir("figure_inputs"))
LOG = open(os.path.join(OUT, "extract_tables_log.txt"), "a")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
ID_OK = re.compile(r"^(Pt|HD)-\d+$")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def check_ids(df, col, name):
    """Every identifier written must be a de-identified label."""
    bad = df[col].astype(str).map(lambda v: not ID_OK.match(v)).sum()
    assert bad == 0, "%s: %d identifiers are not Pt-###/HD-##" % (name, bad)


def save(df, name, id_col=None):
    if id_col is not None:
        check_ids(df, id_col, name)
    df.to_csv(os.path.join(OUT, name), sep="\t", index=False)
    log("wrote %s: %d rows, columns %s" % (name, len(df), list(df.columns)[:12]))


def obs_col(f, k):
    e = f["obs"][k]
    import h5py
    if isinstance(e, h5py.Group):                                     # categorical: codes + categories
        cats = e["categories"][:].astype(str)
        codes = e["codes"][:]
        return np.where(codes < 0, "nan", cats[np.maximum(codes, 0)])
    return e[:].astype(str)


def sample_to_patient(samples, meta):
    """Map tissue-sample IDs to patient labels through the clinical table (the column holding these IDs is found by
    coverage, so no ID is printed)."""
    samples = pd.Index(samples).astype(str)
    best = None
    for c in meta.columns:
        v = set(meta[c].dropna().astype(str))
        cov = np.mean([s in v for s in samples])
        if best is None or cov > best[1]:
            best = (c, cov)
    log("tissue-sample IDs found in clinical column %r (coverage %.3f)" % best)
    assert best[1] > 0.99
    m = meta[[best[0], "patient_id"]].dropna().astype(str).drop_duplicates().set_index(best[0])["patient_id"]
    assert not m.index.duplicated().any()
    return pd.Series(m.reindex(samples).values, index=samples)


# ---- per-patient tables --------------------------------------------------------------------------------------------
def part_tables():
    for name, parts in [("patient_id_mean_probability.csv.gz", ("paper", "classifier", "patient_id_mean_probability.pkl")),
                        ("per-sample-mean-expr-score.csv.gz", ("paper", "signatures", "per-sample-mean-expr-score.pkl"))]:
        df = pd.read_pickle(data(*parts))
        df.to_csv(os.path.join(OUT, name))
        log("wrote %s: %d rows" % (name, len(df)))


# ---- Fig. 5E-G, Extended Data Fig. 9C,D -----------------------------------------------------------------------------
def part_wgs():
    import statsmodels.api as sm
    from scipy.stats import spearmanr
    from sklearn.linear_model import LassoCV
    from statsmodels.stats.multitest import multipletests

    tissue_probs = pd.read_pickle(data("paper", "classifier", "tissue_id_mean_probability.pkl"))
    score = "MST Score" if "MST Score" in tissue_probs.columns else "scPC-MM Score"
    tissue_tumors = tissue_probs[tissue_probs["annotation"] == "tumor"].set_index("sample")
    mm_like = pd.read_csv(data("paper", "wgs-mm-like-score-final.csv"), index_col=0)
    samples = list(tissue_tumors.index.intersection(mm_like.index))
    d0 = tissue_tumors.loc[samples].join(mm_like.loc[samples])
    assert d0.shape[0] == 84, d0.shape
    model = sm.OLS(d0[score], sm.add_constant(d0["total_sig"])).fit()                     # MST ~ WGS MM-like score
    d0["lm_fitted"], d0["lm_resid"] = model.fittedvalues, model.resid

    def residual_correlation(d, columns, resid):
        res = []
        for col in columns:
            rho, p = spearmanr(d[col], d[resid])
            res.append((col, rho, p))
        res = pd.DataFrame(res, columns=["feature", "rho", "p"])
        res["fdr"] = multipletests(res["p"], method="fdr_bh")[1]
        return res.sort_values("fdr")

    xds = pd.read_pickle(data("paper", "classifier", "tissue_id_mean_X_ds_scores.pkl"))   # classifier features per tumor
    xds = tissue_probs[["annotation", "sample"]].join(xds)
    xds = xds[xds["annotation"] == "tumor"].set_index("sample")
    feats = [c for c in xds.columns if c != "annotation"]
    lm_res = residual_correlation(d0[["lm_resid"]].join(xds), feats, "lm_resid")
    r = lm_res.set_index("feature")
    assert abs(r.loc["auc:circulatory_capacity", "rho"] - (-0.528946)) < 1e-5            # expected value
    save(lm_res, "wgs_residual_features_lm.tsv")

    muts = pd.read_csv(data("paper", "wgs-mutations-final.csv"), index_col=0)              # alterations per tumor
    drop_cols, rename_cols = [], {}
    muts[muts == "Negative"] = 0
    muts[muts == "SNV/INDEL"] = 1
    for col in muts.columns:
        muts.loc[muts[col] == col, col] = 1
        values = muts[col].unique()
        if len(values) == 1 and values[0] == 0:
            drop_cols.append(col)
            continue
        ints, prefix = [], []
        for v in muts[col]:
            if isinstance(v, int):
                ints.append(v)
                continue
            num = re.sub("^[A-Za-z]*", "", v)
            ints.append(int(num))
            prefix.append(v.split(num)[0])
        if len(prefix) > 0:
            prefix = sorted(set(prefix))[0]
            muts[col] = ints
            rename_cols[col] = col + " [{}]".format(prefix)
    assert len(drop_cols) == 28
    muts = muts.drop(columns=drop_cols).rename(columns=rename_cols)
    d = tissue_tumors.loc[samples, ["Malignancy Score", score, "disease_status"]].join(muts.loc[samples])
    pmeta = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
    d["disease_status"] = stages(sample_to_patient(d.index, pmeta).values, ["BM"] * len(d), d["disease_status"].astype(str))
    X, y = d[muts.columns].values, d[score].values
    lasso = LassoCV(cv=5, random_state=42).fit(X, y)
    d["lasso_pred"] = lasso.predict(X)
    d["lasso_resid"] = y - d["lasso_pred"]
    coef = pd.Series(lasso.coef_, index=muts.columns)
    sel = coef[coef != 0].sort_values(ascending=False)
    names = [n.replace(" [SV]", " SV").split(" [")[0] for n in sel.index]
    expected = {"IRF4": 0.263867, "t(11;14)(CCND1)": 0.105107, "-12q24.33": -0.121570, "NSD2 SV": 0.081244, "BRAF": 0.000899}
    got = dict(zip(names, sel.values))
    assert len(sel) == 27 and all(abs(got[k] - v) < 1e-6 for k, v in expected.items()), "LASSO coefficients differ"
    mm_enrich = ["Trisomies", "t(MYC;IgH/K/L)", "KRAS", "NRAS", "FAM46C", "+1q", "-1p22.1", "-1p12", "-16q12.1", "-4p16.3",
                 "-16q", "-8p", "-1p32.3", "-14q23.3", "-14q31.1", "-8p23.3", "+1q21.2", "-4q34.3", "-17q21.2", "+3q26.2",
                 "+8q24.21", "12p13.33"]                                                      # alterations of the WGS MM-like score
    carriers = d[sel.index].apply(lambda s: int((s > 0).sum()))
    save(pd.DataFrame({"mutation": names, "coefficient": sel.values, "carriers": carriers.values,
                       "in_wgs_mm_like_score": [n in mm_enrich for n in names]}), "wgs_lasso_coef.tsv")
    rows = []
    for mut, nm in zip(sel.index, names):
        chunk = d.loc[d[mut] > 0, [score, "disease_status"]]
        rows += [(nm, v, s) for v, s in zip(chunk[score], chunk["disease_status"])]
    save(pd.DataFrame(rows, columns=["mutation", "mst", "disease_status"]), "wgs_lasso_carriers.tsv")
    lasso_res = residual_correlation(d[["lasso_resid"]].join(xds), feats, "lasso_resid")
    r = lasso_res.set_index("feature")
    assert abs(r.loc["auc:circulatory_capacity", "rho"] - (-0.500294)) < 1e-5             # expected value
    save(lasso_res, "wgs_residual_features_lasso.tsv")

    meta = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
    pt = sample_to_patient(d0.index, meta)
    out = pd.DataFrame({"patient_id": pt.values, "disease_status": d.loc[d0.index, "disease_status"].values,
                        "sample_type": d0["sample_type"].astype(str).values if "sample_type" in d0.columns else "BM",
                        "mst": d0[score].values, "malignancy": d.loc[d0.index, "Malignancy Score"].values,
                        "wgs_mm_like_score": d0["total_sig"].values, "lm_fitted": d0["lm_fitted"].values,
                        "lm_resid": d0["lm_resid"].values, "lasso_pred": d.loc[d0.index, "lasso_pred"].values,
                        "lasso_resid": d.loc[d0.index, "lasso_resid"].values})
    rho, p = spearmanr(out["wgs_mm_like_score"], out["mst"])
    log("Fig. 5E: n = %d, rho = %.4f, P = %.4g" % (len(out), rho, p))
    save(out, "wgs_samples.tsv", "patient_id")


# ---- Fig. 4B,C: confusion counts -------------------------------------------------------------------------------------
def part_confusion():
    import random
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    load = lambda name: [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", name))]

    def format_data(expr):
        df = expr.copy()
        df = df[~df["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
        df["class"] = df["disease_status"]
        df.loc[df["annotation"] == "normal", "class"] = "normal"
        mg, ag, cy = load("mean_expr_16_gene_modules.txt"), load("mean_expr_1_aging_signature.txt"), \
            load("cytogenetics_genes.txt")
        au, arm = load("auc_16_gene_modules.txt"), load("mean_expr_fc_driver_cnv_arms.txt")
        ss = StandardScaler().fit(df[mg + ag + cy])
        X_tn = pd.DataFrame(ss.transform(df[mg + ag + cy]), index=df.index, columns=mg + ag + cy)
        qt = QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE).fit(df[arm])
        X_ds = df[au].join(pd.DataFrame(qt.transform(df[arm]), index=df.index, columns=arm))
        y_tn = pd.Categorical(list(df["annotation"]), categories=["normal", "tumor"])
        return y_tn, np.array(df["class"]), X_tn, X_ds

    tn_params = dict(n_estimators=100, max_features="sqrt", max_depth=30, min_samples_split=20, min_samples_leaf=30,
                     bootstrap=True, class_weight="balanced_subsample", n_jobs=-1)
    ds_params = dict(n_estimators=200, max_features="sqrt", max_depth=25, min_samples_split=20, min_samples_leaf=10,
                     bootstrap=True, class_weight="balanced", n_jobs=-1)
    train = pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl"))
    y_tn, y_ds, X_tn, X_ds = format_data(train)
    m_tn = RandomForestClassifier(**tn_params).fit(X_tn, y_tn)
    keep = y_tn == "tumor"
    m_ds = RandomForestClassifier(**ds_params).fit(X_ds[keep], pd.Categorical(list(y_ds[keep]), categories=["MGUS", "MM"]))
    test = pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-knn30-expr_47samples.pkl"))
    t_tn, t_ds, T_tn, T_ds = format_data(test)
    pred = m_tn.predict(T_tn)
    pred_ds = m_ds.predict(T_ds)
    pred[pred == "tumor"] = pred_ds[pred == "tumor"]
    truth = np.array(t_tn)
    truth[truth == "tumor"] = t_ds[truth == "tumor"]
    rows = []
    a, b = np.where(truth == "normal", "normal", "tumor"), np.where(pred == "normal", "normal", "tumor")
    for (i, t), (j, p) in [(x, y) for x in enumerate(["normal", "tumor"]) for y in enumerate(["normal", "tumor"])]:
        rows.append(("malignancy", t, p, int(confusion_matrix(a, b, labels=["normal", "tumor"])[i, j])))
    k = truth != "normal"
    cm = confusion_matrix(truth[k], pred[k], labels=["MGUS", "MM"])
    for i, t in enumerate(["MGUS", "MM"]):
        for j, p in enumerate(["MGUS", "MM"]):
            rows.append(("mgus_mm", t, p, int(cm[i, j])))
    res = pd.DataFrame(rows, columns=["tier", "true", "predicted", "cells"])
    res["expected"] = [11651, 178, 389, 12611, 2325, 1392, 1070, 7824]                    # counts of Fig. 4B,C
    log("confusion counts as expected:", list(res["cells"]) == list(res["expected"]), list(res["cells"]))
    save(res, "confusion_counts.tsv")


# ---- Fig. 4B,C: SHAP -------------------------------------------------------------------------------------------------
def part_shap():
    import pickle
    for tier, name, n in [("malignancy", "sc-tumor-model-knn30-shap.pkl", 26), ("mgus_mm", "sc-disease-model-knn30-shap.pkl", 21)]:
        with open(data("classifier", name), "rb") as fh:
            ex = pickle.load(fh)
        vals = np.asarray(ex.values)
        values = np.asarray(ex.data)
        names = list(ex.feature_names) if ex.feature_names is not None else []
        log(tier, "SHAP values", vals.shape, "data", values.shape, "features", len(names))
        assert vals.ndim == 3 and vals.shape[1] == n and values.shape == vals.shape[:2]
        np.savez_compressed(os.path.join(OUT, "shap_%s.npz" % tier), shap=vals[:, :, 1].astype(np.float32),
                            value=values.astype(np.float32), feature=np.array(names, dtype=str))
        log("wrote shap_%s.npz" % tier)


# ---- Fig. 3A,B -------------------------------------------------------------------------------------------------------
def part_hotspot():
    z = pd.read_pickle(data("signature", "hotspot", "consensus-modules_local-correlation-z.pkl"))
    genes = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
    log("Hotspot Z", z.shape, "| gene table", genes.shape, list(genes.columns))
    np.savez_compressed(os.path.join(OUT, "hotspot_z.npz"), z=z.values.astype(np.float32),
                        rows=np.array(z.index.astype(str)), cols=np.array(z.columns.astype(str)))
    genes.reset_index().to_csv(os.path.join(OUT, "hotspot_genes.tsv"), sep="\t", index=False)
    log("wrote hotspot_z.npz, hotspot_genes.tsv")


# ---- Extended Data Fig. 6A-C ------------------------------------------------------------------------------------------
def part_tuning():
    for name, parts in [("tumor_normal", ("classifier", "performance", "init_model", "parts", "tumor-normal-classifier.csv")),
                        ("mgus_mm", ("classifier", "performance", "init_model", "full", "cross-combined-mgus-mm-classifier.csv")),
                        ("knn", ("classifier", "kNN_param_accuracy.csv"))]:
        df = pd.read_csv(data(*parts))
        log(name, df.shape, list(df.columns)[:15])
        df.to_csv(os.path.join(OUT, "tuning_%s.tsv" % name), sep="\t", index=False)


def part_colors():
    df = pd.read_csv(data("paper", "colors", "patient_colors.csv"))
    check_ids(df, df.columns[0], "patient_colors")
    df.to_csv(os.path.join(OUT, "colors_patient_colors.tsv"), sep="\t", index=False)
    log("copied patient colors:", df.shape)


# ---- Fig. 1B -----------------------------------------------------------------------------------------------------------
def part_cohort():
    import h5py
    meta = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
    with h5py.File(data("paper", "anndata", "plasma-cell-GEX5-umap.h5ad"), "r") as f:
        obs = pd.DataFrame({k: obs_col(f, k) for k in ["patient_id", "sample_type", "annotation", "cytogenetics",
                                                        "hrd_status", "has_vdj", "isotype_status"]})
    with_pcs = set(obs["patient_id"])
    dropped = sorted(set(meta["patient_id"].astype(str)) - with_pcs)
    assert len(dropped) == 1                                       # the profiled individual without PCs passing QC
    meta = meta[meta["patient_id"].astype(str).isin(with_pcs)].copy()
    meta["patient_id"] = meta["patient_id"].astype(str)
    meta["disease_status"] = stages(meta["patient_id"], meta["sample_type"], meta["disease_status"])
    rows = []
    for col, name in [("disease_status", "Disease stage"), ("sample_type", "Sample type")]:
        c = meta[["patient_id", col]].dropna().drop_duplicates()
        rows += [(p, name, str(v)) for p, v in zip(c["patient_id"], c[col])]
    reached = progression_reached()                                 # most advanced stage reached after sampling
    rows += [(p, "Progression", reached[p]) for p in sorted(set(meta["patient_id"]) & set(reached))]
    log("progression track:", pd.Series([reached[p] for p in set(meta["patient_id"]) & set(reached)]).value_counts().to_dict())
    for col, name in [("Age", "Age"), ("BMBx", "BMBx %"), ("Mspike", "M spike"), ("FLC", "FLC ratio")]:
        v = meta[["patient_id", "disease_status", col]].copy()
        if col == "BMBx":
            v.loc[v[col] == "<10", col] = 10
        v[col] = pd.to_numeric(v[col], errors="coerce")
        v = v.dropna()
        rows += [(p, name, x) for p, x in v.groupby("patient_id")[col].mean().items()]
        if col != "Age":                                            # 2/20/20 criteria: one value per SMM risk group
            sv = v[v["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])].groupby(["patient_id", "disease_status"])[col].mean()
            rows += [(p, name + " by stage", "%s|%r" % (st, x)) for (p, st), x in sv.items()]
    tumor = obs[obs["annotation"] == "tumor"]
    t = tumor.loc[tumor["cytogenetics"] != "HRD", ["patient_id", "cytogenetics"]].drop_duplicates()
    rows += [(p, "Translocation", v) for p, v in zip(t["patient_id"], t["cytogenetics"])]
    h = tumor[["patient_id", "hrd_status"]].drop_duplicates()
    rows += [(p, "HRD status", "HRD") for p, v in zip(h["patient_id"], h["hrd_status"]) if v == "Y"]
    tracks = pd.DataFrame(rows, columns=["patient_id", "track", "value"])
    save(tracks, "cohort_tracks.tsv", "patient_id")
    iso = obs[obs["has_vdj"].isin(["True", "1", "true"])].copy()
    iso["isotype_status"] = iso["isotype_status"].replace({"None": "Light chain only"})
    iso = iso[iso["isotype_status"] != "nan"]
    frac = iso.groupby(["patient_id", "isotype_status"]).size().unstack(fill_value=0)
    frac = frac.div(frac.sum(1), axis=0).reset_index()
    save(frac, "cohort_isotypes.tsv", "patient_id")
    counts = obs.groupby(["patient_id", "sample_type", "annotation"]).size().unstack([1, 2], fill_value=0)
    counts.columns = ["%s:%s" % (a, b) for a, b in counts.columns]
    counts = counts.reset_index()
    assert counts.drop(columns="patient_id").values.sum() == len(obs)
    save(counts, "cohort_counts.tsv", "patient_id")


# ---- Extended Data Fig. 1C ---------------------------------------------------------------------------------------------
def part_clones():
    import seaborn as sns
    clones = pd.read_pickle(data("paper", "cnvs", "clone-phylogeny-data.pkl"))
    clones = clones[~clones["GT"].isna()]
    clones["disease_status"] = stages(clones["patient_id"], clones["sample_type"], clones["disease_status"])
    band = pd.read_pickle(data("paper", "cnvs", "cytoband-genome-cnv-map.pkl"))
    # clone sets (one per sample group) of one patient that now share tissue and risk group: the same Numbat clones
    # (one run per patient), kept once
    profile = lambda idx: sorted((str(c), tuple(r)) for c, r in zip(clones.loc[idx, "clonality"], band.loc[idx].values.tolist()))
    drop = []
    for _, g in clones.groupby(["patient_id", "sample_type", "disease_status"]):
        sgs = sorted(g["sample_group"].unique())
        if len(sgs) > 1:
            ref = profile(g.index[g["sample_group"] == sgs[0]])
            for sg in sgs[1:]:
                assert profile(g.index[g["sample_group"] == sg]) == ref, "same clones and clonality"
                drop += list(g.index[g["sample_group"] == sg])
    clones = clones.drop(index=drop)
    log("clone rows kept once (merged sample groups):", len(drop))
    pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
    to_mm = set(pub.loc[pub["progression_stage"] == "MM", "patient_id"])
    had = clones["progressor"].notna()
    clones.loc[had, "progressor"] = clones.loc[had, "patient_id"].isin(to_mm).astype(float)
    cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
    meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
    cells["patient_id"] = cells["redcap_id"].astype(str).map(meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"])
    cells["sample_group"] = cells["disease_status"].astype(str) + ":" + cells["sample_type"].astype(str) + ":" + cells["patient_id"]
    cf = cells.loc[cells["annotation"] == "tumor", ["sample_group", "cytogenetics", "hrd_status"]].astype(str).value_counts().reset_index()
    clones["cytogenetics"], clones["hrd_status"] = np.nan, np.nan
    for sg, df in cf.groupby("sample_group"):
        clones.loc[clones["sample_group"] == sg, "cytogenetics"] = df["cytogenetics"].values[0]
        clones.loc[clones["sample_group"] == sg, "hrd_status"] = "HRD" if df["hrd_status"].values[0] == "Y" else np.nan
    rows, cid = [], {}
    for stage in ["MM", "HRSMM", "IRSMM", "LRSMM", "MGUS"]:
        sel = clones[(clones["disease_status"] == stage) & (clones["sample_type"] == "BM")]
        m = band.loc[sel.index]
        g = sns.clustermap(m, method="ward", row_cluster=True, col_cluster=False)            # row order of the heatmap
        order = list(m.index[g.dendrogram_row.reordered_ind])
        import matplotlib.pyplot as plt
        plt.close("all")
        for k, c in enumerate(order):
            cid[c] = "%s-%03d" % (stage, k)
        np.savez_compressed(os.path.join(OUT, "clones_%s.npz" % stage), cnv=m.loc[order].values.astype(np.int8),
                            clone=np.array([cid[c] for c in order]))
        for c in order:
            r = sel.loc[c]
            rows.append((cid[c], stage, str(r["patient_id"]), r["clonality"], r["progressor"], r["hrd_status"],
                         r["cytogenetics"]))
    out = pd.DataFrame(rows, columns=["clone", "stage", "patient_id", "clonality", "progressor", "hrd_status", "cytogenetics"])
    save(out, "clones_rows.tsv", "patient_id")
    pd.DataFrame({"band": band.columns.astype(str)}).to_csv(os.path.join(OUT, "clones_bands.tsv"), sep="\t", index=False)
    log("bands:", band.shape[1])


PARTS = {"tables": part_tables, "wgs": part_wgs, "confusion": part_confusion, "shap": part_shap, "hotspot": part_hotspot,
         "tuning": part_tuning, "colors": part_colors, "cohort": part_cohort, "clones": part_clones}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PARTS):
        log("==== part", name)
        PARTS[name]()
    LOG.close()
