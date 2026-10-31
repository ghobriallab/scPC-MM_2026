#!/usr/bin/env python3
"""MGUS-MM classification with signatures restricted to genes selected without comparing disease stages (Results and
Methods, MGUS-MM classifier).

Purpose:      1. A signature gene is called stage-free when it is significant (padj < 0.01, |log2 fold change| > 1) in a
              pooled pseudobulk DESeq2 comparison that does not involve disease stages: malignant vs normal PCs (all
              patients), patients' vs donors' normal PCs, BM vs PB normal PCs, BM vs PB malignant PCs.
              2. Per-cell signature activity (mean log-normalized expression) is computed twice: with all signature genes
              and with the stage-free genes only.
              3. The same random forest (MGUS-MM parameters, seed 42; 16 signature features, z-scored within each
              dataset; no CNV-arm features in either version, as they involve no gene selection) is trained on the
              training patients of the classifier split and evaluated per patient: out-of-fold scores of the training
              patients (5 folds grouped by patient), test patients, SMM patients (never trained on; Spearman's rho with
              the IMWG 2/20/20 risk group) and the external units of Fig. 5C (MGUS vs MM AUROC pooled over within-study
              pairs). Paired stratified bootstrap of the differences (stage-free minus all genes, 2,000 resamples), units
              resampled within stage (and study).

Inputs:       data/signature/hotspot/consensus-modules_genes.csv: signature genes (index), column context = signature.
              data/signature/hotspot/mm-BM-PB-4980-DEs_module_assignment.csv: the 4,980 candidate genes (index).
              data/pseudobulk/result/whole_de and GEX5_only: four pooled DESeq2 result tables (padj, log2FoldChange).
              data/signature/hotspot/mm-BM-PB-4980-DEs/mm-BM-PB-4980-DEs_hotspot_input.h5ad: layers/counts (CSR, cells x
              4,980 genes); obs gex_id, barcode, redcap_id, sample_type, total_counts (whole-transcriptome UMIs).
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: gex_id, barcode, annotation (tumor/normal), disease_status.
              data/classifier/datasets/sc-train-cohort-smoothed-knn30-expr_134samples.pkl, sc-test-...47samples.pkl:
              redcap_id, disease_status (training/test split; SMM units).
              data/h5/external/Ledergor2018-Boiarsky2022-Dang2023_log1p_count.h5ad: X (CSR, log-normalized), obs source,
              sample_id, disease_status, annotation.
              results/classifiers/external_mst_units.tsv (22_external_mst_scores.py): study, sample_id, disease_status.

Outputs:      results/classifiers/stage_free_signature_genes.tsv: signature, gene, stage_free, comparisons that select it.
              results/classifiers/stage_free_signature_counts.tsv: genes per signature, all and stage-free.
              results/classifiers/stage_free_model_comparison.tsv: all vs stage-free genes per evaluation set, CIs.
              results/classifiers/stage_free_signature_test_log.txt: run log.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, scikit-learn, h5py); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

RS = 42
NBOOT = 2000
rng = np.random.default_rng(RS)
OUT = result_dir("classifiers")
FH = open(os.path.join(OUT, "stage_free_signature_test_log.txt"), "w")
SMM = ["LRSMM", "IRSMM", "HRSMM"]
# tuned parameters of the MGUS-MM tier (as in 24_nfkb_index_model.py)
DS_PARAMS = dict(n_estimators=200, max_features="sqrt", max_depth=25, min_samples_split=20, min_samples_leaf=10,
                 bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=RS)
FREE = {  # pooled pseudobulk comparisons that do not involve disease stages
    "malignant vs normal PCs, all patients":
        data("pseudobulk", "result", "whole_de", "tumor_vs_normal", "BM_tumor_n205_vs_BM_normal_n170.res.csv"),
    "patients' vs donors' normal PCs":
        data("pseudobulk", "result", "GEX5_only", "normal_vs_donor", "BM_normal_n160_vs_BM_donor_n28.res.csv"),
    "BM vs PB normal PCs":
        data("pseudobulk", "result", "GEX5_only", "BM_npc_vs_PB_npc", "BM_npc_GEX5_n187_vs_PB_npc_GEX5_n85.res.csv"),
    "BM vs PB malignant PCs":
        data("pseudobulk", "result", "GEX5_only", "BM_vs_PB_tumor", "BM_tumor_pair_n77_vs_PB_tumor_pair_n53.res.csv")}
STUDY = {"Ledergor_etal_NatMed2018": "Ledergor et al.", "Boiarsky_etal_NatCommun2022": "Boiarsky et al.",
         "Dang_etal_CancerCell2023": "Dang et al."}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def obs_col(g, k):
    e = g[k]
    if isinstance(e, h5py.Group):                                    # categorical: codes + categories
        cats, codes = e["categories"][:].astype(str), e["codes"][:]
        return np.where(codes < 0, "nan", cats[np.maximum(codes, 0)])
    return e[:].astype(str)


def csr_rows(grp, rows, ncol, chunk=20000):
    """rows (sorted) of an h5ad CSR group, read in chunks"""
    ip = grp["indptr"][:]
    out = []
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        a, b = ip[r[0]], ip[r[-1] + 1]
        vals, ind = grp["data"][a:b], grp["indices"][a:b]
        sub = sp.csr_matrix((vals, ind, ip[r[0]:r[-1] + 2] - a), shape=(r[-1] - r[0] + 1, ncol))
        out.append(sub[r - r[0]])
    return sp.vstack(out).tocsr()


# ---- 1. stage-free signature genes ---------------------------------------------------------------------------------------
pool = set(pd.read_csv(data("signature", "hotspot", "mm-BM-PB-4980-DEs_module_assignment.csv"), index_col=0).index)
sig = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
sig = pd.DataFrame({"gene": sig.index.astype(str), "signature": sig["context"].astype(str).values})
via = {g: [] for g in pool}
for name, f in FREE.items():
    d = pd.read_csv(f, index_col=0)
    for g in d[(d["padj"] < 0.01) & (d["log2FoldChange"].abs() > 1)].index:
        if g in via:
            via[g].append(name)
sig["stage_free"] = sig["gene"].map(lambda g: len(via.get(g, [])) > 0)
sig["stage_free_comparisons"] = sig["gene"].map(lambda g: "; ".join(via.get(g, [])))
sig.to_csv(os.path.join(OUT, "stage_free_signature_genes.tsv"), sep="\t", index=False)
cnt = sig.groupby("signature")["stage_free"].agg(genes="size", stage_free="sum")
cnt.to_csv(os.path.join(OUT, "stage_free_signature_counts.tsv"), sep="\t")
SIGS = list(cnt.index)
log("signature genes %d (%d unique), stage-free %d (%.1f%%); per signature %d-%d%% stage-free"
    % (len(sig), sig["gene"].nunique(), sig["stage_free"].sum(), 100 * sig["stage_free"].mean(),
       100 * (cnt["stage_free"] / cnt["genes"]).min(), 100 * (cnt["stage_free"] / cnt["genes"]).max()))
ARMS = {"all signature genes": sig, "stage-free genes only": sig[sig["stage_free"]]}


def activities(X, genes, arm):
    """mean log-normalized expression of each signature's genes (zeros included) per cell"""
    gi = {g: i for i, g in enumerate(genes)}
    W = np.zeros((len(genes), len(SIGS)))
    for j, s in enumerate(SIGS):
        idx = [gi[g] for g in ARMS[arm].loc[ARMS[arm]["signature"] == s, "gene"] if g in gi]
        W[idx, j] = 1.0 / len(idx)
    return np.asarray(X @ W)


# ---- 2. cohort cells: malignant BM PCs of the classifier units (Hotspot cells, <= 1,000 per individual and tissue) -------
units = []
for split, f in (("train", data("classifier", "datasets", "sc-train-cohort-smoothed-knn30-expr_134samples.pkl")),
                 ("test", data("classifier", "datasets", "sc-test-cohort-smoothed-knn30-expr_47samples.pkl"))):
    d = pd.read_pickle(f)[["redcap_id", "disease_status"]].astype(str).drop_duplicates()
    units.append(d.assign(split=split))
units = pd.concat(units)
units.loc[units["disease_status"].isin(SMM), "split"] = "SMM"           # SMM never used for training
units = units[units["disease_status"].isin(["MGUS", "MM"] + SMM)].drop_duplicates(["redcap_id", "disease_status"])
H = h5py.File(str(data("signature", "hotspot", "mm-BM-PB-4980-DEs", "mm-BM-PB-4980-DEs_hotspot_input.h5ad")), "r")
hobs = pd.DataFrame({k: obs_col(H["obs"], k) for k in ("gex_id", "barcode", "redcap_id", "sample_type")})
hobs["total_counts"] = H["obs"]["total_counts"][:]
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[["gex_id", "barcode", "annotation", "disease_status"]]
ann = ann.astype(str).drop_duplicates(["gex_id", "barcode"])
hobs = hobs.merge(ann, on=["gex_id", "barcode"], how="left")
assert hobs["annotation"].notna().all()
keep = hobs.merge(units, on=["redcap_id", "disease_status"], how="left")["split"].values
sel = np.where((hobs["sample_type"].values == "BM") & (hobs["annotation"].values == "tumor") & pd.notna(keep))[0]
cells = hobs.iloc[sel].assign(split=keep[sel]).reset_index(drop=True)
hgenes = obs_col(H["var"], "_index")
C = csr_rows(H["layers"]["counts"], sel, len(hgenes)).astype(np.float64)
C = sp.diags(1e4 / cells["total_counts"].values) @ C                    # normalized to 10,000 UMIs (whole transcriptome)
C.data = np.log1p(C.data)
log("cohort cells: %s; units: %s" % (cells["split"].value_counts().to_dict(),
                                     cells.drop_duplicates(["redcap_id", "disease_status"])["split"].value_counts().to_dict()))

# ---- 3. external malignant PCs, the units of Fig. 5C ----------------------------------------------------------------------
E = h5py.File(str(data("h5", "external", "Ledergor2018-Boiarsky2022-Dang2023_log1p_count.h5ad")), "r")
eobs = pd.DataFrame({k: obs_col(E["obs"], k) for k in ("source", "sample_id", "disease_status", "annotation")})
eobs["study"] = eobs["source"].map(STUDY)
ref = pd.read_csv(result("classifiers", "external_mst_units.tsv"), sep="\t")
ref = ref[ref["annotation"] == "tumor"][["study", "sample_id", "disease_status"]].astype(str)
esel = eobs.reset_index().merge(ref, on=["study", "sample_id", "disease_status"])
esel = esel[esel["annotation"] == "tumor"].sort_values("index")
egenes = obs_col(E["var"], "_index")
EX = csr_rows(E["X"], esel["index"].values, len(egenes)).astype(np.float64)
log("external malignant cells per study: %s; units: %d (external MST units: %d)"
    % (esel["study"].value_counts().to_dict(), esel.drop_duplicates(["study", "sample_id", "disease_status"]).shape[0], len(ref)))
for arm in ARMS:
    g = set(ARMS[arm]["gene"])
    log("%s: %d genes, %d measured in the external data" % (arm, len(g), len(g & set(egenes))))


# ---- 4. models and evaluation ---------------------------------------------------------------------------------------------
def z(a):
    return (a - a.mean(0)) / a.std(0)


def unit_scores(df, p, keys):
    return df.assign(p=p).groupby(keys)["p"].mean().reset_index()


def auroc_ci_paired(s0, s1, y, strata=None):
    """AUROC (MM vs MGUS) for both versions, stratified bootstrap CIs and CI of the difference (same resamples)"""
    y = np.asarray(y)
    strata = np.zeros(len(y), int) if strata is None else np.asarray(strata)
    grp = [np.where((strata == s) & (y == k))[0] for s in np.unique(strata) for k in (0, 1)]

    def pooled(s, i):
        # AUROC pooled over within-stratum pairs (as in 22_external_mst_scores.py): Mann-Whitney U summed over strata
        num = den = 0.0
        for st in np.unique(strata[i]):
            j = i[strata[i] == st]
            a, b = s[j][y[j] == 0], s[j][y[j] == 1]
            if len(a) and len(b):
                num += roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]) * len(a) * len(b)
                den += len(a) * len(b)
        return num / den
    allidx = np.arange(len(y))
    est = [pooled(s0, allidx), pooled(s1, allidx)]
    boot = []
    for _ in range(NBOOT):
        i = np.concatenate([rng.choice(g, len(g)) for g in grp if len(g)])
        boot.append((pooled(s0, i), pooled(s1, i)))
    boot = np.array(boot)
    return est, np.percentile(boot, [2.5, 97.5], axis=0), np.percentile(boot[:, 1] - boot[:, 0], [2.5, 97.5])


Xc = {arm: z(activities(C, list(hgenes), arm)) for arm in ARMS}
Xe = {}
for arm in ARMS:
    a = activities(EX, list(egenes), arm)
    Xe[arm] = np.vstack([z(a[(esel["study"] == s).values]) for s in STUDY.values()])
esel = pd.concat([esel[esel["study"] == s] for s in STUDY.values()])        # same order as the stacked features
tr = (cells["split"] == "train").values
y = (cells["disease_status"] == "MM").values.astype(int)
groups = cells["redcap_id"].values
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Xc[list(ARMS)[0]][tr], y[tr], groups[tr]))
res = {}
for arm in ARMS:
    oof = np.zeros(tr.sum())
    for a, b in folds:
        m = RandomForestClassifier(**DS_PARAMS).fit(Xc[arm][tr][a], y[tr][a])
        oof[b] = m.predict_proba(Xc[arm][tr][b])[:, 1]
    m = RandomForestClassifier(**DS_PARAMS).fit(Xc[arm][tr], y[tr])
    res[arm] = dict(oof=unit_scores(cells[tr], oof, ["redcap_id", "disease_status"]),
                    test=unit_scores(cells[cells["split"] == "test"], m.predict_proba(Xc[arm][(cells["split"] == "test").values])[:, 1],
                                     ["redcap_id", "disease_status"]),
                    smm=unit_scores(cells[cells["split"] == "SMM"], m.predict_proba(Xc[arm][(cells["split"] == "SMM").values])[:, 1],
                                    ["redcap_id", "disease_status"]),
                    ext=unit_scores(esel[esel["disease_status"].isin(["MGUS", "MM"])],
                                    m.predict_proba(Xe[arm][esel["disease_status"].isin(["MGUS", "MM"]).values])[:, 1],
                                    ["study", "sample_id", "disease_status"]))

A0, A1 = list(ARMS)
rows = []
for key, label, strata in (("oof", "training patients, out-of-fold", None), ("test", "held-out test patients", None),
                           ("ext", "external studies, pooled", "study")):
    u0, u1 = res[A0][key], res[A1][key]
    k = ["study", "sample_id", "disease_status"] if key == "ext" else ["redcap_id", "disease_status"]
    u = u0.merge(u1, on=k, suffixes=("_0", "_1"))
    yy = (u["disease_status"] == "MM").astype(int).values
    est, ci, dci = auroc_ci_paired(u["p_0"].values, u["p_1"].values, yy, None if strata is None else u[strata].values)
    rows.append(dict(evaluation=label, measure="MGUS vs MM AUROC", n_MGUS=int((yy == 0).sum()), n_MM=int(yy.sum()),
                     all_genes=est[0], all_lo=ci[0, 0], all_hi=ci[1, 0], stage_free=est[1], free_lo=ci[0, 1], free_hi=ci[1, 1],
                     difference=est[1] - est[0], diff_lo=dci[0], diff_hi=dci[1]))
s0, s1 = res[A0]["smm"], res[A1]["smm"]
u = s0.merge(s1, on=["redcap_id", "disease_status"], suffixes=("_0", "_1"))
code = u["disease_status"].map({g: i for i, g in enumerate(SMM)}).values
r0, p0 = stats.spearmanr(code, u["p_0"])
r1, p1 = stats.spearmanr(code, u["p_1"])
rows.append(dict(evaluation="SMM patients (never trained on)", measure="Spearman rho with 2/20/20 risk group", n_MGUS=np.nan,
                 n_MM=np.nan, all_genes=r0, all_lo=np.nan, all_hi=p0, stage_free=r1, free_lo=np.nan, free_hi=p1,
                 difference=r1 - r0, diff_lo=np.nan, diff_hi=np.nan))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "stage_free_model_comparison.tsv"), sep="\t", index=False)
log("\n" + R.round(3).to_string(index=False))
log("(SMM row: all_hi and free_hi hold the P values; n SMM units %d)" % len(u))
FH.close()
