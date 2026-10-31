#!/usr/bin/env python3
"""A compact MGUS-MM classifier that keeps only the top-ranked genes of each signature (Results and Methods, MGUS-MM
classifier).

Purpose:      Set up as in 25_stage_free_signature_test.py so that both analyses line up: 16 features = per-cell mean
              log-normalized expression of each signature's genes (no AUCell, no CNV-arm features, so the gene count is
              exact), z-scored within each dataset; random forest with the MGUS-MM parameters (seed 42); the training/test
              split of the classifier; malignant BM PCs of the classifier samples and the external units of Fig. 5C.
              Compact versions keep the top N genes of each signature ranked by Hotspot Z-score, N = 5, 10, 20, versus all
              genes. Selection rule: the smallest N whose out-of-fold patient-level AUROC (5-fold cross-validation of the
              training patients, grouped by patient) is at least the all-genes AUROC minus one bootstrap SD of that
              AUROC. The chosen N is then evaluated once on the test patients and the three external studies (AUROC
              pooled over within-study pairs), with stratified bootstrap 95% CIs (2,000 resamples) and paired bootstrap
              CIs of the difference from all genes; SMM patients: Spearman's rho with the IMWG 2/20/20 risk group.

Inputs:       data/signature/hotspot/consensus-modules_genes.csv: signature genes (index), context = signature, Z =
              Hotspot Z-score.
              data/signature/hotspot/mm-BM-PB-4980-DEs/mm-BM-PB-4980-DEs_hotspot_input.h5ad: layers/counts (CSR); obs
              gex_id, barcode, redcap_id, sample_type, total_counts (whole-transcriptome UMIs).
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: gex_id, barcode, annotation (tumor/normal), disease_status.
              data/classifier/datasets/sc-train-cohort-smoothed-knn30-expr_134samples.pkl, sc-test-...47samples.pkl:
              redcap_id, disease_status (training/test split).
              data/h5/external/Ledergor2018-Boiarsky2022-Dang2023_log1p_count.h5ad: X (CSR, log-normalized), obs source,
              sample_id, disease_status, annotation.
              results/classifiers/external_mst_units.tsv (22_external_mst_scores.py): external units of Fig. 5C.

Outputs:      results/classifiers/compact_model_genes.tsv: signature, gene, Z, rank, included at N = 5/10/20.
              results/classifiers/compact_model_selection.tsv: per N, unique genes, out-of-fold AUROC with CI, threshold.
              results/classifiers/compact_model_evaluation.tsv: chosen N vs all genes on test, external and SMM patients.
              results/classifiers/compact_model_log.txt: run log.

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
FH = open(os.path.join(OUT, "compact_model_log.txt"), "w")
SMM = ["LRSMM", "IRSMM", "HRSMM"]
NS = [5, 10, 20]
# tuned parameters of the MGUS-MM tier (as in 24_nfkb_index_model.py and 25_stage_free_signature_test.py)
DS_PARAMS = dict(n_estimators=200, max_features="sqrt", max_depth=25, min_samples_split=20, min_samples_leaf=10,
                 bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=RS)
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


# ---- 1. gene sets: top N genes per signature by Hotspot Z-score -----------------------------------------------------------
sig = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
sig = pd.DataFrame({"gene": sig.index.astype(str), "signature": sig["context"].astype(str).values, "Z": sig["Z"].values})
sig["rank"] = sig.groupby("signature")["Z"].rank(ascending=False, method="first").astype(int)
for n in NS:
    sig["top%d" % n] = sig["rank"] <= n
sig.sort_values(["signature", "rank"]).to_csv(os.path.join(OUT, "compact_model_genes.tsv"), sep="\t", index=False)
SIGS = sorted(sig["signature"].unique())
ARMS = {"all genes": sig}
ARMS.update({"top %d" % n: sig[sig["top%d" % n]] for n in NS})
for arm, d in ARMS.items():
    log("%s: %d signature genes, %d unique" % (arm, len(d), d["gene"].nunique()))


def activities(X, genes, arm):
    """mean log-normalized expression of each signature's genes (zeros included) per cell"""
    gi = {g: i for i, g in enumerate(genes)}
    W = np.zeros((len(genes), len(SIGS)))
    for j, s in enumerate(SIGS):
        idx = [gi[g] for g in ARMS[arm].loc[ARMS[arm]["signature"] == s, "gene"] if g in gi]
        W[idx, j] = 1.0 / len(idx)
    return np.asarray(X @ W)


# ---- 2. cohort cells: malignant BM PCs of the classifier units (as in 25_stage_free_signature_test.py) ---------------------
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
for arm in ARMS:
    g = set(ARMS[arm]["gene"])
    log("%s: %d genes measured in the external data of %d" % (arm, len(g & set(egenes)), len(g)))


# ---- 4. models and evaluation ---------------------------------------------------------------------------------------------
def z(a):
    return (a - a.mean(0)) / a.std(0)


def unit_scores(df, p, keys):
    return df.assign(p=p).groupby(keys)["p"].mean().reset_index()


def pooled(s, y, strata, i):
    """AUROC (MM vs MGUS) pooled over within-stratum pairs (as in 22_external_mst_scores.py): Mann-Whitney U summed over strata"""
    num = den = 0.0
    for st in np.unique(strata[i]):
        j = i[strata[i] == st]
        a, b = s[j][y[j] == 0], s[j][y[j] == 1]
        if len(a) and len(b):
            num += roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]) * len(a) * len(b)
            den += len(a) * len(b)
    return num / den


def boot_auroc(scores, y, strata=None):
    """AUROC of each score column; stratified bootstrap (same resamples for all columns)"""
    y = np.asarray(y)
    strata = np.zeros(len(y), int) if strata is None else np.asarray(strata)
    grp = [np.where((strata == s) & (y == k))[0] for s in np.unique(strata) for k in (0, 1)]
    allidx = np.arange(len(y))
    est = np.array([pooled(s, y, strata, allidx) for s in scores])
    boot = []
    for _ in range(NBOOT):
        i = np.concatenate([rng.choice(g, len(g)) for g in grp if len(g)])
        boot.append([pooled(s, y, strata, i) for s in scores])
    return est, np.array(boot)


tr = (cells["split"] == "train").values
y = (cells["disease_status"] == "MM").values.astype(int)
groups = cells["redcap_id"].values
Xc = {arm: z(activities(C, list(hgenes), arm)) for arm in ARMS}
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Xc["all genes"][tr], y[tr], groups[tr]))
oof = {}
for arm in ARMS:
    p = np.zeros(tr.sum())
    for a, b in folds:
        m = RandomForestClassifier(**DS_PARAMS).fit(Xc[arm][tr][a], y[tr][a])
        p[b] = m.predict_proba(Xc[arm][tr][b])[:, 1]
    oof[arm] = unit_scores(cells[tr], p, ["redcap_id", "disease_status"])

# selection rule on the out-of-fold AUROCs
u = oof["all genes"][["redcap_id", "disease_status"]]
yy = (u["disease_status"] == "MM").astype(int).values
est, boot = boot_auroc([oof[a]["p"].values for a in ARMS], yy)
sd_ref = boot[:, 0].std()
threshold = est[0] - sd_ref
chosen = next((a for a in ["top %d" % n for n in NS] if est[list(ARMS).index(a)] >= threshold), "all genes")
sel_rows = []
for k, arm in enumerate(ARMS):
    sel_rows.append(dict(genes_per_signature=arm, unique_genes=ARMS[arm]["gene"].nunique(), n_MGUS=int((yy == 0).sum()),
                         n_MM=int(yy.sum()), oof_AUROC=est[k], lo=np.percentile(boot[:, k], 2.5),
                         hi=np.percentile(boot[:, k], 97.5), reference_minus_1SD=threshold, chosen=arm == chosen))
SEL = pd.DataFrame(sel_rows)
SEL.to_csv(os.path.join(OUT, "compact_model_selection.tsv"), sep="\t", index=False)
log("\nselection (out-of-fold, training patients; reference all genes %.3f, bootstrap SD %.3f, threshold %.3f):\n%s"
    % (est[0], sd_ref, threshold, SEL.round(3).to_string(index=False)))

# single evaluation of the chosen panel and the all-genes reference
Xe = {}
for arm in ("all genes", chosen):
    a = activities(EX, list(egenes), arm)
    Xe[arm] = np.vstack([z(a[(esel["study"] == s).values]) for s in STUDY.values()])
esel = pd.concat([esel[esel["study"] == s] for s in STUDY.values()])        # same order as the stacked features
te, sm = (cells["split"] == "test").values, (cells["split"] == "SMM").values
emask = esel["disease_status"].isin(["MGUS", "MM"]).values
res = {}
for arm in ("all genes", chosen):
    m = RandomForestClassifier(**DS_PARAMS).fit(Xc[arm][tr], y[tr])
    res[arm] = dict(test=unit_scores(cells[te], m.predict_proba(Xc[arm][te])[:, 1], ["redcap_id", "disease_status"]),
                    ext=unit_scores(esel[emask], m.predict_proba(Xe[arm][emask])[:, 1], ["study", "sample_id", "disease_status"]),
                    smm=unit_scores(cells[sm], m.predict_proba(Xc[arm][sm])[:, 1], ["redcap_id", "disease_status"]))
rows = []
for key, label, strata in (("test", "test patients", None), ("ext", "external studies, pooled", "study")):
    k = ["study", "sample_id", "disease_status"] if key == "ext" else ["redcap_id", "disease_status"]
    uu = res["all genes"][key].merge(res[chosen][key], on=k, suffixes=("_0", "_1"))
    y2 = (uu["disease_status"] == "MM").astype(int).values
    e2, b2 = boot_auroc([uu["p_0"].values, uu["p_1"].values], y2, None if strata is None else uu[strata].values)
    d = b2[:, 1] - b2[:, 0]
    rows.append(dict(evaluation=label, n_MGUS=int((y2 == 0).sum()), n_MM=int(y2.sum()), all_genes=e2[0],
                     all_lo=np.percentile(b2[:, 0], 2.5), all_hi=np.percentile(b2[:, 0], 97.5), compact=e2[1],
                     compact_lo=np.percentile(b2[:, 1], 2.5), compact_hi=np.percentile(b2[:, 1], 97.5),
                     difference=e2[1] - e2[0], diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5)))
uu = res["all genes"]["smm"].merge(res[chosen]["smm"], on=["redcap_id", "disease_status"], suffixes=("_0", "_1"))
code = uu["disease_status"].map({g: i for i, g in enumerate(SMM)}).values
r0, p0 = stats.spearmanr(code, uu["p_0"])
r1, p1 = stats.spearmanr(code, uu["p_1"])
rows.append(dict(evaluation="SMM patients (rho with 2/20/20 risk group; lo/hi columns hold P)", n_MGUS=np.nan, n_MM=len(uu),
                 all_genes=r0, all_lo=p0, all_hi=np.nan, compact=r1, compact_lo=p1, compact_hi=np.nan,
                 difference=r1 - r0, diff_lo=np.nan, diff_hi=np.nan))
R = pd.DataFrame(rows).assign(compact_panel=chosen, compact_unique_genes=ARMS[chosen]["gene"].nunique())
R.to_csv(os.path.join(OUT, "compact_model_evaluation.tsv"), sep="\t", index=False)
log("\nchosen: %s (%d unique genes)\n%s" % (chosen, ARMS[chosen]["gene"].nunique(), R.round(3).T.to_string()))
log("done")
FH.close()
