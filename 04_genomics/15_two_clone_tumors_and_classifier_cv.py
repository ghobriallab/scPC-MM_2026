#!/usr/bin/env python3
"""Patients with two malignant PC populations (BCR, constant genes, partner genes) and leave-one-patient-out performance
of the translocation classifier (Results; Methods, Classification of cytogenetic alterations).

Purpose:      A. The four patients with two cytogenetically distinct malignant populations (tumor populations :0 and :1 of
              the cytogenetic table). Malignant PCs are defined by expanded BCR clonotypes (Methods), so each population
              is compared on its BCR, measured independently of the expression-based cytogenetic call: modal heavy-chain
              CDR3 (amino acid) and the share of cells carrying it or the other population's; whether the modal CDR3
              occurs in any other individual of the cohort (cross-sample contamination); heavy- and light-chain constant
              genes; libraries and tissues; median UMIs; the clinical immunofixation entry; and the expression of the
              translocation partner genes in each population, the patient's normal PCs and all healthy-donor PCs.
              B. Performance of the translocation classifier, re-implemented as described in the Methods: one logistic
              regression per translocation class (standardized features, scikit-learn defaults, max_iter 5000) on the
              per-population median Zhan subtype scores (norm_CD_1, norm_CD_2, norm_MS, norm_MF, norm_HP) and the mean
              expression of CCND1, CCND2, NSD2, FGFR3, MAF and MAFB; the class with the highest probability if > 0.5,
              otherwise no translocation. Reference: FISH, else WGS, for the main population of the patients with a
              translocation or HRD call. Leave-one-patient-out cross-validation with all malignant PCs and with 20 random
              subsets of 100 malignant PCs per patient (sensitivity and specificity per class); the model fitted on all
              reference patients is applied to 20 random sets of 100 normal PCs per individual (healthy donors and
              patients' normal PCs) and to each population of the four patients. Random draws: default_rng(42).

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, gex_id, sample_type, annotation, tumor_id,
              cytogenetics, heavy_cdr3_aa, heavy_c_gene, light_c_gene, total_counts, norm_CD_1, norm_CD_2, norm_MS,
              norm_MF, norm_HP, CCND1, CCND2, NSD2, FGFR3, MAF, MAFB.
              data/paper/cytogenetics-status.csv: tumor_id (index, <Pt-###>:tumor:<k>), patient_id, cytogenetics, FISH,
              WGS, regression_pred.
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id, disease_status (Donor), IFX
              (clinical serum immunofixation).

Outputs:      results/genomics/two_clone_tumors.tsv: one row per population of the four patients (no sequences or
              internal IDs).
              results/genomics/two_clone_partner_genes.tsv: partner-gene expression by population, normal PCs and donor
              PCs.
              results/genomics/translocation_classifier_cv.tsv: sensitivity and specificity per class (all cells; 100
              cells).
              results/genomics/translocation_classifier_normal_pc_calls.tsv: classifier calls on sets of 100 normal PCs.
              results/genomics/two_clone_tumors_and_classifier_cv_log.txt: tables and agreement counts printed by the
              script.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scikit-learn); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = result_dir("genomics")
FH = open(OUT / "two_clone_tumors_and_classifier_cv_log.txt", "w")
ZHAN = ["norm_CD_1", "norm_CD_2", "norm_MS", "norm_MF", "norm_HP"]
GENES = ["CCND1", "CCND2", "NSD2", "FGFR3", "MAF", "MAFB"]
CLASSES = ["t(11;14)", "t(4;14)", "t(14;16)", "t(14;20)", "t(6;14)", "t(8;14)"]
PARTNER = {"t(11;14)": "CCND1", "t(14;16)": "MAF", "t(14;20)": "MAFB", "t(4;14)": "NSD2"}
N_SMALL, N_REP = 100, 20
RNG = np.random.default_rng(42)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def features(cells):
    """classifier input for one population: median Zhan scores, mean partner-gene expression."""
    return np.r_[cells[ZHAN].median().values, cells[GENES].mean().values]


def subsets(cells, n=N_SMALL, rep=N_REP):
    return [features(cells.iloc[RNG.choice(len(cells), n, replace=False)]) for _ in range(rep)]


# ---- cells --------------------------------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
red2pt = meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"]
donors = set(meta.loc[meta["disease_status"] == "Donor", "patient_id"])
cols = ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id", "cytogenetics", "heavy_cdr3_aa",
        "heavy_c_gene", "light_c_gene", "total_counts"] + ZHAN + GENES
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[cols]
for c in ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id", "cytogenetics", "heavy_cdr3_aa",
          "heavy_c_gene", "light_c_gene"]:
    ann[c] = ann[c].astype(str)
ann["patient_id"] = ann["redcap_id"].map(red2pt)
assert ann["patient_id"].notna().all()
# population IDs: the cell table names populations <record ID>:tumor:<k>, the cytogenetic table <Pt-###>:tumor:<k>
ann["tumor_id"] = np.where(ann["annotation"] == "tumor", ann["patient_id"] + ":tumor:" + ann["tumor_id"].str.rsplit(":", n=1).str[-1], "nan")
log("PCs: %d | individuals: %d | mean of centered Zhan scores over all PCs: %s"
    % (len(ann), ann["patient_id"].nunique(), ann[ZHAN].mean().round(3).to_dict()))

cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"), index_col=0)
tum = ann[ann["annotation"] == "tumor"]
log("malignant PCs whose population is in the cytogenetic table: %.4f" % tum["tumor_id"].isin(cyto.index).mean())
assert tum["tumor_id"].isin(cyto.index).all()

# ---- Part A: the four two-population patients ---------------------------------------------------
two = cyto.groupby("patient_id").size()
two = sorted(two[two > 1].index)
log("\nPART A: patients with two malignant populations:", two)
rows, expr = [], []
for p in two:
    d = ann[ann["patient_id"] == p]
    t = d[d["annotation"] == "tumor"]
    pops = sorted(t["tumor_id"].unique(), key=lambda x: x.split(":")[-1])
    modal = {}
    for tid in pops:
        q = t[t["tumor_id"] == tid]
        h = q.loc[q["heavy_cdr3_aa"] != "nan", "heavy_cdr3_aa"]
        modal[tid] = h.value_counts().index[0]
    for tid in pops:
        q = t[t["tumor_id"] == tid]
        other = [o for o in pops if o != tid][0]
        h = q.loc[q["heavy_cdr3_aa"] != "nan", "heavy_cdr3_aa"]
        elsewhere = ann.loc[(ann["heavy_cdr3_aa"] == modal[tid]) & (ann["patient_id"] != p), "patient_id"].nunique()
        rows.append(dict(
            patient=p, population=tid.split(":")[-1], cytogenetics=cyto.loc[tid, "cytogenetics"], n_cells=len(q),
            pct_of_malignant=100 * len(q) / len(t),
            pct_cells_modal_heavy_cdr3=100 * (h == modal[tid]).mean(),
            pct_cells_with_other_population_cdr3=100 * (h == modal[other]).mean(),
            other_individuals_with_modal_cdr3=elsewhere,
            heavy_constant=q["heavy_c_gene"].mode().iloc[0], light_constant=q["light_c_gene"].mode().iloc[0],
            pct_light_mode=100 * (q["light_c_gene"] == q["light_c_gene"].mode().iloc[0]).mean(),
            n_libraries=q["gex_id"].nunique(), tissues="/".join(sorted(q["sample_type"].unique())),
            median_umi=q["total_counts"].median()))
    rows.append(dict(patient=p, population="normal PCs", n_cells=int((d["annotation"] == "normal").sum()),
                     median_umi=d.loc[d["annotation"] == "normal", "total_counts"].median()))
    rows.append(dict(patient=p, population="clinical immunofixation",
                     clinical_ifx=" | ".join(sorted(meta.loc[meta["patient_id"] == p, "IFX"].dropna().astype(str).unique()))))
    # partner genes: each population, the patient's normal PCs and all healthy-donor PCs
    groups = {("population " + tid.split(":")[-1] + " " + cyto.loc[tid, "cytogenetics"]): t[t["tumor_id"] == tid] for tid in pops}
    groups["normal PCs (patient)"] = d[d["annotation"] == "normal"]
    for g in sorted(set(PARTNER[c] for c in cyto.loc[pops, "cytogenetics"] if c in PARTNER)):
        for lab, q in list(groups.items()) + [("healthy-donor PCs", None)]:
            v = ann.loc[ann["patient_id"].isin(donors), g] if q is None else q[g]
            expr.append(dict(patient=p, gene=g, group=lab, n_cells=len(v), mean_expr=v.mean(), pct_expressing=100 * (v > 0).mean()))
A = pd.DataFrame(rows)
A.to_csv(OUT / "two_clone_tumors.tsv", sep="\t", index=False)
E = pd.DataFrame(expr)
E.to_csv(OUT / "two_clone_partner_genes.tsv", sep="\t", index=False)
log(A.round(2).to_string())
log(E.round(3).to_string())

# ---- Part B: classifier performance --------------------------------------------------------------
# truth: FISH, else WGS, for the main population of each patient (FISH and WGS agree where both exist); the
# informative calls are a translocation class or HRD, as in the 122 patients used to train the classifier
main = cyto[cyto.index.str.endswith("tumor:0")].copy()
valid = CLASSES + ["HRD"]
main["truth"] = np.where(main["FISH"].isin(valid), main["FISH"], np.where(main["WGS"].isin(valid), main["WGS"], None))
both = main["FISH"].isin(valid) & main["WGS"].isin(valid)
log("\nPART B: patients with FISH or WGS call: %d (FISH %d, WGS %d, both %d; FISH/WGS disagree: %d)"
    % (main["truth"].notna().sum(), main["FISH"].isin(valid).sum(), main["WGS"].isin(valid).sum(), both.sum(),
       (both & (main["FISH"] != main["WGS"])).sum()))
main = main[main["truth"].notna()]
main["truth"] = main["truth"].where(main["truth"].isin(CLASSES), "none")
log("truth classes:", main["truth"].value_counts().to_dict())
pops = {tid: tum[tum["tumor_id"] == tid] for tid in main.index}
X = np.vstack([features(pops[t]) for t in main.index])
y = main["truth"].values
fitted = [c for c in CLASSES if (y == c).sum() >= 2]


def fit(Xtr, ytr):
    return {c: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, (ytr == c).astype(int))
            for c in fitted if (ytr == c).sum() >= 1}


def predict(models, Xte):
    P = np.column_stack([models[c].predict_proba(Xte)[:, 1] for c in models])
    best = P.argmax(axis=1)
    return np.where(P.max(axis=1) > 0.5, np.array(list(models))[best], "none")


pred_all, pred_small = [], []
for i in range(len(y)):
    tr = np.arange(len(y)) != i
    m = fit(X[tr], y[tr])
    pred_all.append(predict(m, X[i:i + 1])[0])
    pred_small.append(predict(m, np.vstack(subsets(pops[main.index[i]]))) if len(pops[main.index[i]]) >= N_SMALL else None)
pred_all = np.array(pred_all)
orig = main["regression_pred"].replace({"no_translocation_detected": "none"}).values
log("in-sample agreement of the reported classifier calls (regression_pred) with FISH/WGS: %d/%d" % ((orig == y).sum(), len(y)))
log("leave-one-out agreement (all cells): %d/%d" % ((pred_all == y).sum(), len(y)))
ok_small = [j for j, s in enumerate(pred_small) if s is not None]
acc_small = np.mean([np.mean(pred_small[j] == y[j]) for j in ok_small])
log("leave-one-out agreement with %d-cell subsets: %.3f (%d patients with >= %d cells, %d subsets each)"
    % (N_SMALL, acc_small, len(ok_small), N_SMALL, N_REP))
rows = []
for c in CLASSES + ["none"]:
    pos, neg = y == c, y != c
    r = dict(cls=c, n_positive=int(pos.sum()), n_negative=int(neg.sum()))
    if pos.sum():
        r["sensitivity_all_cells"] = np.mean(pred_all[pos] == c)
    r["specificity_all_cells"] = np.mean(pred_all[neg] != c)
    sp = [j for j in ok_small if y[j] == c]
    sn = [j for j in ok_small if y[j] != c]
    if sp:
        r["sensitivity_100_cells"] = np.mean([np.mean(pred_small[j] == c) for j in sp])
    r["specificity_100_cells"] = np.mean([np.mean(pred_small[j] != c) for j in sn])
    rows.append(r)
C = pd.DataFrame(rows)
C.to_csv(OUT / "translocation_classifier_cv.tsv", sep="\t", index=False)
log(C.round(3).to_string())
tp = sum(((pred_all == c) & (y == c)).sum() for c in CLASSES)
log("any translocation: sensitivity %.3f (%d/%d), specificity %.3f (%d/%d)"
    % (tp / (y != "none").sum(), tp, (y != "none").sum(), np.mean(pred_all[y == "none"] == "none"),
       (pred_all[y == "none"] == "none").sum(), (y == "none").sum()))

# false positives on normal PCs (model trained on all characterized patients)
final = fit(X, y)
rows = []
for lab, src in [("healthy donors", ann[(ann["patient_id"].isin(donors)) & (ann["annotation"] == "normal")]),
                 ("patients' normal PCs", ann[(~ann["patient_id"].isin(donors)) & (ann["annotation"] == "normal")])]:
    calls = []
    for p, q in src.groupby("patient_id"):
        if len(q) >= N_SMALL:
            calls += list(predict(final, np.vstack(subsets(q))))
    calls = pd.Series(calls)
    rows.append(dict(source=lab, n_individuals=int((src.groupby("patient_id").size() >= N_SMALL).sum()), n_sets=len(calls),
                     pct_called_translocation=100 * (calls != "none").mean(), calls=calls.value_counts().to_dict()))
N = pd.DataFrame(rows)
N.to_csv(OUT / "translocation_classifier_normal_pc_calls.tsv", sep="\t", index=False)
log(N.to_string())

# each population of the four patients with the model fitted on all characterized patients (all their cells and
# 100-cell subsets)
for p in two:
    for tid in sorted(cyto.index[cyto["patient_id"] == p]):
        q = tum[tum["tumor_id"] == tid]
        sm = predict(final, np.vstack(subsets(q))) if len(q) >= N_SMALL else []
        log("%s population %s: reported call %s | model fitted here %s | 100-cell subsets: %s" % (
            p, tid.split(":")[-1], cyto.loc[tid, "cytogenetics"], predict(final, features(q)[None, :])[0],
            pd.Series(sm).value_counts().to_dict()))
log("\ndone")
