#!/usr/bin/env python3
"""Robustness of the scRNA-seq cytogenetic calls: population sizes, partner-gene detection, accuracy with few malignant
PCs, calls on healthy-donor PCs and HRD calls against FISH and WGS (Methods, Classification of cytogenetic alterations).

Purpose:      Calls are made per malignant population (one per patient, two in the four patients with two malignant
              populations), never per cell: the translocation classifier takes the median Zhan subtype scores and the mean
              partner-gene expression over all malignant PCs of the population (Methods). The classifier is the
              re-implementation of 15_two_clone_tumors_and_classifier_cv.py (one logistic regression per translocation
              class on standardized features, scikit-learn defaults, max_iter 5000; class with the highest probability if
              > 0.5, otherwise no translocation), trained on the 122 patients with a FISH or WGS call. Random draws:
              default_rng(42).
              1. Size of the malignant populations that received a call.
              2. Detection of the translocation partner genes in single cells: percentage of malignant PCs with at least
                 one UMI of the partner gene, per tumor, against healthy-donor PCs.
              3. Minimum number of cells: leave-one-patient-out agreement with FISH/WGS when only n random malignant PCs
                 per patient are used (n = 10, 20, 30, 50, 100; 20 random sets each; the patients with >= 100 malignant
                 PCs).
              4. False positives in healthy donors (all 29): the classifier trained on the 122 patients applied to (a) all
                 PCs of each donor, (b) the PCs of each donor BM library, (c) 20 random sets of n PCs per donor
                 (n = 10-100); and, for comparison, to the normal PCs of the patients.
              5. HRD calls (Numbat, tumors without a translocation): agreement with FISH and WGS calls of HRD (Numbat uses
                 donor PCs as its normal reference, so it cannot be tested on donor cells).

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, gex_id, sample_type, annotation, tumor_id,
              norm_CD_1, norm_CD_2, norm_MS, norm_MF, norm_HP, CCND1, CCND2, NSD2, FGFR3, MAF, MAFB.
              data/paper/cytogenetics-status.csv: tumor_id (index, <Pt-###>:tumor:<k>), patient_id, cytogenetics, FISH,
              WGS, scRNA.
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id, disease_status.

Outputs:      results/genomics/cytogenetic_population_sizes.tsv: malignant PCs per called population, by reference.
              results/genomics/partner_gene_detection.tsv: partner-gene detection in malignant PCs and healthy-donor PCs.
              results/genomics/cytogenetic_cells_per_call.tsv: leave-one-patient-out agreement by number of malignant PCs.
              results/genomics/cytogenetic_donor_calls.tsv: classifier calls on healthy-donor and patients' normal PCs.
              results/genomics/hrd_concordance.tsv: scRNA-seq HRD calls against FISH and WGS.
              results/genomics/cytogenetic_call_robustness_log.txt: tables printed by the script (no internal IDs).

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
FH = open(OUT / "cytogenetic_call_robustness_log.txt", "w")
ZHAN = ["norm_CD_1", "norm_CD_2", "norm_MS", "norm_MF", "norm_HP"]
GENES = ["CCND1", "CCND2", "NSD2", "FGFR3", "MAF", "MAFB"]
CLASSES = ["t(11;14)", "t(4;14)", "t(14;16)", "t(14;20)", "t(6;14)", "t(8;14)"]
PARTNER = {"t(11;14)": "CCND1", "t(4;14)": "NSD2", "t(14;16)": "MAF", "t(14;20)": "MAFB", "t(12;14)": "CCND2"}
SIZES = [10, 20, 30, 50, 100]
N_REP = 20
RNG = np.random.default_rng(42)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def features(cells):
    """classifier input for one population: median Zhan scores, mean partner-gene expression."""
    return np.r_[cells[ZHAN].median().values, cells[GENES].mean().values]


def subsets(cells, n, rep=N_REP):
    return np.vstack([features(cells.iloc[RNG.choice(len(cells), n, replace=False)]) for _ in range(rep)])


# ---- cells (as in 15_two_clone_tumors_and_classifier_cv.py) ---------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
red2pt = meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"]
donors = set(meta.loc[meta["disease_status"] == "Donor", "patient_id"])
cols = ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id"] + ZHAN + GENES
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[cols]
for c in ["redcap_id", "gex_id", "sample_type", "annotation", "tumor_id"]:
    ann[c] = ann[c].astype(str)
ann["patient_id"] = ann["redcap_id"].map(red2pt)
assert ann["patient_id"].notna().all()
ann["tumor_id"] = np.where(ann["annotation"] == "tumor", ann["patient_id"] + ":tumor:" + ann["tumor_id"].str.rsplit(":", n=1).str[-1], "nan")
don = ann[ann["patient_id"].isin(donors)]
log("healthy donors: %d individuals, %d PCs (annotation of donor PCs: %s)"
    % (don["patient_id"].nunique(), len(don), don["annotation"].value_counts().to_dict()))
assert don["patient_id"].nunique() == 29 and (don["annotation"] == "normal").all()

cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"), index_col=0)
tum = ann[ann["annotation"] == "tumor"]
assert tum["tumor_id"].isin(cyto.index).all()
pops = {tid: q for tid, q in tum.groupby("tumor_id")}

# ---- 1. size of the populations that received a call ----------------------------------------------
valid = CLASSES + ["HRD"]
cyto["n_cells"] = [len(pops[t]) for t in cyto.index]
cyto["reference"] = np.where(cyto["FISH"].isin(valid) | cyto["WGS"].isin(valid), "FISH or WGS", "scRNA-seq only")
S = cyto.groupby("reference")["n_cells"].describe(percentiles=[0.1, 0.5]).reset_index()
S = pd.concat([S, cyto["n_cells"].describe(percentiles=[0.1, 0.5]).to_frame().T.assign(reference="all")])
S.to_csv(OUT / "cytogenetic_population_sizes.tsv", sep="\t", index=False)
log("\n1. malignant PCs per called population (%d populations, %d patients)" % (len(cyto), cyto["patient_id"].nunique()))
log(S.round(1).to_string(index=False))
log("populations with < 100 / < 50 malignant PCs: %d / %d" % ((cyto["n_cells"] < 100).sum(), (cyto["n_cells"] < 50).sum()))
log("calls of populations with < 100 malignant PCs: %s (reference: %s)" % (
    cyto.loc[cyto["n_cells"] < 100, "scRNA"].value_counts().to_dict(), cyto.loc[cyto["n_cells"] < 100, "reference"].value_counts().to_dict()))

# ---- 2. partner-gene detection in single cells -------------------------------------------------------
rows = []
for t, g in PARTNER.items():
    with_t = [tid for tid in cyto.index if cyto.loc[tid, "cytogenetics"] == t]
    other = [tid for tid in cyto.index if cyto.loc[tid, "cytogenetics"] != t]
    det = pd.Series({tid: 100 * (pops[tid][g] > 0).mean() for tid in cyto.index})
    rows.append(dict(translocation=t, gene=g, n_tumors=len(with_t),
                     pct_cells_detected_median=det[with_t].median(), pct_cells_detected_min=det[with_t].min(),
                     other_tumors_pct_median=det[other].median(),
                     donor_pct_cells=100 * (don[g] > 0).mean(),
                     donor_pct_cells_max_individual=don.groupby("patient_id")[g].apply(lambda v: 100 * (v > 0).mean()).max()))
    low = [tid for tid in with_t if det[tid] < 25]
    if low:
        log("  %s tumors with %s detected in < 25%% of malignant PCs: %d (FISH %s; WGS %s)" % (
            t, g, len(low), cyto.loc[low, "FISH"].tolist(), cyto.loc[low, "WGS"].tolist()))
D = pd.DataFrame(rows)
D.to_csv(OUT / "partner_gene_detection.tsv", sep="\t", index=False)
log("\n2. partner gene detected (>= 1 UMI) in malignant PCs of tumors with the translocation vs healthy-donor PCs")
log(D.round(2).to_string(index=False))

# ---- 3. leave-one-patient-out accuracy with n cells per patient -----------------------------------------
main = cyto[cyto.index.str.endswith("tumor:0")].copy()
main["truth"] = np.where(main["FISH"].isin(valid), main["FISH"], np.where(main["WGS"].isin(valid), main["WGS"], None))
main = main[main["truth"].notna()]
main["truth"] = main["truth"].where(main["truth"].isin(CLASSES), "none")
X = np.vstack([features(pops[t]) for t in main.index])
y = main["truth"].values
fitted = [c for c in CLASSES if (y == c).sum() >= 2]
log("\n3. patients with a FISH or WGS call: %d; truth classes %s" % (len(y), pd.Series(y).value_counts().to_dict()))


def fit(Xtr, ytr):
    return {c: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, (ytr == c).astype(int))
            for c in fitted if (ytr == c).sum() >= 1}


def proba(models, Xte):
    return np.column_stack([models[c].predict_proba(Xte)[:, 1] for c in models])


def predict(models, Xte):
    P = proba(models, Xte)
    return np.where(P.max(axis=1) > 0.5, np.array(list(models))[P.argmax(axis=1)], "none")


big = [i for i, t in enumerate(main.index) if len(pops[t]) >= max(SIZES)]
res = {n: [] for n in SIZES + ["all"]}
for i in range(len(y)):
    tr = np.arange(len(y)) != i
    m = fit(X[tr], y[tr])
    res["all"].append((i, predict(m, X[i:i + 1])))
    if i in big:
        for n in SIZES:
            res[n].append((i, predict(m, subsets(pops[main.index[i]], n))))
rows = []
for n in SIZES + ["all"]:
    ii = [i for i, _ in res[n]]
    agree = np.mean([np.mean(p == y[i]) for i, p in res[n]])
    trans = [(i, p) for i, p in res[n] if y[i] != "none"]
    neg = [(i, p) for i, p in res[n] if y[i] == "none"]
    rows.append(dict(cells_per_call=n, n_patients=len(ii),
                     pct_agreement=100 * agree,
                     pct_translocations_detected=100 * np.mean([np.mean(p == y[i]) for i, p in trans]),
                     pct_translocations_missed=100 * np.mean([np.mean(p == "none") for i, p in trans]),
                     pct_wrong_translocation=100 * np.mean([np.mean((p != "none") & (p != y[i])) for i, p in trans]),
                     pct_false_translocation_in_none=100 * np.mean([np.mean(p != "none") for i, p in neg])))
C = pd.DataFrame(rows)
C.to_csv(OUT / "cytogenetic_cells_per_call.tsv", sep="\t", index=False)
log("leave-one-patient-out agreement with FISH/WGS by number of malignant PCs per call (20 random sets per patient;"
    " 'all' = all malignant PCs, all %d patients)" % len(y))
log(C.round(1).to_string(index=False))

# ---- 4. false positives in healthy donors -------------------------------------------------------------
final = fit(X, y)
rows = []


def add(source, unit, n, P, n_ind):
    calls = np.where(P.max(axis=1) > 0.5, np.array(list(final))[P.argmax(axis=1)], "none")
    rows.append(dict(source=source, unit=unit, cells_per_call=n, n_individuals=n_ind, n_calls=len(calls),
                     n_called_translocation=int((calls != "none").sum()),
                     pct_called_translocation=100 * (calls != "none").mean(),
                     max_class_probability=P.max(), median_max_class_probability=np.median(P.max(axis=1))))


Pd = proba(final, np.vstack([features(q) for _, q in don.groupby("patient_id")]))
add("healthy donors", "all PCs of each donor", "all", Pd, don["patient_id"].nunique())
bm = don[don["sample_type"] == "BM"]
Pb = proba(final, np.vstack([features(q) for _, q in bm.groupby("gex_id")]))
add("healthy donors", "PCs of each BM library", "all", Pb, bm["patient_id"].nunique())
nrm = ann[(~ann["patient_id"].isin(donors)) & (ann["annotation"] == "normal")]
for lab, src in [("healthy donors", don), ("patients' normal PCs", nrm)]:
    g = {p: q for p, q in src.groupby("patient_id")}
    for n in SIZES:
        ok = [p for p in g if len(g[p]) >= n]
        add(lab, "random sets of PCs", n, proba(final, np.vstack([subsets(g[p], n) for p in ok])), len(ok))
N = pd.DataFrame(rows)
N.to_csv(OUT / "cytogenetic_donor_calls.tsv", sep="\t", index=False)
log("\n4. classifier calls on normal PCs (model trained on the 122 patients)")
log("donor PCs per individual: median %d, range %d-%d; donor BM libraries: %d"
    % (don.groupby("patient_id").size().median(), don.groupby("patient_id").size().min(), don.groupby("patient_id").size().max(),
       bm["gex_id"].nunique()))
log(N.round(3).to_string(index=False))
# for reference: the class probability of the called populations of the patients
Pp = proba(final, X)
log("max class probability in patients with a translocation (in-sample): median %.3f, min %.3f"
    % (np.median(Pp[y != "none"].max(axis=1)), Pp[y != "none"].max(axis=1).min()))

# ---- 5. HRD calls against FISH and WGS ------------------------------------------------------------------
m0 = cyto[cyto.index.str.endswith("tumor:0")].copy()
m0["scRNA"] = m0["scRNA"].replace({"no_translocation_detected": "none"})
rows = []
for assay in ["FISH", "WGS"]:
    h = m0[m0[assay] == "HRD"]
    rows.append(dict(assay=assay, reference="HRD", n=len(h), scRNA_HRD=int((h["scRNA"] == "HRD").sum()),
                     scRNA_calls=h["scRNA"].value_counts().to_dict()))
    o = m0[m0[assay].isin(["Other"])]
    if len(o):
        rows.append(dict(assay=assay, reference="Other (no translocation or HRD)", n=len(o),
                         scRNA_HRD=int((o["scRNA"] == "HRD").sum()), scRNA_calls=o["scRNA"].value_counts().to_dict()))
H = pd.DataFrame(rows)
H.to_csv(OUT / "hrd_concordance.tsv", sep="\t", index=False)
log("\n5. scRNA-seq HRD calls (Numbat) in the main population of patients with an HRD or 'Other' call by FISH or WGS")
log(H.to_string(index=False))
log("\ndone")
