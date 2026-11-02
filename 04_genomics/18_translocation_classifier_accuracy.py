#!/usr/bin/env python3
"""Accuracy of the translocation classifier (Results; Methods, Classification of cytogenetic alterations).

Purpose:      From the fits of 17_translocation_classifier.R:
              1. the cell table reproduces the input matrix of the classifier;
              2. leave-one-out accuracy, each tumor classified by the model fitted without it, under the most probable
                 class and under the rule applied to the cohort (top translocation class if its probability is > 0.5);
              3. accuracy with few malignant PCs: each tumor scored by the model fitted without it on 20 random sets of 10,
                 20, 30, 50 and 100 of its malignant PCs;
              4. the final model on all PCs of each healthy donor, on the PCs of each donor BM library and on 20 random sets
                 of 10-100 normal PCs of each donor and patient;
              5. agreement of the BM and PB calls of a patient, the four patients with two malignant clones, and the sizes
                 and FISH or WGS support of the calls;
              6. training tumors outside the cohort;
              7. normal PCs by tissue, whole populations and random sets;
              8. PB normal populations with each feature set to its BM normal median;
              9. leave-one-out models on the PB malignant PCs of the same patient;
              10. errors, with the single CCND3 tumor counted, and the reviewed calls;
              11. clonality of the BM normal populations that receive a call;
              12. the calls that rest on PB;
              13. the cells covered by the min-max scaling of the Zhan scores.
              Random draws: numpy default_rng(42).

Inputs:       data/cytogenetics/classifier_cells.csv: record_id, gex_id, sample_type, annotation, annotation3, clonotype, CD_1,
              CD_2, MS, MF, HP and the same with :Norm, MAF, MAFA, MAFB, ITGB7, CCND1, CCND2, CCND3, NSD2, FGFR3.
              results/genomics/translocation_classifier_input_matrix.csv, _loo_coefficients.csv, _loo_predictions.csv,
              _final_coefficients.csv, _pb_predictions.csv, _all_predictions.csv (17_translocation_classifier.R).
              data/metadata/tumor-cohort-metadata-final-240325.csv: patient_id, disease_status, gex_id.
              data/paper/cytogenetics-status.csv: tumor_id, patient_id, FISH, WGS, scRNA, regression_pred, cytogenetics.

Outputs:      results/genomics/translocation_classifier_*.tsv: tables of the sections above (patient labels and counts only).
              results/genomics/translocation_classifier_accuracy_log.txt: everything printed.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir

import numpy as np
import pandas as pd

OUT = result_dir("genomics")
FH = open(OUT / "translocation_classifier_accuracy_log.txt", "w")
ZHAN = ["CD_1", "CD_2", "MS", "MF", "HP"]
GENES = ["MAF", "MAFA", "MAFB", "ITGB7", "CCND1", "CCND2", "CCND3", "NSD2", "FGFR3"]
FEATS = ZHAN + GENES                                   # the coefficient order of the saved models
ORDER = ["t(11;14)", "t(4;14)", "t(14;16)", "t(14;20)", "t(6;14)", "t(8;14)"]
SIZES = [10, 20, 30, 50, 100]
N_REP = 20
RNG = np.random.default_rng(42)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def features(cells):
    """Classifier input of one population: median normalized Zhan scores, mean gene expression (input-matrix step)."""
    return np.r_[np.median(cells[0], axis=0), cells[1].mean(axis=0)]


def as_arrays(q):
    return q[[z + ":Norm" for z in ZHAN]].values.astype(float), q[GENES].values.astype(float)


def subset_features(arr, n, rep=N_REP):
    return np.vstack([features((arr[0][i], arr[1][i])) for i in (RNG.choice(len(arr[0]), n, replace=False) for _ in range(rep))])


def probabilities(coef, X):
    """Class probabilities of a multinomial model with HRD as reference, in log space: the leave-one-out models reach log-odds
    above 1,000, which overflow a plain exp()."""
    classes = list(coef.index)
    t = coef["Intercept"].values[None, :] + X @ coef[FEATS].values.T
    logden = np.logaddexp(0.0, np.logaddexp.reduce(t, axis=1, keepdims=True))
    return classes, np.exp(t - logden), np.exp(-logden)


def trxscore_as_published(coef, X):
    """The arithmetic of the classifier script (linear predictor capped at 709, plain exp()), to check it against the rule above."""
    classes = list(coef.index)
    t = np.minimum(coef["Intercept"].values[None, :] + X @ coef[FEATS].values.T, 709.0)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(t)
        p = e / (1 + e.sum(axis=1, keepdims=True))
    top = np.nanargmax(p, axis=1)
    return np.where(p[np.arange(len(p)), top] > 0.5, np.array(classes)[top], "No Trx")


def rule_threshold(coef, X):
    """Rule applied to the cohort: top translocation class if its probability is > 0.5, otherwise no translocation."""
    classes, p, _ = probabilities(coef, X)
    top = p.argmax(axis=1)
    return np.where(p[np.arange(len(p)), top] > 0.5, np.array(classes)[top], "No Trx"), p.max(axis=1)


def rule_argmax(coef, X):
    """Rule of predict() on the fitted model: most probable class, HRD included."""
    classes, p, ref = probabilities(coef, X)
    top = p.argmax(axis=1)
    return np.where(p[np.arange(len(p)), top] > ref[:, 0], np.array(classes)[top], "No Trx")


# ---- inputs ---------------------------------------------------------------------------------------------------
USE = ["record_id", "gex_id", "sample_type", "annotation", "annotation3", "clonotype"] + ZHAN + [z + ":Norm" for z in ZHAN] + GENES
cells = pd.read_csv(data("cytogenetics", "classifier_cells.csv"), usecols=USE, low_memory=False)
for c in ["record_id", "gex_id", "sample_type", "annotation", "annotation3", "clonotype"]:
    cells[c] = cells[c].astype(str)
models = pd.read_csv(result("genomics", "translocation_classifier_loo_coefficients.csv"))
final = pd.read_csv(result("genomics", "translocation_classifier_final_coefficients.csv")).set_index("feature")
loo = pd.read_csv(result("genomics", "translocation_classifier_loo_predictions.csv"))
inp = pd.read_csv(result("genomics", "translocation_classifier_input_matrix.csv"))
overall = pd.read_csv(result("genomics", "translocation_classifier_all_predictions.csv"))
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), low_memory=False)
cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"))
# individuals are matched on the library ID: the record ID columns of the two tables are not always written the same way
lib2pt = meta.drop_duplicates("gex_id").set_index("gex_id")["patient_id"].astype(str).to_dict()
donors = set(meta.loc[meta["disease_status"] == "Donor", "patient_id"].astype(str))
cells["pt"] = cells["gex_id"].map(lib2pt)
pang2pt = cells.dropna(subset=["pt"]).drop_duplicates("record_id").set_index("record_id")["pt"].to_dict()
log("cell table: %d cells, %d in the cohort libraries (%d individuals)" % (len(cells), cells["pt"].notna().sum(), cells["pt"].nunique()))

# ---- 1. the cell table reproduces the classifier's input matrix --------------------------------------------------
tum = cells[cells["annotation"] == "Tumor"].copy()
tum["key"] = tum["record_id"] + ":" + tum["annotation3"] + ";" + tum["sample_type"]
pops = {k: as_arrays(q) for k, q in tum.groupby("key", sort=False)}
inp = inp.set_index("Sample")
common = [k for k in inp.index if k in pops]
new = np.vstack([features(pops[k]) for k in common])
old = inp.loc[common, FEATS].values.astype(float)
log("\n1. input matrix: %d of %d populations found in the cell table; largest absolute difference %.2e" % (len(common), len(inp), np.abs(new - old).max()))

# ---- 2. leave-one-out accuracy ------------------------------------------------------------------------------------
loo["model"] = ["m%d" % (i + 1) for i in range(len(loo))]
loo["pt"] = loo["PatientID"].map(pang2pt)
loo["ref"] = loo["FISHorWGS"].replace({"HRD": "No Trx"})
coef_by_model = {m: q.set_index("Outcome") for m, q in models.groupby("model")}
X = loo[FEATS].values.astype(float)
loo["argmax"] = [rule_argmax(coef_by_model[m], X[[i]])[0] for i, m in enumerate(loo["model"])]
loo["threshold"] = [rule_threshold(coef_by_model[m], X[[i]])[0][0] for i, m in enumerate(loo["model"])]
loo["saved"] = loo["predictions"].replace({"HRD": "No Trx"})
log("\n2. leave-one-out: saved calls reproduced from the saved coefficients: %d of %d" % ((loo["argmax"] == loo["saved"]).sum(), len(loo)))
assert (loo["argmax"] == loo["saved"]).all()
Xall = overall[FEATS].values.astype(float)
a, b = rule_threshold(final, Xall)[0], trxscore_as_published(final, Xall)
log("final model: saved calls reproduced by the rule used here %d of %d; by the original arithmetic %d of %d" % (
    (a == overall["model_agg_predictions"]).sum(), len(overall), (b == overall["model_agg_predictions"]).sum(), len(overall)))
assert (a == overall["model_agg_predictions"]).all()
rows = []
for name, sub in [("all training tumors", loo), ("cohort patients", loo[loo["pt"].notna()])]:
    s = sub[sub["FISHorWGS"] != "t(6;14)"]
    tr = s[s["ref"] != "No Trx"]
    neg = s[s["ref"] == "No Trx"]
    for rule in ["argmax", "threshold"]:
        rows.append(dict(tumors=name, rule=rule, n=len(s), correct=int((s[rule] == s["ref"]).sum()),
                         translocations=len(tr), detected_right_class=int((tr[rule] == tr["ref"]).sum()),
                         missed=int((tr[rule] == "No Trx").sum()), wrong_class=int(((tr[rule] != "No Trx") & (tr[rule] != tr["ref"])).sum()),
                         no_translocation=len(neg), called_translocation=int((neg[rule] != "No Trx").sum())))
L = pd.DataFrame(rows)
L.to_csv(OUT / "translocation_classifier_loo.tsv", sep="\t", index=False)
log(L.to_string(index=False))
s = loo[(loo["FISHorWGS"] != "t(6;14)") & (loo["threshold"] != loo["ref"]) & loo["pt"].notna()]
log("errors under the cohort's rule, cohort patients:", s[["pt", "FISHorWGS", "threshold"]].to_dict("records"))
log("t(6;14) tumor left out by the original script (its class has no training example when it is held out); call:", loo.loc[loo["FISHorWGS"] == "t(6;14)", ["pt", "argmax", "threshold"]].to_dict("records"))

# ---- 3. few malignant PCs: each tumor scored by the model fitted without it -----------------------------------------
keep = loo[loo["FISHorWGS"] != "t(6;14)"].copy()
keep["key"] = keep["Sample"]
big = keep[[len(pops[k][0]) >= max(SIZES) if k in pops else False for k in keep["key"]]]
log("\n3. tumors with >= %d malignant PCs: %d of %d (cohort patients: %d)" % (max(SIZES), len(big), len(keep), big["pt"].notna().sum()))
rows = []
calls = {n: [] for n in SIZES}
for _, r in big.iterrows():
    for n in SIZES:
        F = subset_features(pops[r["key"]], n)
        calls[n].append((r["ref"], r["pt"], rule_threshold(coef_by_model[r["model"]], F)[0]))
for name, flt in [("all training tumors", lambda p: True), ("cohort patients", lambda p: isinstance(p, str))]:
    for n in SIZES:
        v = [(a, c) for a, p, c in calls[n] if flt(p)]
        tr = [(a, c) for a, c in v if a != "No Trx"]
        ne = [(a, c) for a, c in v if a == "No Trx"]
        rows.append(dict(tumors=name, cells_per_call=n, n_tumors=len(v), pct_agreement=100 * np.mean([np.mean(c == a) for a, c in v]),
                         pct_translocations_detected=100 * np.mean([np.mean(c == a) for a, c in tr]),
                         pct_translocations_missed=100 * np.mean([np.mean(c == "No Trx") for a, c in tr]),
                         pct_wrong_translocation=100 * np.mean([np.mean((c != "No Trx") & (c != a)) for a, c in tr]),
                         pct_false_translocation=100 * np.mean([np.mean(c != "No Trx") for a, c in ne])))
for name, q in [("all training tumors", big), ("cohort patients", big[big["pt"].notna()])]:
    tr, ne = q[q["ref"] != "No Trx"], q[q["ref"] == "No Trx"]
    rows.append(dict(tumors=name, cells_per_call="all", n_tumors=len(q), pct_agreement=100 * (q["threshold"] == q["ref"]).mean(),
                     pct_translocations_detected=100 * (tr["threshold"] == tr["ref"]).mean(), pct_translocations_missed=100 * (tr["threshold"] == "No Trx").mean(),
                     pct_wrong_translocation=100 * ((tr["threshold"] != "No Trx") & (tr["threshold"] != tr["ref"])).mean(), pct_false_translocation=100 * (ne["threshold"] != "No Trx").mean()))
C = pd.DataFrame(rows)
C.to_csv(OUT / "translocation_classifier_cells_per_call.tsv", sep="\t", index=False)
log(C.round(1).to_string(index=False))

# ---- 4. false positives in normal PCs -------------------------------------------------------------------------------
don = cells[cells["pt"].isin(donors) & (cells["annotation3"] == "Normal")]
log("\n4. donor PCs: %d cells, %d donors" % (len(don), don["pt"].nunique()))
rows = []


def add(source, unit, n, F, n_ind):
    c, p = rule_threshold(final, F)
    rows.append(dict(source=source, unit=unit, cells_per_call=n, n_individuals=n_ind, n_calls=len(c), n_called_translocation=int((c != "No Trx").sum()),
                     max_class_probability=float(p.max()), median_max_class_probability=float(np.median(p))))


if len(don):
    add("healthy donors", "all PCs of each donor", "all", np.vstack([features(as_arrays(q)) for _, q in don.groupby("pt")]), don["pt"].nunique())
    bm = don[don["sample_type"] == "BM"]
    add("healthy donors", "PCs of each BM library", "all", np.vstack([features(as_arrays(q)) for _, q in bm.groupby("gex_id")]), bm["pt"].nunique())
    log("donor BM libraries:", bm["gex_id"].nunique())
nrm = cells[cells["pt"].notna() & (~cells["pt"].isin(donors)) & (cells["annotation3"] == "Normal")]
add("patients' normal PCs", "all normal PCs of each patient", "all", np.vstack([features(as_arrays(q)) for _, q in nrm.groupby("pt")]), nrm["pt"].nunique())
det = []
for lab, src in [("healthy donors", don), ("patients' normal PCs", nrm)]:
    g = {p: as_arrays(q) for p, q in src.groupby("pt")}
    for n in SIZES:
        ok = [p for p in g if len(g[p][0]) >= n]
        if ok:
            F = np.vstack([subset_features(g[p], n) for p in ok])
            add(lab, "random sets of PCs", n, F, len(ok))
            if lab == "healthy donors":
                c = rule_threshold(final, F)[0]
                det += [(d, n, x) for d, x in zip(np.repeat(ok, N_REP), c) if x != "No Trx"]
N = pd.DataFrame(rows)
N.to_csv(OUT / "translocation_classifier_donor_calls.tsv", sep="\t", index=False)
log(N.round(3).to_string(index=False))

# which donors and classes produce the calls in small random sets (the sets of the table above)
det = pd.DataFrame(det, columns=["donor", "cells", "call"])
n_sets = int(N[(N["source"] == "healthy donors") & (N["unit"] == "random sets of PCs")]["n_calls"].sum())
log("donor random sets: %d of %d called (%.1f%%); donors with any call %d of %d; classes %s; calls per donor %s" % (
    len(det), n_sets, 100 * len(det) / n_sets, det["donor"].nunique(), don["pt"].nunique(), det["call"].value_counts().to_dict(), det["donor"].value_counts().tolist()))
top = det["donor"].value_counts().index[0]
one = rule_threshold(final, features(as_arrays(don[don["pt"] == top]))[None, :])
log("donor with most calls: %s, %d of its 100 sets; its call on all its PCs %s, probability %.2f; PCs %d" % (top, det["donor"].eq(top).sum(), one[0][0], one[1][0], (don["pt"] == top).sum()))
# patients' normal PCs: is the call the patient's own tumor group, as expected if malignant cells are labeled normal?
cy = cyto.copy()
own = cy[cy["tumor_id"].str.endswith(":0")].drop_duplicates("patient_id").set_index("patient_id")["regression_pred"].replace({"no_translocation_detected": "No Trx"})
allcy = cy.assign(reg=cy["regression_pred"].replace({"no_translocation_detected": "No Trx"})).groupby("patient_id")["reg"].apply(set)
rows2 = []
for p, q in nrm.groupby("pt"):
    c, pr = rule_threshold(final, features(as_arrays(q))[None, :])
    rows2.append(dict(pt=p, n_normal=len(q), call=c[0], prob=pr[0], own=own.get(p, "no tumor call"), own_all=allcy.get(p, set())))
R = pd.DataFrame(rows2)
called = R[R["call"] != "No Trx"]
same = [r["call"] in r["own_all"] for _, r in called.iterrows()]
log("patients' normal PCs, whole population: %d of %d called; same group as the patient's own tumor %d; other %d; patient has no tumor call %d" % (
    len(called), len(R), sum(same), (~np.array(same)).sum() - (called["own"] == "no tumor call").sum(), (called["own"] == "no tumor call").sum()))
log("  median normal PCs in called %d vs not called %d" % (called["n_normal"].median(), R[R["call"] == "No Trx"]["n_normal"].median()))
log("  called with another group:", called[[not x for x in same]][["pt", "n_normal", "call", "own"]].to_dict("records"))
R.drop(columns="own_all").to_csv(OUT / "translocation_classifier_patient_normal_calls.tsv", sep="\t", index=False)

# ---- 5. BM and PB calls of the same patient ---------------------------------------------------------------------------
overall["pt"] = overall["PatientID"].map(pang2pt)
o = overall[overall["pt"].notna()].copy()
for col in ["model_agg_predictions", "predictions"]:
    w = o.pivot_table(index="PatientID_TumorID", columns="Tissue", values=col, aggfunc="first").dropna()
    log("\n5. %s: populations with both BM and PB %d, same call %d" % (col, len(w), (w["BM"] == w["PB"]).sum()))
pb = pd.read_csv(result("genomics", "translocation_classifier_pb_predictions.csv"))
pb["ref"] = pb["FISHorWGS"].replace({"HRD": "No Trx"})
log("final model on PB of tumors with FISH or WGS: %d of %d correct" % ((pb["predictions"] == pb["ref"]).sum(), len(pb)))

# ---- 5b. the four patients with two malignant clones: stability of each clone's call --------------------------------
two = sorted(cyto.loc[cyto["tumor_id"].str.endswith(":1"), "patient_id"])
rows3 = []
for k, a in pops.items():
    pid = pang2pt.get(k.split(":")[0])
    if pid in two:
        clone, tissue = k.split(":")[1].split(";")
        rec = dict(pt=pid, clone=clone, tissue=tissue, n_cells=len(a[0]), call_all_cells=rule_threshold(final, features(a)[None, :])[0][0])
        if len(a[0]) >= 100:
            rec["calls_100_cell_sets"] = pd.Series(rule_threshold(final, subset_features(a, 100))[0]).value_counts().to_dict()
        rows3.append(rec)
T2 = pd.DataFrame(rows3).sort_values(["pt", "clone", "tissue"])
T2.to_csv(OUT / "translocation_classifier_two_clone_calls.tsv", sep="\t", index=False)
log("\n5b. two-clone patients, final model"); log(T2.to_string(index=False))

# ---- 5c. sizes of the populations whose call is in the cytogenetic table; calls without FISH or WGS support -------------
overall["clone"] = overall["PatientID_TumorID"].str.split(":").str[-1].map({"Tumor": 0, "Tumor1": 0, "Tumor2": 1})
sz = pd.DataFrame([dict(pt=pang2pt.get(k.split(":")[0]), clone={"Tumor": 0, "Tumor1": 0, "Tumor2": 1}.get(k.split(":")[1].split(";")[0]), tissue=k.split(";")[-1], n=len(a[0])) for k, a in pops.items()])
sz = sz[sz["pt"].isin(cyto["patient_id"])]
used = pd.concat([sz[sz["tissue"] == "BM"], sz[sz["tissue"] == "PB"]]).drop_duplicates(["pt", "clone"])
log("\n5c. populations in the cytogenetic table: %d; malignant PCs per call %d to %d (median %d); from PB because no BM call: %s" % (
    len(used), used["n"].min(), used["n"].max(), used["n"].median(), sorted(used[used["tissue"] == "PB"]["pt"].tolist())))
valid = ["t(11;14)", "t(4;14)", "t(14;16)", "t(14;20)", "t(6;14)", "t(8;14)", "t(12;14)", "HRD"]
cyto["orthogonal"] = cyto["FISH"].isin(valid) | cyto["WGS"].isin(valid)
log("calls by group and FISH or WGS support (rows of the cytogenetic table):")
log(pd.crosstab(cyto["scRNA"], cyto["orthogonal"]).to_string())

# ---- 6. training tumors outside the final cohort --------------------------------------------------------------------
out = loo[loo["pt"].isna()]
log("\n6. training tumors outside the cohort: %d of %d; reference groups %s" % (len(out), len(loo), out["FISHorWGS"].value_counts().to_dict()))

# ---- 7. normal PCs by tissue ------------------------------------------------------------------------------------------
pop_rows, set_rows = [], []
pop_feat = {}
for lab, src in [("healthy donors", don), ("patients", nrm)]:
    for tissue in ["BM", "PB"]:
        g = {p: as_arrays(q) for p, q in src[src["sample_type"] == tissue].groupby("pt")}
        F = np.vstack([features(a) for a in g.values()])
        n = np.array([len(a[0]) for a in g.values()])
        c, pr = rule_threshold(final, F)
        pop_feat[(lab, tissue)] = (list(g), F, n)
        pop_rows.append(dict(source=lab, tissue=tissue, n_populations=len(g), n_called=int((c != "No Trx").sum()),
                             n_populations_30=int((n >= 30).sum()), n_called_30=int(((c != "No Trx") & (n >= 30)).sum()), median_PCs=int(np.median(n)),
                             max_probability=float(pr.max()), classes_called=pd.Series(c[c != "No Trx"]).value_counts().to_dict()))
        for k in SIZES:
            ok = [p for p in g if len(g[p][0]) >= k]
            if ok:
                cs = rule_threshold(final, np.vstack([subset_features(g[p], k) for p in ok]))[0]
                set_rows.append(dict(source=lab, tissue=tissue, cells_per_call=k, n_individuals=len(ok), n_sets=len(cs), n_called=int((cs != "No Trx").sum()), pct_called=100 * (cs != "No Trx").mean()))
P7, S7 = pd.DataFrame(pop_rows), pd.DataFrame(set_rows)
P7.to_csv(OUT / "translocation_classifier_normal_by_tissue_populations.tsv", sep="\t", index=False)
S7.to_csv(OUT / "translocation_classifier_normal_by_tissue_sets.tsv", sep="\t", index=False)
log("\n7. normal PCs by tissue, whole populations"); log(P7.round(3).to_string(index=False))
log("random sets"); log(S7.round(1).to_string(index=False))
log("HD-2 tissues with PCs:", don[don["pt"] == "HD-2"]["sample_type"].value_counts().to_dict())

# ---- 8. which feature drives the calls in blood normal PCs --------------------------------------------------------------
bm_ref = np.vstack([pop_feat[("healthy donors", "BM")][1][pop_feat[("healthy donors", "BM")][2] >= 30], pop_feat[("patients", "BM")][1][pop_feat[("patients", "BM")][2] >= 30]])
ref = np.median(bm_ref, axis=0)
rows8 = []
for lab in ["patients", "healthy donors"]:
    _, F, n = pop_feat[(lab, "PB")]
    F = F[n >= 30]
    rows8.append(dict(source=lab, feature="(none)", n_populations=len(F), n_called=int((rule_threshold(final, F)[0] != "No Trx").sum())))
    for j, fe in enumerate(FEATS):
        G_ = F.copy()
        G_[:, j] = ref[j]
        rows8.append(dict(source=lab, feature=fe, n_populations=len(F), n_called=int((rule_threshold(final, G_)[0] != "No Trx").sum())))
C8 = pd.DataFrame(rows8)
C8.to_csv(OUT / "translocation_classifier_pb_normal_counterfactual.tsv", sep="\t", index=False)
log("\n8. PB normal populations (>= 30 PCs): calls when one feature is set to the BM normal median")
log(C8.pivot(index="feature", columns="source", values="n_called").to_string())
j = FEATS.index("ITGB7")
log("ITGB7 population mean, median over populations: BM normal %.3f; PB normal patients %.3f; PB normal donors %.3f; BM malignant %.3f" % (
    ref[j], np.median(pop_feat[("patients", "PB")][1][pop_feat[("patients", "PB")][2] >= 30, j]), np.median(pop_feat[("healthy donors", "PB")][1][pop_feat[("healthy donors", "PB")][2] >= 30, j]),
    np.median(inp.loc[[k for k in inp.index if k.endswith(";BM")], "ITGB7"].values)))

# ---- 9. leave-one-out models on the same patient's blood ---------------------------------------------------------------
pb["pt"] = pb["PatientID"].map(pang2pt).fillna("outside cohort")
m_of = loo.set_index("PatientID_TumorID")["model"].to_dict()
q = pb[pb["PatientID_TumorID"].isin(m_of)].reset_index(drop=True)
Xp = q[FEATS].values.astype(float)
q["loo_call"] = [rule_threshold(coef_by_model[m_of[t]], Xp[[i]])[0][0] for i, t in enumerate(q["PatientID_TumorID"])]
q["final_call"] = rule_threshold(final, Xp)[0]
q["n_cells"] = [len(pops[k][0]) for k in q["Sample"]]
hasclass = q["FISHorWGS"] != "t(6;14)"
log("\n9. PB of tumors in the leave-one-out set: %d of %d PB rows; leave-one-out model correct %d of %d; final model correct %d of these" % (
    len(q), len(pb), (q["loo_call"] == q["ref"]).sum(), len(q), (q["final_call"] == q["ref"]).sum()))
log("PB rows not scored here (tumor not in the training set):", len(pb) - len(q))
E9 = q[q["loo_call"] != q["ref"]][["pt", "n_cells", "FISHorWGS", "loo_call", "final_call"]]
E9.to_csv(OUT / "translocation_classifier_pb_loo_errors.tsv", sep="\t", index=False)
log(E9.to_string(index=False))

# ---- 10. errors, reviewed calls and force-calls, as tables ----------------------------------------------------------------
loo["label"] = loo["pt"].fillna("outside cohort")
rows10 = []
for name, sub in [("all training tumors, CCND3 tumor counted", loo), ("cohort patients, CCND3 tumor counted", loo[loo["pt"].notna()])]:
    tr, neg = sub[sub["ref"] != "No Trx"], sub[sub["ref"] == "No Trx"]
    rows10.append(dict(tumors=name, n=len(sub), correct=int((sub["threshold"] == sub["ref"]).sum()), translocations=len(tr), detected_right_class=int((tr["threshold"] == tr["ref"]).sum()),
                       missed=int((tr["threshold"] == "No Trx").sum()), wrong_class=int(((tr["threshold"] != "No Trx") & (tr["threshold"] != tr["ref"])).sum()),
                       no_translocation=len(neg), called_translocation=int((neg["threshold"] != "No Trx").sum())))
L10 = pd.DataFrame(rows10)
L10.to_csv(OUT / "translocation_classifier_loo_ccnd3_counted.tsv", sep="\t", index=False)
log("\n10. leave-one-out with the CCND3 tumor counted"); log(L10.to_string(index=False))
E10 = loo[loo["threshold"] != loo["ref"]][["label", "FISHorWGS", "threshold"]].rename(columns={"threshold": "call"}).sort_values(["FISHorWGS", "label"])
E10.to_csv(OUT / "translocation_classifier_loo_errors.tsv", sep="\t", index=False)
log("errors, all training tumors:"); log(E10.to_string(index=False))
R10 = overall[overall["pt"].notna() & (overall["model_agg_predictions"] != overall["predictions"])].assign(clone=lambda d: d["PatientID_TumorID"].str.split(":").str[-1])[["pt", "Tissue", "clone", "model_agg_predictions", "predictions", "FISHorWGS"]].sort_values(["Tissue", "pt"])
R10.to_csv(OUT / "translocation_classifier_reviewed_calls.tsv", sep="\t", index=False)
log("calls changed by review, cohort patients: %d rows, %d BM" % (len(R10), (R10["Tissue"] == "BM").sum())); log(R10.to_string(index=False))
F10 = overall[overall["Numbat_HRD"] != overall["Numbat_HRD_Final"]].assign(label=lambda d: d["PatientID"].map(pang2pt).fillna("outside cohort"))
log("HRD force-calls: %d rows, %d tumors, of which outside the cohort %d; in the cohort: %s" % (len(F10), F10["PatientID_TumorID"].nunique(), F10[F10["label"] == "outside cohort"]["PatientID_TumorID"].nunique(), F10[F10["label"] != "outside cohort"][["label", "predictions", "FinalClassification"]].drop_duplicates().to_dict("records")))
F10[["label", "Tissue", "predictions", "Numbat_HRD", "Numbat_HRD_Final", "FinalClassification"]].to_csv(OUT / "translocation_classifier_hrd_force_calls.tsv", sep="\t", index=False)

# ---- 11. clonality of the BM normal populations that receive a call --------------------------------------------------------
def top_share(q):
    v = q.loc[~q["clonotype"].isin(["nan", "None", ""]), "clonotype"]
    return (v.value_counts(normalize=True).iloc[0] if len(v) else np.nan), len(v)


rows11 = []
for p, q in nrm[nrm["sample_type"] == "BM"].groupby("pt"):
    if len(q) < 30:
        continue
    c = rule_threshold(final, features(as_arrays(q))[None, :])[0][0]
    sh, nv = top_share(q)
    rows11.append(dict(pt=p, n_normal=len(q), n_with_bcr=nv, call=c, own=own.get(p, "no tumor call"), top_clonotype_share=sh, pct_CCND1_detected=100 * (q["CCND1"] > 0).mean()))
B11 = pd.DataFrame(rows11)
B11.to_csv(OUT / "translocation_classifier_bm_normal_clonality.tsv", sep="\t", index=False)
cal = B11[B11["call"] != "No Trx"]
log("\n11. BM normal populations with >= 30 PCs: %d, called %d; top clonotype share in called: median %.1f%%, range %.1f-%.1f%%; not called: median %.1f%%" % (
    len(B11), len(cal), 100 * cal["top_clonotype_share"].median(), 100 * cal["top_clonotype_share"].min(), 100 * cal["top_clonotype_share"].max(), 100 * B11[B11["call"] == "No Trx"]["top_clonotype_share"].median()))
log(cal.round(3).to_string(index=False))
log("CCND1 detected in normal PCs: BM patients median %.1f%%; donors %.1f%%" % (B11["pct_CCND1_detected"].median(), 100 * (don["CCND1"] > 0).mean()))

# ---- 12. the three calls that rest on blood alone, against the blood signal of section 8 -----------------------------
pb_only = sorted(used[used["tissue"] == "PB"]["pt"])
rows12 = []
for _, r in overall[(overall["Tissue"] == "PB") & overall["pt"].isin(pb_only)].iterrows():
    c = cyto[cyto["patient_id"] == r["pt"]].iloc[0]
    nn = nrm[(nrm["pt"] == r["pt"]) & (nrm["sample_type"] == "PB")]
    rows12.append(dict(pt=r["pt"], n_malignant_PB=len(pops[r["Sample"]][0]), model_call=r["model_agg_predictions"], after_review=r["predictions"], final_group=r["FinalClassification"],
                       FISH=c["FISH"], WGS=c["WGS"], ITGB7=round(float(r["ITGB7"]), 3), n_normal_PB=len(nn)))
T12 = pd.DataFrame(rows12)
T12.to_csv(OUT / "translocation_classifier_pb_only_calls.tsv", sep="\t", index=False)
log("\n12. calls that rest on blood alone"); log(T12.to_string(index=False))
log("median ITGB7 of BM normal populations %.3f, of PB normal populations %.3f" % (ref[j], np.median(pop_feat[("patients", "PB")][1][pop_feat[("patients", "PB")][2] >= 30, j])))

# ---- 13. which cells the min-max scaling of the Zhan scores covered ----------------------------------------------------------
annotated = cells["annotation3"].isin(["Tumor", "Tumor1", "Tumor2", "Normal"])
cohort_pcs = cells["pt"].notna() & annotated
log("\n13. scoring table: %d cells, %d libraries; in the cohort libraries %d cells, %d libraries; annotated malignant or normal in the cohort libraries %d cells" % (
    len(cells), cells["gex_id"].nunique(), cells["pt"].notna().sum(), cells.loc[cells["pt"].notna(), "gex_id"].nunique(), cohort_pcs.sum()))
rows13 = []
for z in ZHAN:
    x, n = cells[z].astype(float), cells[z + ":Norm"].astype(float)
    rebuilt = (x - x.min()) / (x.max() - x.min())
    xp = x[cohort_pcs]
    rebuilt_p = (xp - xp.min()) / (xp.max() - xp.min())
    rows13.append(dict(score=z, norm_min_all=n.min(), norm_max_all=n.max(), max_diff_scaled_on_all_cells=(rebuilt - n).abs().max(),
                       norm_min_cohort_PCs=n[cohort_pcs].min(), norm_max_cohort_PCs=n[cohort_pcs].max(), max_diff_if_scaled_on_cohort_PCs=(rebuilt_p - n[cohort_pcs]).abs().max()))
Z13 = pd.DataFrame(rows13)
Z13.to_csv(OUT / "translocation_classifier_scaling_check.tsv", sep="\t", index=False)
log(Z13.to_string(index=False, float_format=lambda v: "%.3g" % v))
print("done")
FH.close()
