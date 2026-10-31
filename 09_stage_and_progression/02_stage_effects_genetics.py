#!/usr/bin/env python3
"""Stage associations of the MST score and of the five progression-associated signatures with and without adjustment
for genetic alterations, and malignancy and MST scores by cytogenetic group (Supplementary Figs. 14D and 16A; Results).

Purpose:      Units: the mean over the malignant BM PCs of one patient at one disease stage, for units with at least
              30 such cells (patients sampled at two stages contribute two units; SMM risk groups follow the IMWG
              2/20/20 model, risk.py, and units of one patient that share a risk group are pooled). Each outcome (MST score and the five
              progression-associated signature activities) is z-scored across units and regressed by ordinary least
              squares on disease stage coded 0-4 (MGUS, LRSMM, IRSMM, HRSMM, MM; coefficient = change per stage step),
              without and with adjustment for the primary translocation group, HRD, Numbat-detected 1q gain, del(1p),
              del(13q), del(14q), del(17p) and 8q24 (MYC) gain, and the mean MYC expression of the unit (z-scored);
              standard errors clustered by patient; two-sided Wald tests. Sensitivity: stage as four indicators (MGUS
              reference), adjusted. Malignancy and MST scores across the nine cytogenetic groups: Kruskal-Wallis test
              and, adjusted for stage, a joint Wald test of the translocation and HRD terms.

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv (redcap_id, patient_id);
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (per cell: redcap_id, sample_type, annotation,
              disease_status); data/signature/score/cell-level-cytogenetics-gene-expr.pkl (MYC, log-normalized, per
              cell); data/paper/classifier/patient_id_mean_probability.pkl (sample, disease_status, sample_type,
              annotation, "MST Score", "Malignancy Score"); data/paper/signatures/per-sample-mean-expr-score.pkl
              (patient_id, disease_status, sample_type, annotation, mean_expr:<signature>);
              data/paper/cytogenetics-status.csv (patient_id, cytogenetics, hrd_status);
              data/paper/cnvs/cnv_segments_152_tumors.csv (patient_id, driver_arm_cnv: per-segment driver calls, e.g.
              1q+, 13q-, 8q+ = gain covering 8q24.21/MYC).

Outputs:      results/stage_and_progression/bm_tumor_units.tsv: one row per unit (patient, stage, outcomes, covariates).
              results/stage_and_progression/stage_effects_genetics.tsv: stage coefficient per outcome, unadjusted and adjusted.
              results/stage_and_progression/stage_effects_genetics_coefficients.tsv: every coefficient of the adjusted models.
              results/stage_and_progression/stage_effects_genetics_tests.tsv: Wald and Kruskal-Wallis tests.
              results/stage_and_progression/stage_effects_genetics_log.txt: units, covariates and model summaries.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "stage_effects_genetics_log.txt"), "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
SIGS = {"mean_expr:cell_cycling_G1S": "Cell cycling G1/S", "mean_expr:cell_cycling_G2M": "Cell cycling G2/M",
        "mean_expr:interferon_inducible": "Interferon inducible", "mean_expr:histone_chaperone_gene": "Histone/chaperone",
        "mean_expr:B_cell_activation": "B cell activation"}
CNVS = {"1q+": "1q gain", "1p-": "del(1p)", "13q-": "del(13q)", "14q-": "del(14q)", "17p-": "del(17p)",
        "8q+": "8q24 (MYC) gain"}
TRANS = {"t(11;14)": "t(11;14)", "t(4;14)": "t(4;14)", "t(14;16)": "MAF translocation",
         "t(14;20)": "MAF translocation", "t(8;14)": "MAF translocation", "t(6;14)": "Other IgH translocation",
         "t(12;14)": "Other IgH translocation"}
MIN_CELLS = 30
TESTS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def add_test(name, detail, n, stat, p):
    TESTS.append(dict(test=name, detail=detail, n=n, statistic=stat, P=p))
    log("  TEST %s | %s | n=%s | stat=%s | P=%.3g" % (name, detail, n, stat, p))


def z(s):
    return (s - s.mean()) / s.std()


# ---- 1. units: malignant BM PCs of one patient at one stage -------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
red2pt = meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"]
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[["redcap_id", "sample_type", "annotation", "disease_status"]]
ann = ann[(ann["annotation"].astype(str) == "tumor") & (ann["sample_type"].astype(str) == "BM")].copy()
ann["patient_id"] = ann["redcap_id"].astype(str).map(red2pt)
ann["disease_status"] = ann["disease_status"].astype(str)
assert ann["patient_id"].notna().all()
n_recorded = ann.groupby(["patient_id", "disease_status"]).size()       # cells per unit of the cell annotation (pooling weights)
ann["disease_status"] = stages(ann["patient_id"], ann["sample_type"], ann["disease_status"])

# MYC: mean log-normalized expression over the unit's malignant BM cells
myc = pd.read_pickle(data("signature", "score", "cell-level-cytogenetics-gene-expr.pkl"))["MYC"]
ann = ann.join(myc, how="left")
log("malignant BM PCs: %d | without MYC value: %d" % (len(ann), ann["MYC"].isna().sum()))
units = ann.groupby(["patient_id", "disease_status"]).agg(n_cells=("MYC", "size"), MYC=("MYC", "mean"))
units = units[units["n_cells"] >= MIN_CELLS].reset_index()
units = units[units["disease_status"].isin(STAGES)]

mst = pd.read_pickle(data("paper", "classifier", "patient_id_mean_probability.pkl"))
mst = mst[(mst["annotation"] == "tumor") & (mst["sample_type"] == "BM")]
mst = mst[["sample", "disease_status", "MST Score", "Malignancy Score"]].rename(columns={"sample": "patient_id"})
sig = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
sig = sig[(sig["annotation"] == "tumor") & (sig["sample_type"] == "BM")][["patient_id", "disease_status"] + list(SIGS)]


def imwg_units(d):
    """IMWG risk groups for a per-unit table; units that become one are pooled (cell-count weighted means)."""
    d = d.copy()
    d["patient_id"] = d["patient_id"].astype(str)
    d["disease_status"] = d["disease_status"].astype(str)
    d["w"] = [n_recorded.get(k, np.nan) for k in zip(d["patient_id"], d["disease_status"])]
    d["disease_status"] = stages(d["patient_id"], ["BM"] * len(d), d["disease_status"])
    d = d[d["w"].notna()]
    vals = [c for c in d.columns if c not in ("patient_id", "disease_status", "w")]
    return d.groupby(["patient_id", "disease_status"]).apply(
        lambda g: pd.Series({c: np.average(g[c].astype(float), weights=g["w"]) for c in vals})).reset_index()


mst, sig = imwg_units(mst), imwg_units(sig)
units = units.merge(mst, on=["patient_id", "disease_status"], how="left").merge(sig, on=["patient_id", "disease_status"], how="left")
log("BM units with >= 30 malignant PCs: %d (%d patients); missing MST: %d; missing signatures: %d"
    % (len(units), units["patient_id"].nunique(), units["MST Score"].isna().sum(), units[list(SIGS)].isna().any(axis=1).sum()))

# ---- 2. genetic covariates (patient level) -----------------------------------------------------
cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"), index_col=0)
cyto = cyto[cyto.index.str.endswith("tumor:0")].set_index("patient_id")
units = units.join(cyto[["cytogenetics", "hrd_status"]], on="patient_id")
log("units without a cytogenetic call:", units["cytogenetics"].isna().sum())
units = units.dropna(subset=["cytogenetics"])
units["translocation"] = units["cytogenetics"].map(TRANS).fillna("None")
units["HRD"] = (units["hrd_status"] == "Y").astype(int)

segs = pd.read_csv(data("paper", "cnvs", "cnv_segments_152_tumors.csv"))
units["numbat_cnv"] = units["patient_id"].isin(segs["patient_id"]).astype(int)
for ev in CNVS:
    units[ev] = units["patient_id"].isin(segs.loc[segs["driver_arm_cnv"] == ev, "patient_id"]).astype(int)
units["stage"] = units["disease_status"].map(STAGES.index)
units = units.sort_values(["stage", "patient_id"]).reset_index(drop=True)

log("\nunits by stage:", units["disease_status"].value_counts().reindex(STAGES).to_dict())
log("patients:", units["patient_id"].nunique(), "| patients with 2 units:", (units["patient_id"].value_counts() > 1).sum())
log("translocation groups (units):", units["translocation"].value_counts().to_dict(), "| HRD:", units["HRD"].sum())
log("Numbat: units of tumors with any CNV:", units["numbat_cnv"].sum())
log(pd.DataFrame({ev: units.groupby("disease_status")[ev].sum().reindex(STAGES) for ev in CNVS}).to_string())
log("MYC expression by stage (median of unit means):", units.groupby("disease_status")["MYC"].median().reindex(STAGES).round(3).to_dict())
log("MYC expression, 8q24 gain vs not (median):", units.groupby("8q+")["MYC"].median().round(3).to_dict())

# ---- 3. models ---------------------------------------------------------------------------------
COV = (["t(11;14)", "t(4;14)", "MAF translocation", "Other IgH translocation", "HRD"] + list(CNVS) + ["MYC (z)"])
X_gen = pd.DataFrame({t: (units["translocation"] == t).astype(int) for t in COV[:4]})
X_gen["HRD"] = units["HRD"]
for ev in CNVS:
    X_gen[ev] = units[ev]
X_gen["MYC (z)"] = z(units["MYC"])
groups = units["patient_id"].astype("category").cat.codes.values


def fit(y, X):
    ok = y.notna() & X.notna().all(axis=1)
    return sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="cluster", cov_kwds={"groups": groups[ok.values]}), ok.sum()


def wald(res, names):
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, list(res.params.index).index(n)] = 1
    w = res.wald_test(R, scalar=True)
    return float(np.squeeze(w.statistic)), float(w.pvalue)


rows, coefs = [], []
outcomes = {"MST Score": "MST score"}
outcomes.update(SIGS)
for col, lab in outcomes.items():
    y = z(units[col])
    stage = units[["stage"]]
    un, n_un = fit(y, stage)
    ad, n_ad = fit(y, pd.concat([stage, X_gen], axis=1))
    sd = units[col].std()
    for model, res, n in [("unadjusted", un, n_un), ("adjusted", ad, n_ad)]:
        lo, hi = res.conf_int().loc["stage"]
        rows.append(dict(outcome=lab, model=model, n_units=n, coef_sd_per_stage=res.params["stage"], ci_low=lo, ci_high=hi,
                         P=res.pvalues["stage"], coef_raw_per_stage=res.params["stage"] * sd, r2=res.rsquared))
    gen_stat, gen_p = wald(ad, COV)
    gonly, _ = fit(y, X_gen)
    rows[-1].update(retained_pct=100 * ad.params["stage"] / un.params["stage"], genetics_wald=gen_stat, genetics_P=gen_p,
                    r2_genetics_only=gonly.rsquared)
    log("\n%s: stage per step %.3f SD (P = %.2g) unadjusted, %.3f SD (P = %.2g) adjusted (%.0f%% retained); "
        "raw %.4f -> %.4f; R2 stage %.3f, genetics %.3f, both %.3f; genetics jointly P = %.2g"
        % (lab, un.params["stage"], un.pvalues["stage"], ad.params["stage"], ad.pvalues["stage"], rows[-1]["retained_pct"],
           un.params["stage"] * sd, ad.params["stage"] * sd, un.rsquared, gonly.rsquared, ad.rsquared, gen_p))
    ci = ad.conf_int()
    for k in ad.params.index[1:]:
        coefs.append(dict(outcome=lab, term=CNVS.get(k, k), coef_sd=ad.params[k], ci_low=ci.loc[k, 0], ci_high=ci.loc[k, 1], P=ad.pvalues[k]))
    # sensitivity: stage as a categorical variable (4 indicators, MGUS reference), adjusted
    dummies = pd.DataFrame({s: (units["disease_status"] == s).astype(int) for s in STAGES[1:]})
    cat, n_cat = fit(y, pd.concat([dummies, X_gen], axis=1))
    st, p = wald(cat, STAGES[1:])
    add_test("Wald (cluster-robust)", "%s: stage as 4 indicators, adjusted for genetics" % lab, n_cat, round(st, 2), p)
    add_test("Wald (cluster-robust)", "%s: MM vs MGUS, adjusted (%.2f SD)" % (lab, cat.params["MM"]), n_cat, round(cat.tvalues["MM"], 2), cat.pvalues["MM"])

# ---- 4. scores by primary cytogenetic group (Supplementary Fig. 14D; Results) -------------------
# unadjusted (nine groups as in Supplementary Fig. 14D), then the 5 translocation/HRD terms adjusted for stage
for col in ["MST Score", "Malignancy Score"]:
    d = units.dropna(subset=[col])
    st, p = kruskal(*[g[col].values for _, g in d.groupby("cytogenetics")])
    add_test("Kruskal-Wallis", "%s across %d cytogenetic groups (BM units)" % (col, d["cytogenetics"].nunique()), len(d), round(st, 2), p)
    res, n = fit(z(units[col]), pd.concat([units[["stage"]], X_gen[COV[:5]]], axis=1))
    st, p = wald(res, COV[:5])
    add_test("Wald (cluster-robust)", "%s: translocation group + HRD jointly, adjusted for stage" % col, n, round(st, 2), p)
    log("    " + ", ".join("%s %.2f SD (P = %.2g)" % (k, res.params[k], res.pvalues[k]) for k in COV[:5]))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage_effects_genetics.tsv"), sep="\t", index=False)
pd.DataFrame(coefs).to_csv(os.path.join(OUT, "stage_effects_genetics_coefficients.tsv"), sep="\t", index=False)
pd.DataFrame(TESTS).to_csv(os.path.join(OUT, "stage_effects_genetics_tests.tsv"), sep="\t", index=False)
keep = ["patient_id", "disease_status", "n_cells", "MST Score", "Malignancy Score"] + list(SIGS) + \
       ["cytogenetics", "translocation", "HRD", "numbat_cnv"] + list(CNVS) + ["MYC"]
units[keep].to_csv(os.path.join(OUT, "bm_tumor_units.tsv"), sep="\t", index=False)
log("\ndone")
