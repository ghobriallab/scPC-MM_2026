#!/usr/bin/env python3
"""Circulatory capacity signature in CoMMpass and in paired PB and BM samples (Results; Supplementary Fig. 9B-D; Methods).

Purpose:      1. CoMMpass, bulk RNA-seq of CD138-selected BM cells (754 baseline samples): activity of the signature (mean
              TPM of its 39 genes) against the percentage of PCs in blood measured by flow cytometry at the visit of the
              RNA sample (mean of the low and high values): compared between tertiles of activity (cut-offs over all
              754 patients) with two-sided Wilcoxon rank-sum tests and Benjamini-Hochberg correction; Spearman's rho
              with a 95% CI from 2,000 bootstrap resamples (numpy default_rng(42)). Spearman correlations of the
              percentage of PCs in BM (flow cytometry) with the activity and with the blood percentage, and the partial
              Spearman correlation of activity and blood percentage adjusted for the BM percentage.
              2. CoMMpass CD138-selected PB and BM samples of the same visit (one pair per patient, the earliest visit):
              activity in PB versus BM, two-sided Wilcoxon signed-rank test; per gene, pairs with higher PB expression;
              PC marker expression in PB relative to BM.
              3. This cohort: mean activity per individual, disease stage, tissue and PC status; PB versus BM for
              malignant and for normal PCs, and malignant versus normal PCs within PB and within BM, two-sided Wilcoxon
              signed-rank tests.

Inputs:       data/commpass/MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv (rows = PUBLIC_ID): SPECTRUM_SEQ,
              mean_expr:circulatory_capacity. data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv: SPECTRUM_SEQ,
              D_IM_FLOWCYT_PCT_PC_IN_PB_LOW/_HIGH, D_IM_FLOWCYT_PCT_PC_IN_BM_LOW/_HIGH.
              data/commpass/raw_data/MMRF_CoMMpass_IA22_salmon_geneUnstranded_tpm.scRNA_compatible.tsv: gene TPM of
              every CoMMpass CD138+ sample, BM and PB. data/commpass/MMRF_CoMMpass_IA22_BM-PB_pair_id.csv: public_id,
              sample_id, BM, PB. data/signature/hotspot/consensus-modules_genes.csv: gene (index), context.
              data/paper/signatures/per-sample-mean-expr-score.pkl: mean_expr:circulatory_capacity, patient_id,
              sample_type, annotation, disease_status. data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl and
              data/metadata/tumor-cohort-metadata-final-240325.csv: cells per unit (pooling weights); SMM risk groups of
              the IMWG 2/20/20 model (risk.py).

Outputs:      results/commpass/circulatory_signature_patients.tsv: CoMMpass patients with blood flow cytometry (activity,
              tertile, % PCs in blood and in BM; controlled-access patient-level data).
              results/commpass/circulatory_signature_tertiles.tsv: n, median % PCs in blood, % with detectable blood PCs.
              results/commpass/circulatory_signature_pairs.tsv: activity in the paired PB and BM samples (patient-level).
              results/commpass/circulatory_signature_pair_genes.tsv: per signature gene, pairs with higher PB expression.
              results/commpass/circulatory_paired_samples.tsv: pairs of this cohort and their signed-rank tests.
              results/commpass/circulatory_signature_tests.tsv: every test and correlation (CoMMpass and this cohort).
              results/commpass/circulatory_signature_log.txt: log.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import pool_units, stages

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

OUT = result_dir("commpass")
FH = open(OUT / "circulatory_signature_log.txt", "w")
SIG = "mean_expr:circulatory_capacity"
TESTS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


def test(analysis, comparison, n, statistic, value, p, ci_low=np.nan, ci_high=np.nan, q=np.nan):
    TESTS.append(dict(analysis=analysis, comparison=comparison, n=n, statistic=statistic, value=value,
                      ci_low=ci_low, ci_high=ci_high, p=p, q=q))


def partial_spearman(x, y, z):
    """Spearman correlation of x and y adjusted for z: Pearson correlation of the rank residuals."""
    r = pd.DataFrame({"x": x, "y": y, "z": z}).rank()
    design = np.c_[np.ones(len(r)), r["z"]]
    res = [r[c] - design @ np.linalg.lstsq(design, r[c], rcond=None)[0] for c in ("x", "y")]
    rho = np.corrcoef(res[0], res[1])[0, 1]
    t = rho * np.sqrt((len(r) - 3) / (1 - rho ** 2))
    return rho, 2 * ss.t.sf(abs(t), len(r) - 3)


# ---- 1. CoMMpass: BM signature activity vs circulating PCs (Supplementary Fig. 9B) -------------------------------
cp = pd.read_csv(data("commpass", "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv"), sep="\t", index_col=0)
q33, q66 = cp[SIG].quantile(q=(0.33, 0.66))                      # tertile cutoffs over all 754 patients
cp["tertile"] = np.where(cp[SIG] > q66, "High", np.where(cp[SIG] > q33, "Medium", "Low"))

visit = pd.read_csv(data("commpass", "MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv"), sep="\t", low_memory=False)
flow = ["D_IM_FLOWCYT_PCT_PC_IN_PB_LOW", "D_IM_FLOWCYT_PCT_PC_IN_PB_HIGH",
        "D_IM_FLOWCYT_PCT_PC_IN_BM_LOW", "D_IM_FLOWCYT_PCT_PC_IN_BM_HIGH"]
visit = visit[visit["SPECTRUM_SEQ"].isin(cp["SPECTRUM_SEQ"])].set_index("SPECTRUM_SEQ")[flow].apply(pd.to_numeric, errors="coerce")
d = cp.join(visit, on="SPECTRUM_SEQ")
d = d.dropna(subset=flow[:2])                                    # patients with blood flow cytometry (633)
d["pb_pct"] = d[flow[:2]].mean(axis=1)                           # % PCs in blood: mean of the low and high values
d["bm_pct"] = d[flow[2:]].mean(axis=1)                           # % PCs in BM by flow cytometry (tumor burden)
d[["SPECTRUM_SEQ", SIG, "tertile", "pb_pct", "bm_pct"]].to_csv(OUT / "circulatory_signature_patients.tsv", sep="\t")
log("CoMMpass: %d of %d patients with blood flow cytometry; tertile cutoffs %.2f, %.2f mean TPM" % (len(d), len(cp), q33, q66))

ORDER = ["Low", "Medium", "High"]
tert = d.groupby("tertile").agg(n=("pb_pct", "size"), median_pb_pct=("pb_pct", "median"),
                                pct_detectable=("pb_pct", lambda s: 100 * (s > 0).mean())).reindex(ORDER)
tert.to_csv(OUT / "circulatory_signature_tertiles.tsv", sep="\t")
log(tert.round(2).to_string())

pairs = [("Low", "Medium"), ("Medium", "High"), ("Low", "High")]
p = [ss.mannwhitneyu(d.loc[d.tertile == a, "pb_pct"], d.loc[d.tertile == b, "pb_pct"], alternative="two-sided").pvalue
     for a, b in pairs]
qv = multipletests(p, method="fdr_bh")[1]
for (a, b), pi, qi in zip(pairs, p, qv):
    test("CoMMpass % PCs in blood by tertile", "%s vs %s" % (a, b), int(d.tertile.isin([a, b]).sum()),
         "two-sided Wilcoxon rank-sum", np.nan, pi, q=qi)
    log("blood PCs %s vs %s: P = %.3g, q = %.3g" % (a, b, pi, qi))

rho, prho = ss.spearmanr(d[SIG], d["pb_pct"])
rng = np.random.default_rng(42)
boot = []
for _ in range(2000):
    i = rng.integers(0, len(d), len(d))
    boot.append(ss.spearmanr(d[SIG].values[i], d["pb_pct"].values[i]).correlation)
lo, hi = np.percentile(boot, [2.5, 97.5])
test("CoMMpass", "signature vs % PCs in blood", len(d), "Spearman rho (bootstrap 95% CI)", rho, prho, lo, hi)
log("Spearman signature vs blood PCs: rho = %.2f (95%% CI %.2f-%.2f), P = %.2g, n = %d" % (rho, lo, hi, prho, len(d)))

b = d.dropna(subset=["bm_pct"])
for label, x, y in [("signature vs % PCs in BM", b[SIG], b["bm_pct"]), ("% PCs in BM vs % PCs in blood", b["bm_pct"], b["pb_pct"])]:
    r, pr = ss.spearmanr(x, y)
    test("CoMMpass", label, len(b), "Spearman rho", r, pr)
    log("Spearman %s: rho = %.2f, P = %.2g, n = %d" % (label, r, pr, len(b)))
r, pr = partial_spearman(b[SIG], b["pb_pct"], b["bm_pct"])
test("CoMMpass", "signature vs % PCs in blood, adjusted for % PCs in BM", len(b), "partial Spearman rho", r, pr)
log("partial Spearman, adjusted for BM PCs: rho = %.2f, P = %.2g, n = %d" % (r, pr, len(b)))

# ---- 2. CoMMpass: CD138+ blood vs BM samples of the same visit (Supplementary Fig. 9C) ------------------------------
genes = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
genes = list(genes.index[genes["context"] == "circulatory_capacity"])
tpm = pd.read_csv(data("commpass", "raw_data", "MMRF_CoMMpass_IA22_salmon_geneUnstranded_tpm.scRNA_compatible.tsv"),
                  sep="\t", index_col=0)
assert np.allclose(tpm.loc[genes, [s + "_1_BM_CD138pos" for s in cp.index]].mean().values, cp[SIG].values)  # same activity as section 1
pb_bm = pd.read_csv(data("commpass", "MMRF_CoMMpass_IA22_BM-PB_pair_id.csv"))
log("CoMMpass BM-PB pairs: %d samples from %d patients" % (len(pb_bm), pb_bm.public_id.nunique()))
all_higher = (tpm.loc[genes, pb_bm.PB].mean().values > tpm.loc[genes, pb_bm.BM].mean().values).sum()
log("PB higher than BM in %d of %d pairs (all visits)" % (all_higher, len(pb_bm)))
pb_bm = pb_bm.assign(visit=pb_bm.sample_id.str.split("_").str[-1].astype(int)).sort_values("visit").drop_duplicates("public_id")
pr_tab = pd.DataFrame({"BM": tpm.loc[genes, pb_bm.BM].mean().values, "PB": tpm.loc[genes, pb_bm.PB].mean().values},
                      index=pb_bm.public_id.values)
pr_tab.to_csv(OUT / "circulatory_signature_pairs.tsv", sep="\t")
w = ss.wilcoxon(pr_tab.PB, pr_tab.BM)
ratio = (pr_tab.PB / pr_tab.BM).median()
test("CoMMpass CD138+ PB vs BM, same visit", "PB vs BM (one pair per patient)", len(pr_tab),
     "two-sided Wilcoxon signed-rank; value = median PB/BM ratio", ratio, w.pvalue)
log("PB vs BM, one pair per patient: PB higher in %d of %d, median ratio %.2f, signed-rank P = %.2g"
    % ((pr_tab.PB > pr_tab.BM).sum(), len(pr_tab), ratio, w.pvalue))
lfc = np.log2((tpm.loc[genes, pb_bm.PB].values + 1) / (tpm.loc[genes, pb_bm.BM].values + 1))
gtab = pd.DataFrame({"pairs_PB_higher": (lfc > 0).sum(1), "n_pairs": lfc.shape[1], "median_log2FC": np.median(lfc, 1)},
                    index=genes)
gtab.to_csv(OUT / "circulatory_signature_pair_genes.tsv", sep="\t")
log("signature genes higher in PB in >= 9 of %d pairs: %d of %d" % (lfc.shape[1], (gtab.pairs_PB_higher >= 9).sum(), len(genes)))
pc_markers = ["SDC1", "TNFRSF17", "MZB1", "JCHAIN", "SLAMF7", "XBP1", "PRDM1"]   # PC content of each sample
pc_ratio = (tpm.loc[pc_markers, pb_bm.PB].mean().values / tpm.loc[pc_markers, pb_bm.BM].mean().values)
log("PC marker expression PB/BM: median %.2f (range %.2f-%.2f)" % (np.median(pc_ratio), pc_ratio.min(), pc_ratio.max()))

# ---- 3. This cohort: PB vs BM PCs of the same individual and stage (Supplementary Fig. 9D) ----------------------------
u = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
u["patient_id"] = u["patient_id"].astype(str)
for c_ in ("sample_type", "annotation", "disease_status"):
    u[c_] = u[c_].astype(str)
# IMWG 2/20/20 risk groups (risk.py); units of one individual that then share stage, tissue and PC status are pooled,
# weighted by their numbers of cells (cell table)
cl = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[["redcap_id", "sample_type", "annotation",
                                                                                   "disease_status"]].astype(str)
cm = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), usecols=["redcap_id", "patient_id"]).dropna()
cl["patient_id"] = cl["redcap_id"].map(cm.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"])
ncell = cl.groupby(["patient_id", "sample_type", "annotation", "disease_status"]).size().rename("n_cells").reset_index()
del cl
u = u.merge(ncell, on=["patient_id", "sample_type", "annotation", "disease_status"], how="left")
u["stage"] = stages(u["patient_id"], u["sample_type"], u["disease_status"])
u = pool_units(u, ["patient_id", "sample_type", "stage", "annotation"], "n_cells", [SIG])
u = u.dropna(subset=[SIG])
rows = []
for cells in ("tumor", "normal"):
    x = u[u.annotation == cells].pivot_table(index=["patient_id", "stage"], columns="sample_type", values=SIG).dropna()
    w = ss.wilcoxon(x.PB, x.BM)
    rows.append(dict(comparison="PB vs BM, %s PCs" % ("malignant" if cells == "tumor" else "normal"), n_pairs=len(x),
                     n_individuals=x.index.get_level_values(0).nunique(), first_higher=int((x.PB > x.BM).sum()),
                     median_first=x.PB.median(), median_second=x.BM.median(), p=w.pvalue))
x = u[u.sample_type == "PB"].pivot_table(index=["patient_id", "stage"], columns="annotation", values=SIG).dropna()
w = ss.wilcoxon(x.tumor, x.normal)
rows.append(dict(comparison="PB malignant vs PB normal PCs", n_pairs=len(x), n_individuals=x.index.get_level_values(0).nunique(),
                 first_higher=int((x.tumor > x.normal).sum()), median_first=x.tumor.median(), median_second=x.normal.median(), p=w.pvalue))
x = u[u.sample_type == "BM"].pivot_table(index=["patient_id", "stage"], columns="annotation", values=SIG).dropna()
w = ss.wilcoxon(x.tumor, x.normal)
rows.append(dict(comparison="BM malignant vs BM normal PCs", n_pairs=len(x), n_individuals=x.index.get_level_values(0).nunique(),
                 first_higher=int((x.tumor > x.normal).sum()), median_first=x.tumor.median(), median_second=x.normal.median(), p=w.pvalue))
coh = pd.DataFrame(rows)
coh.to_csv(OUT / "circulatory_paired_samples.tsv", sep="\t", index=False)
for r in rows:
    test("cohort, two-sided Wilcoxon signed-rank", r["comparison"], r["n_pairs"], "pairs with first higher", r["first_higher"], r["p"])
log(coh.to_string())
pd.DataFrame(TESTS).to_csv(OUT / "circulatory_signature_tests.tsv", sep="\t", index=False)
FH.close()
