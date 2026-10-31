#!/usr/bin/env python3
"""MM-like cells in every BM tumor scored out of sample, across disease stages and across the Numbat CNV clones of each
tumor (Fig. 6; Results).

Purpose:      Out-of-sample MST scores for every tumor: the MGUS and MM patients used to build the MGUS-MM classifier are
              scored by 5-fold cross-validation or, for the test set, by the model fit on the training patients
              (08_classifiers/28_misclassified_mgus_cells.py; the classifier's downsampled cells); every other tumor
              (all SMM and the MGUS and MM tumors not used for training) by the final model (08_mm_like_cells.py). A tumor
              is the malignant BM PCs of one patient at one disease stage (IMWG 2/20/20 risk groups, risk.py) with at
              least 30 cells; a cell is MM-like when its MST score is above 0.5.
              1. Percentage of MM-like cells by disease stage: two-sided Spearman correlation with stage coded 0-4
              (MGUS to MM) and two-sided Wilcoxon rank-sum tests between consecutive stages, Benjamini-Hochberg-adjusted.
              2. Numbat CNV clones (clone 1, the diploid clone, left out) with at least 50 cells, in tumors with at least
              two such clones: percentage of MM-like cells per clone; per tumor, the difference between the most and the
              least MM-like clone and its permutation P (the tumor's MM-like cells reassigned at random to clones of the
              same sizes, 10,000 permutations, seed 42), Benjamini-Hochberg-adjusted across tumors; Spearman correlation
              between the two largest clones; share of the variance of clone percentages that lies within tumors.
              3. MGUS tumors: classifier patients (out of fold) against MGUS tumors not used in training, and MAF-family
              translocations against the other cytogenetic groups among the classifier patients (two-sided Wilcoxon
              rank-sum tests); MST scores of the MM-like cells of the MGUS classifier patients.

Inputs:       results/stage_and_progression/mm_like_cells_tumors.tsv, mm_like_cells_clones.tsv (08_mm_like_cells.py);
              results/classifiers/misclassified_mgus_cells.pkl.gz (patient, disease_status, split, mst, clone);
              results/classifiers/misclassified_mgus_patients.tsv (patient, pct_mm_like, cytogenetics).

Outputs:      results/stage_and_progression/mm_like_oos_tumors.tsv: per tumor: stage, source of the scores, cells, MST
              percentiles, fraction of MM-like cells.
              results/stage_and_progression/mm_like_oos_clones.tsv: per tumor and clone: cells, fraction of MM-like cells,
              with the tumor's clone difference, permutation P and q.
              results/stage_and_progression/mm_like_oos_tests.tsv: stage and clone tests.
              results/stage_and_progression/mm_like_oos_log.txt: run log.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import result, result_dir

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "mm_like_oos_log.txt"), "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
MIN_CELLS, MIN_CLONE, N_PERM = 30, 50, 10000
TESTS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def add_test(name, detail, n, stat, p, q=np.nan):
    TESTS.append(dict(test=name, detail=detail, n=n, statistic=stat, P=p, q=q))
    log("  TEST %s | %s | n=%d | stat=%.3f | P=%.3g%s" % (name, detail, n, stat, p, "" if np.isnan(q) else " | q=%.3g" % q))


# ---- 1. out-of-sample scores per tumor ---------------------------------------------------------------
final = pd.read_csv(result("stage_and_progression", "mm_like_cells_tumors.tsv"), sep="\t")
final_cl = pd.read_csv(result("stage_and_progression", "mm_like_cells_clones.tsv"), sep="\t", dtype={"clone": str})
oof = pd.read_pickle(result("classifiers", "misclassified_mgus_cells.pkl.gz"))
oof["mm_like"] = oof["mst"] > 0.5
trained = oof[["patient", "disease_status"]].drop_duplicates().rename(columns={"patient": "patient_id"})
log("classifier tumors (cross-validation or test-set scores): %d (%s)" % (len(trained), trained["disease_status"].value_counts().to_dict()))

a = oof.groupby(["patient", "disease_status"]).agg(cells=("mst", "size"), mean_mst=("mst", "mean"), frac_mm_like=("mm_like", "mean"))
qa = oof.groupby(["patient", "disease_status"])["mst"].quantile([0.1, 0.5, 0.9]).unstack()
qa.columns = ["q10_mst", "median_mst", "q90_mst"]
a = a.join(qa).reset_index().rename(columns={"patient": "patient_id"}).assign(source="classifier, out of fold")
key = final.set_index(["patient_id", "disease_status"]).index
b = final[~key.isin(trained.set_index(["patient_id", "disease_status"]).index)].assign(source="final model, not used in training")
T = pd.concat([a, b], ignore_index=True)
T = T[(T["cells"] >= MIN_CELLS) & T["disease_status"].isin(STAGES)].copy()
T["stage"] = T["disease_status"].map(STAGES.index)
T = T.sort_values(["stage", "median_mst"]).reset_index(drop=True)
log("tumors: %d (%d patients); by stage %s; by source %s" % (len(T), T["patient_id"].nunique(),
    T["disease_status"].value_counts().reindex(STAGES).to_dict(), T["source"].value_counts().to_dict()))
log("percentage of MM-like cells by stage (median, IQR):")
for s, g in T.groupby("stage"):
    log("  %-6s n=%3d  %.1f (%.1f-%.1f)" % (STAGES[s], len(g), 100 * g["frac_mm_like"].median(), 100 * g["frac_mm_like"].quantile(.25),
                                           100 * g["frac_mm_like"].quantile(.75)))
mixed = T["frac_mm_like"].between(0.1, 0.9)
log("tumors with MGUS-like and MM-like cells each >= 10%%: %d of %d (%s)" % (mixed.sum(), len(T),
    T[mixed]["disease_status"].value_counts().reindex(STAGES).to_dict()))

log("\nstage tests")
r, p = spearmanr(T["stage"], T["frac_mm_like"])
add_test("Spearman", "fraction of MM-like cells vs disease stage (0-4)", len(T), r, p)
pairs = list(zip(STAGES[:-1], STAGES[1:]))
ps = [mannwhitneyu(T.loc[T["disease_status"] == x, "frac_mm_like"], T.loc[T["disease_status"] == y, "frac_mm_like"],
                   alternative="two-sided").pvalue for x, y in pairs]
qs = multipletests(ps, method="fdr_bh")[1]
for (x, y), p, q in zip(pairs, ps, qs):
    add_test("Wilcoxon rank-sum", "fraction of MM-like cells, %s vs %s" % (x, y),
             int(T["disease_status"].isin([x, y]).sum()), np.nan, p, q)

# ---- 2. Numbat CNV clones -------------------------------------------------------------------------
oc = oof[~oof["clone"].isin(["none", "1"])].groupby(["patient", "disease_status", "clone"]).agg(
    cells=("mm_like", "size"), mm_like=("mm_like", "sum")).reset_index().rename(columns={"patient": "patient_id"})
fc = final_cl.merge(b[["patient_id", "disease_status"]], on=["patient_id", "disease_status"])
C = pd.concat([oc, fc], ignore_index=True)
C = C.merge(T[["patient_id", "disease_status", "frac_mm_like"]].rename(columns={"frac_mm_like": "tumor_frac_mm_like"}),
            on=["patient_id", "disease_status"])
C = C[C["cells"] >= MIN_CLONE]
C = C[C.groupby(["patient_id", "disease_status"])["clone"].transform("size") >= 2].copy()
C["frac_mm_like"] = C["mm_like"] / C["cells"]
rng = np.random.default_rng(42)
rows = []
for (pt, st), g in C.groupby(["patient_id", "disease_status"], sort=False):
    sizes, k = g["cells"].values, int(g["mm_like"].sum())
    obs = (g["frac_mm_like"].max() - g["frac_mm_like"].min())
    draws = rng.multivariate_hypergeometric(sizes, k, size=N_PERM) / sizes     # MM-like cells reassigned at random
    nr = draws.max(axis=1) - draws.min(axis=1)
    rows.append(dict(patient_id=pt, disease_status=st, clones=len(g), cells_in_clones=int(sizes.sum()),
                     clone_range=obs, null_median_range=np.median(nr), perm_p=(1 + (nr >= obs - 1e-12).sum()) / (N_PERM + 1)))
R = pd.DataFrame(rows)
R["perm_q"] = multipletests(R["perm_p"], method="fdr_bh")[1]
C = C.merge(R, on=["patient_id", "disease_status"])
log("\nNumbat CNV clones: tumors with >= 2 clones of >= %d cells: %d (%s)" % (MIN_CLONE, len(R),
    R["disease_status"].value_counts().reindex(STAGES).fillna(0).astype(int).to_dict()))
log("most minus least MM-like clone: median %.1f points (IQR %.1f-%.1f); expected by chance (median of permutation medians) %.1f"
    % (100 * R["clone_range"].median(), 100 * R["clone_range"].quantile(.25), 100 * R["clone_range"].quantile(.75),
       100 * R["null_median_range"].median()))
log("tumors whose clones differ more than by chance (q < 0.05): %d of %d; among them, median difference %.1f points"
    % ((R["perm_q"] < 0.05).sum(), len(R), 100 * R.loc[R["perm_q"] < 0.05, "clone_range"].median()))
add_test("Permutation (per tumor, BH)", "clone difference larger than chance, tumors with q < 0.05", len(R),
         float((R["perm_q"] < 0.05).sum()), np.nan)
big = C.sort_values("cells", ascending=False).groupby(["patient_id", "disease_status"]).head(2)
two = big.groupby(["patient_id", "disease_status"])["frac_mm_like"].apply(list).apply(pd.Series)
r, p = spearmanr(two[0], two[1])
add_test("Spearman", "percentage of MM-like cells, largest vs second-largest clone", len(two), r, p)
tum = C.groupby(["patient_id", "disease_status"])["tumor_frac_mm_like"].first()
mixed_t = tum[(tum >= 0.1) & (tum <= 0.9)].index
every = C.set_index(["patient_id", "disease_status"]).loc[mixed_t].groupby(level=[0, 1])["frac_mm_like"].min()
log("tumors with 10-90%% MM-like cells (whole tumor): %d; with >= 10%% MM-like cells in every clone: %d" % (len(mixed_t), (every >= 0.1).sum()))
v_w = C.groupby(["patient_id", "disease_status"])["frac_mm_like"].var().mean()
log("variance of clone percentages: between tumors %.4f, within tumors %.4f (within: %.0f%%)" % (tum.var(), v_w,
    100 * v_w / (v_w + tum.var())))

# ---- 3. MGUS tumors ----------------------------------------------------------------------------------
log("\nMGUS tumors")
mg = T[T["disease_status"] == "MGUS"]
cl_, nt_ = mg[mg["source"].str.startswith("classifier")], mg[mg["source"].str.startswith("final")]
log("  %% MM-like cells, median: classifier patients (out of fold) %.1f (n=%d); not used in training %.1f (n=%d)" % (
    100 * cl_["frac_mm_like"].median(), len(cl_), 100 * nt_["frac_mm_like"].median(), len(nt_)))
u = mannwhitneyu(cl_["frac_mm_like"], nt_["frac_mm_like"], alternative="two-sided")
add_test("Wilcoxon rank-sum", "MGUS, fraction of MM-like cells, classifier patients vs not used in training", len(mg),
         u.statistic, u.pvalue)
pt = pd.read_csv(result("classifiers", "misclassified_mgus_patients.tsv"), sep="\t")
maf = pt["cytogenetics"].isin(["t(14;16)", "t(14;20)", "t(8;14)"])
u = mannwhitneyu(pt.loc[maf, "pct_mm_like"], pt.loc[~maf, "pct_mm_like"], alternative="two-sided")
add_test("Wilcoxon rank-sum", "MGUS classifier patients, %% MM-like cells, MAF family (n=%d) vs other groups" % maf.sum(),
         len(pt), u.statistic, u.pvalue)
mmc = oof[(oof["disease_status"] == "MGUS") & oof["mm_like"]]["mst"]
log("  MST score of the MM-like cells of MGUS classifier patients: median %.2f (IQR %.2f-%.2f); above 0.6: %.0f%%" % (
    mmc.median(), mmc.quantile(0.25), mmc.quantile(0.75), 100 * (mmc > 0.6).mean()))
add_test("Median", "MST score of the MM-like cells of MGUS classifier patients (out of fold)", len(mmc), mmc.median(), np.nan)

T.to_csv(os.path.join(OUT, "mm_like_oos_tumors.tsv"), sep="\t", index=False)
C.to_csv(os.path.join(OUT, "mm_like_oos_clones.tsv"), sep="\t", index=False)
pd.DataFrame(TESTS).to_csv(os.path.join(OUT, "mm_like_oos_tests.tsv"), sep="\t", index=False)
FH.close()
