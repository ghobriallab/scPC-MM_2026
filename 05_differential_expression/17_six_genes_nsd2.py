#!/usr/bin/env python3
"""Genes consistently upregulated in malignant PCs at every disease stage, and their upregulation in NSD2 [t(4;14)] versus
other tumors (Results; Methods, Differential expression analysis, single-cell level; Supplementary Fig. 2).

Purpose:      1. Genes upregulated in malignant versus normal PCs of the same patient (q < 0.01 and log2 fold change > 1,
                 the rule of Fig. 2A; main clone, BM) in at least half of the tumors at every disease stage. No gene is
                 excluded from this rule, so an immunoglobulin gene that passes it is kept with the others.
              2. For each of these genes, the share of tumors with upregulation and the median log2 fold change, NSD2
                 [t(4;14)] tumors versus all other tumors: two-sided Fisher's exact test and two-sided Wilcoxon rank-sum
                 test, each with Benjamini-Hochberg adjustment over all genes selected in 1.
              3. Numbers of malignant and normal BM PCs per tumor, NSD2 versus other tumors (two-sided Wilcoxon rank-sum
                 tests), and NSD2 tumors by stage.

Inputs:       results/differential_expression/BM_scRNA_tumor_DE_FDR.csv, BM_scRNA_tumor_DE_logFoldChange.csv: rows = genes, columns
              = <patient>_BM_<stage>_tumor_<clone> (clone 0 = main clone), within-patient malignant vs normal PCs (Fig. 2A).
              data/paper/cytogenetics-status.csv: tumor_id, patient_id, cytogenetics (main tumor).
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, disease_status, sample_type, annotation (cell
              counts).
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id.

Outputs:      results/differential_expression/six_genes.tsv: per selected gene, share of tumors with upregulation at each
              stage and in all tumors.
              results/differential_expression/six_genes_nsd2_vs_other.tsv: per selected gene, upregulation and median log2
              fold change in NSD2 versus other tumors, P and Benjamini-Hochberg q values.
              results/differential_expression/six_genes_nsd2_log.txt: counts and tests printed by the script.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir
from risk import stages

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = result_dir("differential_expression")
FH = open(OUT / "six_genes_nsd2_log.txt", "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


fdr = pd.read_csv(result("differential_expression", "BM_scRNA_tumor_DE_FDR.csv"), index_col=0)
lfc = pd.read_csv(result("differential_expression", "BM_scRNA_tumor_DE_logFoldChange.csv"), index_col=0)
main = [c for c in fdr.columns if c.endswith("_0")]
fdr, lfc = fdr[main], lfc.loc[fdr.index, main]
t = pd.DataFrame({"patient_id": [c.split("_")[0] for c in main], "stage": [c.split("_")[2] for c in main]}, index=main)
cy = pd.read_csv(data("paper", "cytogenetics-status.csv"))
cy = cy[cy["tumor_id"].str.endswith(":tumor:0")].set_index("patient_id")
t["nsd2"] = t["patient_id"].map(cy["cytogenetics"]) == "t(4;14)"
up = (fdr < 0.01) & (lfc > 1)
log("tumors: %d (NSD2 [t(4;14)] %d)" % (len(t), t["nsd2"].sum()))

# ---- 1. the six genes --------------------------------------------------------------------------------------------------
frac = pd.DataFrame({s: up.loc[:, t.index[t["stage"] == s]].mean(axis=1) for s in STAGES})
six = frac.index[(frac >= 0.5).all(axis=1)]
log("genes upregulated in >= 50%% of tumors at every stage: %d: %s" % (len(six), ", ".join(six)))
for g in ["MLL3", "KMT2C", "MLLT3"]:
    log("  %s in the DE table: %s%s" % (g, g in up.index, "; up in %d of %d tumors" % (up.loc[g].sum(), len(t)) if g in up.index else ""))
frac.loc[six].assign(all_tumors=up.loc[six].mean(axis=1)).to_csv(OUT / "six_genes.tsv", sep="\t")

# ---- 2. NSD2 vs other tumors -------------------------------------------------------------------------------------------
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells.loc[cells["sample_type"].astype(str) == "BM", ["redcap_id", "disease_status", "annotation"]].astype(str)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), dtype=str)
cells["patient_id"] = cells["redcap_id"].map(meta[["redcap_id", "patient_id"]].dropna().drop_duplicates()
                                              .set_index("redcap_id")["patient_id"])
cells["disease_status"] = stages(cells["patient_id"], ["BM"] * len(cells), cells["disease_status"])   # IMWG risk groups
n = cells.groupby(["patient_id", "disease_status", "annotation"]).size().unstack(fill_value=0)
t["malignant_PCs"] = [n.loc[(p, s), "tumor"] for p, s in zip(t["patient_id"], t["stage"])]
t["normal_PCs"] = [n.loc[(p, s), "normal"] for p, s in zip(t["patient_id"], t["stage"])]
a, b = t[t["nsd2"]], t[~t["nsd2"]]
log("malignant PCs per tumor, median: NSD2 %d, other %d (P = %.2g); normal PCs: NSD2 %d, other %d (P = %.2g)" % (
    a["malignant_PCs"].median(), b["malignant_PCs"].median(), stats.mannwhitneyu(a["malignant_PCs"], b["malignant_PCs"]).pvalue,
    a["normal_PCs"].median(), b["normal_PCs"].median(), stats.mannwhitneyu(a["normal_PCs"], b["normal_PCs"]).pvalue))
log("NSD2 tumors by stage:", a["stage"].value_counts().reindex(STAGES).to_dict())
rows = []
for g in six:
    u = up.loc[g]
    odds, p_f = stats.fisher_exact([[u[a.index].sum(), (~u[a.index]).sum()], [u[b.index].sum(), (~u[b.index]).sum()]])
    p_w = stats.mannwhitneyu(lfc.loc[g, a.index].dropna(), lfc.loc[g, b.index].dropna()).pvalue
    rows.append({"gene": g, "up_nsd2": int(u[a.index].sum()), "n_nsd2": len(a), "pct_up_nsd2": 100 * u[a.index].mean(),
                 "up_other": int(u[b.index].sum()), "n_other": len(b), "pct_up_other": 100 * u[b.index].mean(),
                 "fisher_P": p_f, "median_lfc_nsd2": lfc.loc[g, a.index].median(), "median_lfc_other": lfc.loc[g, b.index].median(),
                 "ranksum_P": p_w})
res = pd.DataFrame(rows)
res["fisher_q"] = multipletests(res["fisher_P"], method="fdr_bh")[1]
res["ranksum_q"] = multipletests(res["ranksum_P"], method="fdr_bh")[1]
res.to_csv(OUT / "six_genes_nsd2_vs_other.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
log(res.round(3).to_string(index=False))
FH.close()
