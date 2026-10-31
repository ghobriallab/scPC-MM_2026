#!/usr/bin/env python3
"""Frequency of GPRC5D up- and downregulation in malignant versus normal PCs of the same patient, overall, by disease stage
and in t(11;14) versus other tumors, and of CD19 downregulation (Results; Discussion; Methods, Differential expression
analysis, single-cell level).

Purpose:      Counts tumors with GPRC5D up- or downregulated in malignant versus normal PCs of the same patient, with the
              rule of Fig. 2A (two-sided Wilcoxon test, Benjamini-Hochberg q < 0.01 and |log2 fold change| > 1; one
              tumor = the main malignant clone of a patient at one disease stage, BM), overall, by disease stage and for
              t(11;14) versus all other tumors. Tests t(11;14) versus other tumors with a two-sided Fisher's exact test,
              and t(11;14) tumors alone with a one-sided binomial test (upregulated in fewer than half). Counts tumors
              with CD19 downregulated by the same rule.

Inputs:       results/differential_expression/BM_scRNA_tumor_DE_FDR.csv, BM_scRNA_tumor_DE_logFoldChange.csv: rows = genes, columns
              = <patient>_BM_<stage>_tumor_<clone> (clone 0 = main clone), within-patient malignant vs normal PCs.
              data/paper/cytogenetics-status.csv (Supplementary Table 2): patient_id, cytogenetics (one call per patient).

Outputs:      results/differential_expression/gprc5d_counts.tsv: group, tumors, patients, up, down, unchanged, % up.
              results/differential_expression/gprc5d_tests.tsv: Fisher's exact test t(11;14) vs other tumors; binomial
              test of t(11;14) tumors.
              results/differential_expression/gprc5d_frequency_log.txt: counts and tests printed by the script.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir

import pandas as pd
import scipy.stats as ss

OUT = result_dir("differential_expression")
FH = open(OUT / "gprc5d_frequency_log.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


# within-patient DE of malignant vs normal PCs (Fig. 2A); main clone of each tumor
fdr = pd.read_csv(result("differential_expression", "BM_scRNA_tumor_DE_FDR.csv"), index_col=0)
lfc = pd.read_csv(result("differential_expression", "BM_scRNA_tumor_DE_logFoldChange.csv"), index_col=0)
main = [c for c in fdr.columns if c.endswith("_0")]
t = pd.DataFrame({"patient": [c.split("_")[0] for c in main], "stage": [c.split("_")[2] for c in main]}, index=main)
t["up"] = (fdr.loc["GPRC5D", main] < 0.01) & (lfc.loc["GPRC5D", main] > 1)
t["down"] = (fdr.loc["GPRC5D", main] < 0.01) & (lfc.loc["GPRC5D", main] < -1)

# one cytogenetic call per patient (Supplementary Table 2)
cy = pd.read_csv(data("paper", "cytogenetics-status.csv"))
t["t1114"] = t["patient"].map(cy.drop_duplicates("patient_id").set_index("patient_id")["cytogenetics"]) == "t(11;14)"
log("tumors (main clone, patient x stage): %d from %d patients" % (len(t), t.patient.nunique()))
# CD19, downregulated in most tumors (Results; same rule)
cd19_down = (fdr.loc["CD19", main] < 0.01) & (lfc.loc["CD19", main] < -1)
log("CD19 downregulated: %d of %d tumors (%.0f%%)" % (cd19_down.sum(), len(main), 100 * cd19_down.mean()))


def count(name, x):
    return dict(group=name, tumors=len(x), patients=x.patient.nunique(), up=int(x.up.sum()), down=int(x.down.sum()),
                unchanged=int((~x.up & ~x.down).sum()), pct_up=round(100 * x.up.mean(), 1))


rows = [count("all", t)]
rows += [count(s, t[t.stage == s]) for s in ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]]
rows += [count("t(11;14)", t[t.t1114]), count("other", t[~t.t1114])]
counts = pd.DataFrame(rows)
counts.to_csv(OUT / "gprc5d_counts.tsv", sep="\t", index=False)
log(counts.to_string(index=False))

a, b = t[t.t1114], t[~t.t1114]
odds, p_fisher = ss.fisher_exact([[a.up.sum(), (~a.up).sum()], [b.up.sum(), (~b.up).sum()]])
binom = ss.binomtest(int(a.up.sum()), len(a), p=0.5, alternative="less")
tests = pd.DataFrame([
    dict(test="two-sided Fisher's exact test, GPRC5D up: t(11;14) vs other tumors", n="%d vs %d" % (len(a), len(b)),
         estimate="odds ratio %.2f" % odds, p=p_fisher),
    dict(test="one-sided binomial test, GPRC5D up in fewer than half of t(11;14) tumors",
         n=str(len(a)), estimate="%d/%d" % (a.up.sum(), len(a)), p=binom.pvalue)])
tests.to_csv(OUT / "gprc5d_tests.tsv", sep="\t", index=False)
log(tests.to_string(index=False))
FH.close()
