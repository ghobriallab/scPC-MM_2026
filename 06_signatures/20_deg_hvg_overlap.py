#!/usr/bin/env python3
"""Overlap of the 4,980 genes used for signature discovery with the 5,000 most highly variable genes, and the number of
genes tested in the pseudobulk comparisons (Methods, detection of single-cell transcriptional signatures).

Purpose:      Counts the features of the expression matrix and the genes tested in at least one pseudobulk DESeq2
              comparison (non-missing adjusted P value in any result table, i.e. genes expressed enough to be tested);
              checks that the 4,980-gene pool is the union of three DEG lists and counts the genes of each; tests the
              overlap of the 4,980 DEGs, and of the 1,463 signature genes, with the top 5,000 highly variable genes
              (scanpy highly_variable_genes, n_top_genes = 5000, all 1,649,290 PCs) by a one-sided hypergeometric test,
              in two universes: all 39,390 features, and the genes tested by DESeq2 (both gene sets restricted to the
              universe).

Inputs:       data/paper/plasma-cell-top-5000hvg.csv: index = gene (the 39,390 features of the matrix), highly_variable.
              data/signature/hotspot/mm-BM-PB-4980-DEs_module_assignment.csv: index = the 4,980-gene pool.
              data/signature/gene_list/tumor_de/bm_tumor_unionDE_2657genes.txt, tumor_de/bm_vs_ctc_DE_605genes.txt and
              normal_de/npc_core_bm_normal_unionDE_3744genes.txt: the gene lists whose union is the pool.
              data/signature/hotspot/consensus-modules_genes.csv: index = signature gene, context = signature.
              data/pseudobulk/result/**/*.res.csv: DESeq2 results (padj).

Outputs:      results/signatures/deg_hvg_overlap.tsv: universe, gene set, overlap, expected, fold, one-sided P.
              results/signatures/deg_hvg_overlap_log.txt: gene counts and overlaps.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import glob

import pandas as pd
from scipy.stats import hypergeom

OUT = result_dir("signatures")
FH = open(os.path.join(OUT, "deg_hvg_overlap_log.txt"), "w")
GL = data("signature", "gene_list")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


def genes(path):
    return {g.strip() for g in open(path) if g.strip()}


hv = pd.read_csv(data("paper", "plasma-cell-top-5000hvg.csv"), index_col=0)
FEAT = set(hv.index.astype(str))
HVG = set(hv.index[hv["highly_variable"].astype(bool)].astype(str))
POOL = set(pd.read_csv(data("signature", "hotspot", "mm-BM-PB-4980-DEs_module_assignment.csv"), index_col=0).index.astype(str))
SIG = set(pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0).index.astype(str))
parts = {"malignant PCs: vs normal PCs by stage and between stages (BM)": genes(GL / "tumor_de" / "bm_tumor_unionDE_2657genes.txt"),
         "normal PCs: between stages, vs donors, BM vs PB": genes(GL / "normal_de" / "npc_core_bm_normal_unionDE_3744genes.txt"),
         "malignant PCs: paired BM vs PB": genes(GL / "tumor_de" / "bm_vs_ctc_DE_605genes.txt")}
assert set().union(*parts.values()) == POOL
log("features in the expression matrix: %d; top HVGs: %d; DEG pool: %d; signature genes: %d" % (len(FEAT), len(HVG), len(POOL), len(SIG)))
for k, v in parts.items():
    log("  %-62s %d genes" % (k, len(v)))

files = sorted(glob.glob(str(data("pseudobulk", "result", "**", "*.res.csv")), recursive=True))
tested = set()
for f in files:
    d = pd.read_csv(f, index_col=0)
    tested |= set(d.index[d["padj"].notna()].astype(str))
log("DESeq2 result tables: %d; genes tested in at least one (non-missing padj): %d; pool genes among them: %d" % (
    len(files), len(tested), len(POOL & tested)))

rows = []
for uname, U in [("all 39,390 features", FEAT), ("genes tested by DESeq2", tested & FEAT)]:
    for sname, S in [("4,980 DEGs", POOL), ("1,463 signature genes", SIG)]:
        N, K, n = len(U), len(HVG & U), len(S & U)
        k = len(HVG & S & U)
        exp = n * K / N
        rows.append({"universe": uname, "gene_set": sname, "universe_genes": N, "HVGs_in_universe": K, "set_in_universe": n,
                     "overlap": k, "pct_of_set": 100 * k / n, "expected": exp, "fold": k / exp,
                     "P_enrichment": hypergeom.sf(k - 1, N, K, n)})
        log("%-38s %-22s overlap %d of %d (%.1f%%), expected %.0f, %.2f-fold, P = %.2g" % (
            uname, sname, k, n, 100 * k / n, exp, k / exp, rows[-1]["P_enrichment"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "deg_hvg_overlap.tsv"), sep="\t", index=False)
log("HVGs not in the DEG pool: %d of %d; of those, tested by DESeq2: %d" % (len(HVG - POOL), len(HVG), len((HVG - POOL) & tested)))
FH.close()
