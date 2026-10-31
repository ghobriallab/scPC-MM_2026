#!/usr/bin/env python3
"""Matrices of adjusted P values and log2 fold changes of the within-patient single-cell differential expression
results, one column per comparison (Methods, Differential expression analysis, single-cell level; Fig. 2A).

Purpose:      Collects the scanpy rank_genes_groups tables of 13_scrna_de_tumor_vs_normal.py into one gene x comparison
              matrix of Benjamini-Hochberg adjusted P values (pvals_adj) and one of log2 fold changes (logfoldchanges):
              malignant vs normal BM PCs. Genes sorted by name (genes of the first table), columns sorted by comparison
              name (<redcap_id>_BM_<stage>_tumor_<clone>).

Inputs:       data/scrna_de/de_files/tumor_vs_normal/*_BM_*.csv: per-comparison tables (names, pvals_adj, logfoldchanges).

Outputs:      data/scrna_de/matrix/BM_scRNA_tumor_DE_FDR.csv and BM_scRNA_tumor_DE_logFoldChange.csv: genes x
              comparisons.

Dependencies: Python 3.8 with requirements.txt (pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import glob

import pandas as pd

DE_FILES = data("scrna_de", "de_files")
OUT = data("scrna_de", "matrix")
OUT.mkdir(parents=True, exist_ok=True)

# output prefix, folder of the per-comparison tables, file pattern
SETS = [('BM_scRNA_tumor_DE', 'tumor_vs_normal', '*_BM_*.csv')]


def create_matrices(input_files, output_prefix):
    """Adjusted P and log2 fold change of every gene in every comparison."""
    df = pd.read_csv(input_files[0], index_col=0)
    genes = sorted(df.index)
    samples = sorted([os.path.basename(x).replace(".wilcox.csv", "") for x in input_files])

    df_fdr = pd.DataFrame(index=genes, columns=samples)
    df_lfc = pd.DataFrame(index=genes, columns=samples)
    for input_file in input_files:
        sid = os.path.basename(input_file).replace(".wilcox.csv", "")
        df = pd.read_csv(input_file, index_col=0)
        df_fdr[sid] = df.loc[genes, 'pvals_adj']
        df_lfc[sid] = df.loc[genes, 'logfoldchanges']

    df_fdr.to_csv(OUT / (output_prefix + '_FDR.csv'))
    df_lfc.to_csv(OUT / (output_prefix + '_logFoldChange.csv'))


for prefix, folder, pattern in SETS:
    files = sorted(glob.glob(str(DE_FILES / folder / pattern)))
    if files:
        create_matrices(files, prefix)
