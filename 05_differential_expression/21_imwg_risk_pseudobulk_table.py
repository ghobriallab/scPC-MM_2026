#!/usr/bin/env python3
"""Log2 fold changes of malignant vs normal BM PCs by disease stage, SMM split by the IMWG 2/20/20 risk group (Fig. 2B).

Purpose:      Reads the DESeq2 results of 20_deseq2_imwg_risk_groups.R (patients with both pseudobulks at each stage)
              and builds the genes x stages table with the rule of 09_tumor_pseudobulk_gene_lists.py: a gene is
              differentially expressed when the adjusted P < 0.01 and |log2 fold change| > 1 (adjusted P of NA counts
              as 1); immunoglobulin genes (IG[HKL]) and genes named <name>.<number> are removed; the 16 therapeutic
              targets are always listed; log2 fold changes that are not significant are set to 0. Must be run with
              PYTHONHASHSEED=0 (row order).

Inputs:       results/differential_expression/pseudobulk_imwg_risk/BM_<stage>_tumor_mp_n<N>_vs_BM_<stage>_normal_mp_n<M>.res.csv.

Outputs:      results/differential_expression/disease-tumor-matched-imwg-risk-degs.csv: genes x stages (BM_<stage>_tumor_mp).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import result_dir

import numpy as np
import pandas as pd

OUT = result_dir("differential_expression")
RES = OUT / "pseudobulk_imwg_risk"
KNOWN_TARGETS = ['TNFRSF17', 'GPRC5D', 'SDC1', 'CD38', 'SLAMF7', 'FCRL5', 'CD46', 'CD47', 'CD48',
                 'NCAM1', 'ITGB7', 'CD74', 'LY9', 'TNFRSF13B', 'CD19', 'BCL2']


def clean_deseq2_matrix(df):
    """Adjusted P of NA set to 1; immunoglobulin genes and genes named <name>.<number> removed."""
    df.loc[df['padj'].isna(), 'padj'] = 1
    df = df[~df.index.str.match('^IG[HKL]')]
    valid_genes = []
    for gene in df.index:
        try:
            int(gene.split('.')[-1])
        except ValueError:
            valid_genes.append(gene)
    return df.loc[valid_genes, ['baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'padj']]


def de_table(results, marker_genes, fdr=0.01, log2fc=1):
    """log2 fold changes of the DEGs of several DESeq2 results, genes x comparisons, 0 when not a DEG; marker genes
    always listed."""
    de_genes = set()
    table = pd.DataFrame()
    for name, path in results.items():
        df = clean_deseq2_matrix(pd.read_csv(path, index_col=0))
        genes = list(df[(df['padj'] < fdr) & (np.abs(df['log2FoldChange']) > log2fc)].index)
        nonsig_markers = list(set(marker_genes).difference(set(genes)))
        genes.extend(marker_genes)
        df.loc[df.index.isin(nonsig_markers), 'log2FoldChange'] = 0
        df.loc[~df.index.isin(genes), 'log2FoldChange'] = 0
        df = df.rename(columns={'log2FoldChange': name})
        table = df[[name]] if table.empty else table.join(df[[name]], how='outer')
        de_genes = de_genes.union(set(genes))
    table = table.loc[list(de_genes)]
    table[table.isna()] = 0
    return table


results = {}
for stage in ['MM', 'HRSMM', 'IRSMM', 'LRSMM', 'MGUS']:
    f = glob.glob(str(RES / ('BM_%s_tumor_mp_n*_vs_BM_%s_normal_mp_n*.res.csv' % (stage, stage))))
    assert len(f) == 1, (stage, f)
    results['BM_%s_tumor_mp' % stage] = f[0]
    print(os.path.basename(f[0]))
de_table(results, KNOWN_TARGETS).to_csv(OUT / 'disease-tumor-matched-imwg-risk-degs.csv')
