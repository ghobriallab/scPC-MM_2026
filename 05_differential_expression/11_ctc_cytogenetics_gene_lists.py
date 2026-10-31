#!/usr/bin/env python3
"""Pseudobulk DEG tables of malignant BM PCs vs circulating tumor cells (paired BM and PB) and of malignant vs normal
PCs within each cytogenetic group (Methods, Detection of single-cell transcriptional signatures; Differential expression
analysis).

Purpose:      Reads the DESeq2 results of malignant PCs between paired BM and PB samples and of malignant vs normal BM
              PCs within each cytogenetic group (all patients; patients with both pseudobulks; t(11;14) patients by
              stage). A gene is differentially expressed (DEG) when the adjusted P (Benjamini-Hochberg) < 0.01 and
              |log2 fold change| > 1; adjusted P of NA counts as 1; immunoglobulin genes (IG[HKL]) and genes named
              <name>.<number> are removed. Writes tables of log2 fold changes (0 when not a DEG; cytogenetic tables
              always list 16 known therapeutic targets) and the BM vs PB DEG list (candidate genes of signature
              detection). Must be run with PYTHONHASHSEED=0 (gene order).

Inputs:       data/pseudobulk/result/GEX5_only/BM_vs_PB_tumor/BM_tumor_pair_n77_vs_PB_tumor_pair_n53.res.csv and
              data/pseudobulk/result/GEX5_only/cytogenetics/*.res.csv: DESeq2 results (baseMean, log2FoldChange, lfcSE,
              stat, padj) of the comparisons named below.

Outputs:      data/pseudobulk/summary/bm_vs_ctc.csv: BM vs PB log2 fold changes of the DEGs.
              data/signature/gene_list/tumor_de/bm_vs_ctc_DE_605genes.txt: BM vs PB DEGs, one gene per line.
              data/paper/degs/pseudobulk/cytogenetics-tumor-all-degs.csv, cytogenetics-tumor-matched-degs.csv: genes x
              cytogenetic groups; t1114-tumor-matched-mgus-smm-mm-degs.csv, t1114-tumor-matched-20-20-risk-degs.csv:
              t(11;14) patients with both pseudobulks, genes x stages.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd

np.random.seed(0)

GEX5 = data("pseudobulk", "result", "GEX5_only")
CYTO = GEX5 / "cytogenetics"
SUMMARY = data("pseudobulk", "summary")
PAPER = data("paper", "degs", "pseudobulk")
GENE_LIST = data("signature", "gene_list")
for d in [SUMMARY, PAPER, GENE_LIST / "tumor_de"]:
    d.mkdir(parents=True, exist_ok=True)

# known therapeutic targets, always listed in the cytogenetic tables
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


def get_sig_genes(df, fdr=0.01, log2fc=1):
    """DEGs: adjusted P < fdr and |log2 fold change| > log2fc."""
    return df[(df['padj'] < fdr) & (np.abs(df['log2FoldChange']) > log2fc)].index


def de_table(results, marker_genes=None, fdr=0.01, log2fc=1, df_use_key=False):
    """log2 fold changes of the DEGs of several DESeq2 results, genes x comparisons, 0 when not a DEG. Columns are named
    after the case group (or the key with df_use_key); marker genes are always listed."""
    de_genes = set()
    table = pd.DataFrame()
    for key in results.keys():
        case = os.path.basename(results[key]).replace('.res.csv', '').split('_vs_')[0]
        name = key if df_use_key else '_n'.join(case.split('_n')[:-1])     # case group without its sample number
        df = clean_deseq2_matrix(pd.read_csv(results[key], index_col=0))
        genes = list(get_sig_genes(df, fdr, log2fc))
        if marker_genes is not None:
            nonsig_markers = list(set(marker_genes).difference(set(genes)))
            genes.extend(marker_genes)
            df.loc[df.index.isin(nonsig_markers), 'log2FoldChange'] = 0
        df.loc[~df.index.isin(genes), 'log2FoldChange'] = 0
        df.rename(columns={'log2FoldChange': name}, inplace=True)
        table = df[[name]] if table.empty else table.join(df[[name]], how='outer')
        de_genes = de_genes.union(set(genes))
    table = table.loc[list(de_genes)]
    table[table.isna()] = 0
    return table


def write_list(array, outfile):
    with open(outfile, 'w') as fo:
        for elem in array:
            fo.write(elem + '\n')


# 1. malignant PCs, paired BM vs PB samples
bm_vs_pb_tumor = de_table({'BM_vs_CTC': GEX5 / 'BM_vs_PB_tumor' / 'BM_tumor_pair_n77_vs_PB_tumor_pair_n53.res.csv'})
bm_vs_pb_tumor.to_csv(SUMMARY / 'bm_vs_ctc.csv')
write_list(list(bm_vs_pb_tumor.index), GENE_LIST / 'tumor_de' / 'bm_vs_ctc_DE_605genes.txt')

# 2. malignant vs normal BM PCs within each cytogenetic group
cyto_res = {
    'HRD': CYTO / 'BM_HRD_tumor_n74_vs_BM_HRD_normal_n60.res.csv',
    't(4;14)': CYTO / 'BM_t0414_tumor_n15_vs_BM_t0414_normal_n12.res.csv',
    't(6;14)': CYTO / 'BM_t0614_tumor_n4_vs_BM_t0614_normal_n3.res.csv',
    't(8;14)': CYTO / 'BM_t0814_tumor_n4_vs_BM_t0814_normal_n2.res.csv',
    't(11;14)': CYTO / 'BM_t1114_tumor_n47_vs_BM_t1114_normal_n41.res.csv',
    't(12;14)': CYTO / 'BM_t1214_tumor_n2_vs_BM_t1214_normal_n2.res.csv',
    't(14;16)': CYTO / 'BM_t1416_tumor_n14_vs_BM_t1416_normal_n11.res.csv',
    't(14;20)': CYTO / 'BM_t1420_tumor_n7_vs_BM_t1420_normal_n6.res.csv',
    'Other': CYTO / 'BM_Other_tumor_n13_vs_BM_Other_normal_n11.res.csv'}
cyto_match_res = {
    'HRD': CYTO / 'BM_HRD_tumor_mp_n60_vs_BM_HRD_normal_mp_n60.res.csv',
    't(4;14)': CYTO / 'BM_t0414_tumor_mp_n12_vs_BM_t0414_normal_mp_n12.res.csv',
    't(6;14)': CYTO / 'BM_t0614_tumor_mp_n3_vs_BM_t0614_normal_mp_n3.res.csv',
    't(8;14)': CYTO / 'BM_t0814_tumor_mp_n2_vs_BM_t0814_normal_mp_n2.res.csv',
    't(11;14)': CYTO / 'BM_t1114_tumor_mp_n42_vs_BM_t1114_normal_mp_n41.res.csv',
    't(12;14)': CYTO / 'BM_t1214_tumor_mp_n2_vs_BM_t1214_normal_mp_n2.res.csv',
    't(14;16)': CYTO / 'BM_t1416_tumor_mp_n11_vs_BM_t1416_normal_mp_n11.res.csv',
    't(14;20)': CYTO / 'BM_t1420_tumor_mp_n6_vs_BM_t1420_normal_mp_n6.res.csv',
    'Other': CYTO / 'BM_Other_tumor_mp_n11_vs_BM_Other_normal_mp_n11.res.csv'}

# 3. t(11;14) patients with both pseudobulks, by risk group and by MGUS/SMM/MM
t1114_match_res = {
    'MGUS': CYTO / 'BM_t1114_MGUS_tumor_mp_n8_vs_BM_t1114_MGUS_normal_mp_n7.res.csv',
    'LRSMM': CYTO / 'BM_t1114_LRSMM_tumor_mp_n7_vs_BM_t1114_LRSMM_normal_mp_n7.res.csv',
    'IRSMM': CYTO / 'BM_t1114_IRSMM_tumor_mp_n11_vs_BM_t1114_IRSMM_normal_mp_n11.res.csv',
    'HRSMM': CYTO / 'BM_t1114_HRSMM_tumor_mp_n6_vs_BM_t1114_HRSMM_normal_mp_n6.res.csv',
    'MM': CYTO / 'BM_t1114_MM_tumor_mp_n10_vs_BM_t1114_MM_normal_mp_n10.res.csv'}
t1114_sim_match_res = {
    'MGUS': CYTO / 'BM_t1114_MGUS_tumor_mp_n8_vs_BM_t1114_MGUS_normal_mp_n7.res.csv',
    'SMM': CYTO / 'BM_t1114_SMM_tumor_mp_n24_vs_BM_t1114_SMM_normal_mp_n24.res.csv',
    'MM': CYTO / 'BM_t1114_MM_tumor_mp_n10_vs_BM_t1114_MM_normal_mp_n10.res.csv'}

cyto_tumor_all = de_table(cyto_res, marker_genes=KNOWN_TARGETS)
cyto_tumor_mp = de_table(cyto_match_res, marker_genes=KNOWN_TARGETS)
t1114_tumor_mp = de_table(t1114_match_res, marker_genes=KNOWN_TARGETS)
t1114_sim_tumor_mp = de_table(t1114_sim_match_res, marker_genes=KNOWN_TARGETS)
cyto_tumor_all.to_csv(PAPER / 'cytogenetics-tumor-all-degs.csv')
cyto_tumor_mp.to_csv(PAPER / 'cytogenetics-tumor-matched-degs.csv')
t1114_sim_tumor_mp.to_csv(PAPER / 't1114-tumor-matched-mgus-smm-mm-degs.csv')
t1114_tumor_mp.to_csv(PAPER / 't1114-tumor-matched-20-20-risk-degs.csv')
