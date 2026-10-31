#!/usr/bin/env python3
"""Malignant-PC gene lists from the pseudobulk DESeq2 comparisons by disease stage, including the candidate genes of
signature detection (Methods, Detection of single-cell transcriptional signatures; Differential expression analysis).

Purpose:      Reads the DESeq2 results of malignant vs normal PCs (all patients, SMM, MGUS/SMM/MM, risk groups; in BM
              also the patients with both pseudobulks) and of malignant PCs between stages, in BM and PB. A gene is
              differentially expressed (DEG) when the adjusted P (Benjamini-Hochberg) < 0.01 and |log2 fold change|
              > 1; adjusted P of NA counts as 1; immunoglobulin genes (IG[HKL]) and genes named <name>.<number> are
              removed. Tables of log2 fold changes per comparison (0 when not a DEG; BM malignant vs normal tables
              always list 16 known therapeutic targets). Gene lists per tissue: malignant vs normal DEGs at MGUS, SMM
              or MM united with the DEGs between these stages (BM: candidate genes of signature detection), their
              intersection, and the DEGs of all malignant vs normal PCs united with the stage DEGs; and the BM DEGs
              between stages. Must be run with PYTHONHASHSEED=0 (gene order of tables and lists).

Inputs:       data/pseudobulk/result/whole_de/tumor_vs_normal/*.res.csv, data/pseudobulk/result/whole_de/tumor_vs_tumor/
              *.res.csv and data/pseudobulk/result/GEX5_only/tumor_vs_normal/*_mp_*.res.csv: DESeq2 results (baseMean,
              log2FoldChange, lfcSE, stat, padj) of the comparisons named below.

Outputs:      data/paper/degs/pseudobulk/disease-tumor-mgus-smm-mm.csv, disease-tumor-20-20-risk-degs.csv,
              disease-tumor-matched-20-20-risk-degs.csv: BM malignant vs normal log2 fold changes, genes x stages.
              data/pseudobulk/summary/tvn_bm_group.csv, tvn_pb_group.csv, tvt_bm_group.csv, tvt_pb_group.csv: the same
              for MGUS/SMM/MM and for the pairs of these stages, BM and PB.
              data/signature/gene_list/{bm,pb,union}_tumor_interDE_*genes.txt, {pb,union}_tumor_unionDE_*genes.txt,
              {bm,pb,union}_tumor_DE_*genes.txt, data/signature/gene_list/tumor_de/bm_tumor_unionDE_2657genes.txt and
              data/signature/gene_list/tumor_de/bm_tumor_tvtDE_476genes.txt: gene lists, one gene per line.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd

np.random.seed(0)

TVN = data("pseudobulk", "result", "whole_de", "tumor_vs_normal")
MATCH = data("pseudobulk", "result", "GEX5_only", "tumor_vs_normal")
TVT = data("pseudobulk", "result", "whole_de", "tumor_vs_tumor")
SUMMARY = data("pseudobulk", "summary")
PAPER = data("paper", "degs", "pseudobulk")
GENE_LIST = data("signature", "gene_list")
for d in [SUMMARY, PAPER, GENE_LIST / "tumor_de"]:
    d.mkdir(parents=True, exist_ok=True)

# known therapeutic targets, always listed in the BM malignant vs normal tables
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


# 1. malignant vs normal PCs, BM
tvn_bm_tumor_res = {
    'BM:ALL': TVN / 'BM_tumor_n205_vs_BM_normal_n170.res.csv',
    'BM:SMM': TVN / 'BM_SMM_tumor_n131_vs_BM_SMM_normal_n107.res.csv'}
tvn_bm_group_res = {
    'BM:MM': TVN / 'BM_MM_tumor_n48_vs_BM_MM_normal_n28.res.csv',
    'BM:SMM': TVN / 'BM_SMM_tumor_n131_vs_BM_SMM_normal_n107.res.csv',
    'BM:MGUS': TVN / 'BM_MGUS_tumor_n26_vs_BM_MGUS_normal_n35.res.csv'}
tvn_bm_risk_res = {
    'BM:MM': TVN / 'BM_MM_tumor_n48_vs_BM_MM_normal_n28.res.csv',
    'BM:HRSMM': TVN / 'BM_HRSMM_tumor_n26_vs_BM_HRSMM_normal_n12.res.csv',
    'BM:IRSMM': TVN / 'BM_IRSMM_tumor_n40_vs_BM_IRSMM_normal_n39.res.csv',
    'BM:LRSMM': TVN / 'BM_LRSMM_tumor_n65_vs_BM_LRSMM_normal_n56.res.csv',
    'BM:MGUS': TVN / 'BM_MGUS_tumor_n26_vs_BM_MGUS_normal_n35.res.csv'}
tvn_bm_risk_matched_res = {
    'BM:MM': MATCH / 'BM_MM_tumor_mp_n27_vs_BM_MM_normal_mp_n27.res.csv',
    'BM:HRSMM': MATCH / 'BM_HRSMM_tumor_mp_n28_vs_BM_HRSMM_normal_mp_n28.res.csv',
    'BM:IRSMM': MATCH / 'BM_IRSMM_tumor_mp_n39_vs_BM_IRSMM_normal_mp_n38.res.csv',
    'BM:LRSMM': MATCH / 'BM_LRSMM_tumor_mp_n37_vs_BM_LRSMM_normal_mp_n35.res.csv',
    'BM:MGUS': MATCH / 'BM_MGUS_tumor_mp_n24_vs_BM_MGUS_normal_mp_n23.res.csv'}

tvn_bm_tumor = de_table(tvn_bm_tumor_res, marker_genes=KNOWN_TARGETS)
tvn_bm_group = de_table(tvn_bm_group_res, marker_genes=KNOWN_TARGETS)
tvn_bm_risk = de_table(tvn_bm_risk_res, marker_genes=KNOWN_TARGETS)
tvn_bm_risk_matched = de_table(tvn_bm_risk_matched_res, marker_genes=KNOWN_TARGETS)
tvn_bm_group.to_csv(PAPER / 'disease-tumor-mgus-smm-mm.csv')
tvn_bm_risk.to_csv(PAPER / 'disease-tumor-20-20-risk-degs.csv')
tvn_bm_risk_matched.to_csv(PAPER / 'disease-tumor-matched-20-20-risk-degs.csv')
tvn_bm_group.to_csv(SUMMARY / 'tvn_bm_group.csv')

# 2. malignant vs normal PCs, PB
tvn_pb_tumor_res = {
    'PB:ALL': TVN / 'PB_tumor_n57_vs_PB_normal_n77.res.csv',
    'PB:SMM': TVN / 'PB_SMM_tumor_n35_vs_PB_SMM_normal_n42.res.csv'}
tvn_pb_group_res = {
    'PB:MM': TVN / 'PB_MM_tumor_n16_vs_PB_MM_normal_n20.res.csv',
    'PB:SMM': TVN / 'PB_SMM_tumor_n35_vs_PB_SMM_normal_n42.res.csv',
    'PB:MGUS': TVN / 'PB_MGUS_tumor_n6_vs_PB_MGUS_normal_n15.res.csv'}
tvn_pb_tumor = de_table(tvn_pb_tumor_res)
tvn_pb_group = de_table(tvn_pb_group_res)
tvn_pb_group.to_csv(SUMMARY / 'tvn_pb_group.csv')

# 3. malignant PCs between stages, BM and PB
tvt_bm_res = {
    'BM:MM-SMM': TVT / 'BM_MM_tumor_n48_vs_BM_SMM_tumor_n131.res.csv',
    'BM:MM-MGUS': TVT / 'BM_MM_tumor_n48_vs_BM_MGUS_tumor_n26.res.csv',
    'BM:SMM-MGUS': TVT / 'BM_SMM_tumor_n131_vs_BM_MGUS_tumor_n26.res.csv'}
tvt_pb_res = {
    'PB:MM-SMM': TVT / 'PB_MM_tumor_n16_vs_PB_SMM_tumor_n35.res.csv',
    'PB:MM-MGUS': TVT / 'PB_MM_tumor_n16_vs_PB_MGUS_tumor_n6.res.csv',
    'PB:SMM-MGUS': TVT / 'PB_SMM_tumor_n35_vs_PB_MGUS_tumor_n6.res.csv'}
tvt_bm = de_table(tvt_bm_res, df_use_key=True)
tvt_bm.to_csv(SUMMARY / 'tvt_bm_group.csv')
tvt_pb = de_table(tvt_pb_res, df_use_key=True)
tvt_pb.to_csv(SUMMARY / 'tvt_pb_group.csv')

# 4. gene lists
tvn_bm_de_union = set(tvn_bm_group.index)
tvn_pb_de_union = set(tvn_pb_group.index)
tvt_bm_de_union = set(tvt_bm.index)
tvt_pb_de_union = set(tvt_pb.index)
tvn_bm_tumor_genes = set(tvn_bm_tumor.loc[tvn_bm_tumor['BM_tumor'] != 0].index)     # all malignant vs normal PCs
tvn_pb_tumor_genes = set(tvn_pb_tumor.loc[tvn_pb_tumor['PB_tumor'] != 0].index)

# malignant vs normal DEGs that also change between stages
bm_tn_de_tt_inter = tvt_bm_de_union.intersection(tvn_bm_de_union)
pb_tn_de_tt_inter = tvt_pb_de_union.intersection(tvn_pb_de_union)
tn_de_tt_inter = bm_tn_de_tt_inter.union(pb_tn_de_tt_inter)

# malignant vs normal DEGs (MGUS, SMM, MM) united with the DEGs between stages
bm_tn_de_tt_union = tvt_bm_de_union.union(tvn_bm_de_union)
pb_tn_de_tt_union = tvt_pb_de_union.union(tvn_pb_de_union)
tn_de_tt_union = bm_tn_de_tt_union.union(pb_tn_de_tt_union)

# DEGs of all malignant vs normal PCs united with the DEGs between stages
bm_tn_de_tt = tvt_bm_de_union.union(tvn_bm_tumor_genes)
pb_tn_de_tt = tvt_pb_de_union.union(tvn_pb_tumor_genes)
tn_de_tt = bm_tn_de_tt.union(pb_tn_de_tt)

write_list(list(bm_tn_de_tt_inter), GENE_LIST / 'bm_tumor_interDE_249genes.txt')
write_list(list(pb_tn_de_tt_inter), GENE_LIST / 'pb_tumor_interDE_32genes.txt')
write_list(list(tn_de_tt_inter), GENE_LIST / 'union_tumor_interDE_272genes.txt')
write_list(list(bm_tn_de_tt_union), GENE_LIST / 'tumor_de' / 'bm_tumor_unionDE_2657genes.txt')
write_list(list(pb_tn_de_tt_union), GENE_LIST / 'pb_tumor_unionDE_3768genes.txt')
write_list(list(tn_de_tt_union), GENE_LIST / 'union_tumor_unionDE_4987genes.txt')
write_list(list(bm_tn_de_tt), GENE_LIST / 'bm_tumor_DE_1382genes.txt')
write_list(list(pb_tn_de_tt), GENE_LIST / 'pb_tumor_DE_2649genes.txt')
write_list(list(tn_de_tt), GENE_LIST / 'union_tumor_DE_3233genes.txt')
write_list(sorted(set(tvt_bm.index)), GENE_LIST / 'tumor_de' / 'bm_tumor_tvtDE_476genes.txt')
