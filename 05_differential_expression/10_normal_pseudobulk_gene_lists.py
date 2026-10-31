#!/usr/bin/env python3
"""Normal-PC gene lists from the pseudobulk DESeq2 comparisons with healthy donors, between stages and between BM and
PB, including candidate genes of signature detection (Methods, Detection of single-cell transcriptional signatures).

Purpose:      Reads the DESeq2 results (5' libraries) of the normal PCs of patients vs healthy-donor PCs (all patients,
              SMM, MGUS/SMM/MM), of normal PCs between MGUS, SMM and MM, and of normal PCs between BM and PB. A gene
              is differentially expressed (DEG) when the adjusted P (Benjamini-Hochberg) < 0.01 and |log2 fold change|
              > 1; adjusted P of NA counts as 1; immunoglobulin genes (IG[HKL]) and genes named <name>.<number> are
              removed. Tables of log2 fold changes per comparison (0 when not a DEG). Gene lists per tissue: DEGs vs
              healthy donors at MGUS, SMM or MM united with the DEGs between these stages, their intersection, and the
              DEGs of all patients vs donors united with the stage DEGs; and the BM vs PB DEGs united with the BM list
              (candidate genes of signature detection). Must be run with PYTHONHASHSEED=0 (gene order).

Inputs:       data/pseudobulk/result/GEX5_only/normal_vs_donor/*.res.csv, data/pseudobulk/result/GEX5_only/
              normal_vs_normal/*.res.csv and data/pseudobulk/result/GEX5_only/BM_npc_vs_PB_npc/*.res.csv: DESeq2
              results (baseMean, log2FoldChange, lfcSE, stat, padj) of the comparisons named below.

Outputs:      data/pseudobulk/summary/nvd_bm_group.csv, nvd_pb_group.csv, nvn_bm_group.csv, nvn_pb_group.csv: log2
              fold changes, genes x comparisons.
              data/signature/gene_list/{bm,pb,union}_normal_interDE_*genes.txt, {pb,union}_normal_unionDE_*genes.txt,
              {bm,pb,union}_normal_DE_*genes.txt,
              data/signature/gene_list/normal_de/union/bm_normal_unionDE_2245genes.txt and
              data/signature/gene_list/normal_de/npc_core_bm_normal_unionDE_3744genes.txt: gene lists.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd

np.random.seed(0)

NVD = data("pseudobulk", "result", "GEX5_only", "normal_vs_donor")
NVN = data("pseudobulk", "result", "GEX5_only", "normal_vs_normal")
NPC = data("pseudobulk", "result", "GEX5_only", "BM_npc_vs_PB_npc")
SUMMARY = data("pseudobulk", "summary")
GENE_LIST = data("signature", "gene_list")
for d in [SUMMARY, GENE_LIST / "normal_de" / "union"]:
    d.mkdir(parents=True, exist_ok=True)


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


def de_table(results, fdr=0.01, log2fc=1, df_use_key=False):
    """log2 fold changes of the DEGs of several DESeq2 results, genes x comparisons, 0 when not a DEG. Columns are named
    after the case group (or the key with df_use_key)."""
    de_genes = set()
    table = pd.DataFrame()
    for key in results.keys():
        case = os.path.basename(results[key]).replace('.res.csv', '').split('_vs_')[0]
        name = key if df_use_key else '_n'.join(case.split('_n')[:-1])     # case group without its sample number
        df = clean_deseq2_matrix(pd.read_csv(results[key], index_col=0))
        genes = get_sig_genes(df, fdr, log2fc)
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


# 1. normal PCs of patients vs healthy donors, BM and PB
nvd_bm_normal = de_table({
    'BM:ALL': NVD / 'BM_normal_n159_vs_BM_donor_n28.res.csv',
    'BM:SMM': NVD / 'BM_SMM_normal_n100_vs_BM_donor_n28.res.csv'})
nvd_bm_group = de_table({
    'BM:MM': NVD / 'BM_MM_normal_n27_vs_BM_donor_n28.res.csv',
    'BM:SMM': NVD / 'BM_SMM_normal_n100_vs_BM_donor_n28.res.csv',
    'BM:MGUS': NVD / 'BM_MGUS_normal_n32_vs_BM_donor_n28.res.csv'})
nvd_bm_group.to_csv(SUMMARY / 'nvd_bm_group.csv')

nvd_pb_normal = de_table({
    'PB:ALL': NVD / 'PB_normal_n77_vs_PB_donor_n8.res.csv',
    'PB:SMM': NVD / 'PB_SMM_normal_n42_vs_PB_donor_n8.res.csv'})
nvd_pb_group = de_table({
    'PB:MM': NVD / 'PB_MM_normal_n20_vs_PB_donor_n8.res.csv',
    'PB:SMM': NVD / 'PB_SMM_normal_n42_vs_PB_donor_n8.res.csv',
    'PB:MGUS': NVD / 'PB_MGUS_normal_n15_vs_PB_donor_n8.res.csv'})
nvd_pb_group.to_csv(SUMMARY / 'nvd_pb_group.csv')

# 2. normal PCs between stages, BM and PB
nvn_bm = de_table({
    'BM:MM-SMM': NVN / 'BM_MM_normal_n27_vs_BM_SMM_normal_n100.res.csv',
    'BM:MM-MGUS': NVN / 'BM_MM_normal_n27_vs_BM_MGUS_normal_n32.res.csv',
    'BM:SMM-MGUS': NVN / 'BM_SMM_normal_n100_vs_BM_MGUS_normal_n32.res.csv'}, df_use_key=True)
nvn_bm.to_csv(SUMMARY / 'nvn_bm_group.csv')

nvn_pb = de_table({
    'PB:MM-SMM': NVN / 'PB_MM_normal_n20_vs_PB_SMM_normal_n42.res.csv',
    'PB:MM-MGUS': NVN / 'PB_MM_normal_n20_vs_PB_MGUS_normal_n15.res.csv',
    'PB:SMM-MGUS': NVN / 'PB_SMM_normal_n42_vs_PB_MGUS_normal_n15.res.csv'}, df_use_key=True)
nvn_pb.to_csv(SUMMARY / 'nvn_pb_group.csv')

# 3. normal PCs, BM vs PB (5' libraries)
npc = de_table({'nPC:GEX5': NPC / 'BM_npc_GEX5_n187_vs_PB_npc_GEX5_n85.res.csv'}, df_use_key=True)

# 4. gene lists
nvd_bm_de_union = set(nvd_bm_group.index)
nvd_pb_de_union = set(nvd_pb_group.index)
nvn_bm_de_union = set(nvn_bm.index)
nvn_pb_de_union = set(nvn_pb.index)
nvd_bm_normal_genes = set(nvd_bm_normal.loc[nvd_bm_normal['BM_normal'] != 0].index)     # all patients vs donors
nvd_pb_normal_genes = set(nvd_pb_normal.loc[nvd_pb_normal['PB_normal'] != 0].index)
npc_core_genes = set(npc[npc.apply(lambda x: sum(x == 0), axis=1) == 0].index)         # BM vs PB DEGs

# DEGs vs donors that also change between stages (BM, PB, and genes found in both tissues)
bm_nd_de_nn_inter = nvn_bm_de_union.intersection(nvd_bm_de_union)
pb_nd_de_nn_inter = nvn_pb_de_union.intersection(nvd_pb_de_union)
nd_de_nn_inter = bm_nd_de_nn_inter.intersection(pb_nd_de_nn_inter)

# DEGs vs donors (MGUS, SMM, MM) united with the DEGs between stages
bm_nd_de_nn_union = nvn_bm_de_union.union(nvd_bm_de_union)
pb_nd_de_nn_union = nvn_pb_de_union.union(nvd_pb_de_union)
nd_de_nn_union = bm_nd_de_nn_union.union(pb_nd_de_nn_union)

# DEGs of all patients vs donors united with the DEGs between stages
bm_nd_de_nn = nvn_bm_de_union.union(nvd_bm_normal_genes)
pb_nd_de_nn = nvn_pb_de_union.union(nvd_pb_normal_genes)
nd_de_nn = bm_nd_de_nn.union(pb_nd_de_nn)

write_list(list(bm_nd_de_nn_inter), GENE_LIST / 'bm_normal_interDE_502genes.txt')
write_list(list(pb_nd_de_nn_inter), GENE_LIST / 'pb_normal_interDE_5genes.txt')
write_list(list(nd_de_nn_inter), GENE_LIST / 'union_normal_interDE_1genes.txt')
write_list(list(bm_nd_de_nn_union), GENE_LIST / 'normal_de' / 'union' / 'bm_normal_unionDE_2245genes.txt')
write_list(list(pb_nd_de_nn_union), GENE_LIST / 'pb_normal_unionDE_108genes.txt')
write_list(list(nd_de_nn_union), GENE_LIST / 'union_normal_unionDE_2283genes.txt')
write_list(list(bm_nd_de_nn), GENE_LIST / 'bm_normal_DE_1073genes.txt')
write_list(list(pb_nd_de_nn), GENE_LIST / 'pb_normal_DE_91genes.txt')
write_list(list(nd_de_nn), GENE_LIST / 'union_normal_DE_1120genes.txt')
write_list(list(npc_core_genes.union(bm_nd_de_nn_union)),
           GENE_LIST / 'normal_de' / 'npc_core_bm_normal_unionDE_3744genes.txt')
