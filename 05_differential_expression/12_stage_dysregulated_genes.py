#!/usr/bin/env python3
"""Genes dysregulated in malignant BM PCs from MGUS on (early), from SMM on, or in MM only (late), from pseudobulk
comparisons of the malignant PCs of each disease stage with the normal PCs of every stage and of healthy donors
(Methods, Differential expression analysis, pseudobulk level).

Purpose:      For each stage (MGUS, LRSMM, IRSMM, HRSMM, MM), reads the DESeq2 results of its malignant BM PCs vs the
              normal BM PCs of each of these stages and of healthy donors (6 comparisons), and the results of the
              malignant BM PCs of MGUS, SMM, MM and of all patients vs normal BM PCs (4 comparisons). DEG = adjusted P
              (Benjamini-Hochberg) < 0.01 and |log2 fold change| > 1 (adjusted P of NA = 1; IG[HKL] genes and genes
              named <name>.<number> removed). A gene is consistently up (down) in a set of comparisons when it is an
              upregulated (downregulated) DEG in all of them. Lists: per stage and for the 4 comparisons with all
              normal PCs (TvN); early = consistent in all six sets; SMM onward = consistent at MM, in SMM vs all
              normal PCs and in at least two of the three SMM risk groups, not at MGUS; late = consistent at MM and in
              MM vs all normal PCs only. Must be run with PYTHONHASHSEED=0.

Inputs:       data/pseudobulk/result/GEX5_only/stage_tumor_v_normal/{MGUS,LRSMM,IRSMM,HRSMM,MM}/*.res.csv and
              data/pseudobulk/result/GEX5_only/stage_tumor_v_normal/cross/*.res.csv: DESeq2 results (baseMean,
              log2FoldChange, lfcSE, stat, padj) of the comparisons named below.

Outputs:      data/pseudobulk/summary/{mgus,lrsmm,irsmm,hrsmm,mm}_vs_normal.csv: log2 fold changes, genes x comparisons.
              data/signature/gene_list/cross_tvn_de/bm_<stage or TvN>_{upreg,dnreg}_<n>genes.txt: consistent genes.
              data/signature/gene_list/cross_tvn_de/early_upreg_111genes.txt, early_downreg_36genes.txt,
              smm-on_upreg_116genes.txt, smm-on_downreg_66genes.txt, late_upreg_74genes.txt, late_downreg_74genes.txt:
              stage lists, one gene per line.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, upsetplot); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd
import upsetplot

np.random.seed(0)

STAGE = data("pseudobulk", "result", "GEX5_only", "stage_tumor_v_normal")
SUMMARY = data("pseudobulk", "summary")
LISTS = data("signature", "gene_list", "cross_tvn_de")
for d in [SUMMARY, LISTS]:
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


def get_consensus(x, choice='upreg'):
    """Genes with log2 fold change > 1 (upreg) or < -1 (dnreg) in every column."""
    if choice == 'upreg':
        return list(x[x.apply(lambda g: sum(g > 1) == x.shape[1], axis=1)].index)
    elif choice == 'dnreg':
        return list(x[x.apply(lambda g: sum(g < -1) == x.shape[1], axis=1)].index)
    else:
        raise Exception('choose either from upreg or dnreg')


def write_list(array, outfile):
    with open(outfile, 'w') as fo:
        for elem in array:
            fo.write(elem + '\n')


# 1. malignant BM PCs of each stage vs the normal BM PCs of each stage and of healthy donors
mgus_inputs = {
    'MGUS|BM:MM': STAGE / 'MGUS' / 'BM_MGUS_tumor_n26_vs_BM_MM_normal_n27.res.csv',
    'MGUS|BM:HRSMM': STAGE / 'MGUS' / 'BM_MGUS_tumor_n26_vs_BM_HRSMM_normal_n28.res.csv',
    'MGUS|BM:IRSMM': STAGE / 'MGUS' / 'BM_MGUS_tumor_n26_vs_BM_IRSMM_normal_n38.res.csv',
    'MGUS|BM:LRSMM': STAGE / 'MGUS' / 'BM_MGUS_tumor_n26_vs_BM_LRSMM_normal_n35.res.csv',
    'MGUS|BM:MGUS': STAGE / 'MGUS' / 'BM_MGUS_tumor_n26_vs_BM_MGUS_normal_n32.res.csv',
    'MGUS|BM:Donor': STAGE / 'MGUS' / 'BM_MGUS_tumor_n26_vs_BM_donor_n28.res.csv'}
lrsmm_inputs = {
    'LRSMM|BM:MM': STAGE / 'LRSMM' / 'BM_LRSMM_tumor_n40_vs_BM_MM_normal_n27.res.csv',
    'LRSMM|BM:HRSMM': STAGE / 'LRSMM' / 'BM_LRSMM_tumor_n40_vs_BM_HRSMM_normal_n28.res.csv',
    'LRSMM|BM:IRSMM': STAGE / 'LRSMM' / 'BM_LRSMM_tumor_n40_vs_BM_IRSMM_normal_n38.res.csv',
    'LRSMM|BM:LRSMM': STAGE / 'LRSMM' / 'BM_LRSMM_tumor_n40_vs_BM_LRSMM_normal_n35.res.csv',
    'LRSMM|BM:MGUS': STAGE / 'LRSMM' / 'BM_LRSMM_tumor_n40_vs_BM_MGUS_normal_n32.res.csv',
    'LRSMM|BM:Donor': STAGE / 'LRSMM' / 'BM_LRSMM_tumor_n40_vs_BM_donor_n28.res.csv'}
irsmm_inputs = {
    'IRSMM|BM:MM': STAGE / 'IRSMM' / 'BM_IRSMM_tumor_n42_vs_BM_MM_normal_n27.res.csv',
    'IRSMM|BM:HRSMM': STAGE / 'IRSMM' / 'BM_IRSMM_tumor_n42_vs_BM_HRSMM_normal_n28.res.csv',
    'IRSMM|BM:IRSMM': STAGE / 'IRSMM' / 'BM_IRSMM_tumor_n42_vs_BM_IRSMM_normal_n38.res.csv',
    'IRSMM|BM:LRSMM': STAGE / 'IRSMM' / 'BM_IRSMM_tumor_n42_vs_BM_LRSMM_normal_n35.res.csv',
    'IRSMM|BM:MGUS': STAGE / 'IRSMM' / 'BM_IRSMM_tumor_n42_vs_BM_MGUS_normal_n32.res.csv',
    'IRSMM|BM:Donor': STAGE / 'IRSMM' / 'BM_IRSMM_tumor_n42_vs_BM_donor_n28.res.csv'}
hrsmm_inputs = {
    'HRSMM|BM:MM': STAGE / 'HRSMM' / 'BM_HRSMM_tumor_n39_vs_BM_MM_normal_n27.res.csv',
    'HRSMM|BM:HRSMM': STAGE / 'HRSMM' / 'BM_HRSMM_tumor_n39_vs_BM_HRSMM_normal_n28.res.csv',
    'HRSMM|BM:IRSMM': STAGE / 'HRSMM' / 'BM_HRSMM_tumor_n39_vs_BM_IRSMM_normal_n38.res.csv',
    'HRSMM|BM:LRSMM': STAGE / 'HRSMM' / 'BM_HRSMM_tumor_n39_vs_BM_LRSMM_normal_n35.res.csv',
    'HRSMM|BM:MGUS': STAGE / 'HRSMM' / 'BM_HRSMM_tumor_n39_vs_BM_MGUS_normal_n32.res.csv',
    'HRSMM|BM:Donor': STAGE / 'HRSMM' / 'BM_HRSMM_tumor_n39_vs_BM_donor_n28.res.csv'}
mm_inputs = {
    'MM|BM:MM': STAGE / 'MM' / 'BM_MM_tumor_n39_vs_BM_MM_normal_n27.res.csv',
    'MM|BM:HRSMM': STAGE / 'MM' / 'BM_MM_tumor_n39_vs_BM_HRSMM_normal_n28.res.csv',
    'MM|BM:IRSMM': STAGE / 'MM' / 'BM_MM_tumor_n39_vs_BM_IRSMM_normal_n38.res.csv',
    'MM|BM:LRSMM': STAGE / 'MM' / 'BM_MM_tumor_n39_vs_BM_LRSMM_normal_n35.res.csv',
    'MM|BM:MGUS': STAGE / 'MM' / 'BM_MM_tumor_n39_vs_BM_MGUS_normal_n32.res.csv',
    'MM|BM:Donor': STAGE / 'MM' / 'BM_MM_tumor_n39_vs_BM_donor_n28.res.csv'}

mgus_res = de_table(mgus_inputs, df_use_key=True)
mgus_res.to_csv(SUMMARY / 'mgus_vs_normal.csv')
lrsmm_res = de_table(lrsmm_inputs, df_use_key=True)
lrsmm_res.to_csv(SUMMARY / 'lrsmm_vs_normal.csv')
irsmm_res = de_table(irsmm_inputs, df_use_key=True)
irsmm_res.to_csv(SUMMARY / 'irsmm_vs_normal.csv')
hrsmm_res = de_table(hrsmm_inputs, df_use_key=True)
hrsmm_res.to_csv(SUMMARY / 'hrsmm_vs_normal.csv')
mm_res = de_table(mm_inputs, df_use_key=True)
mm_res.to_csv(SUMMARY / 'mm_vs_normal.csv')

# 2. malignant BM PCs of MGUS, SMM, MM and of all patients vs normal BM PCs
tvn_inputs = {
    'BM:MM': STAGE / 'cross' / 'BM_MM_tumor_n39_vs_BM_ALL_normal_n188.res.csv',
    'BM:SMM': STAGE / 'cross' / 'BM_SMM_tumor_n121_vs_BM_ALL_normal_n188.res.csv',
    'BM:MGUS': STAGE / 'cross' / 'BM_MGUS_tumor_n26_vs_BM_ALL_normal_n188.res.csv',
    'BM:ALL': STAGE / 'cross' / 'BM_tumor_n186_vs_BM_normal_n160.res.csv'}
tvn_res = de_table(tvn_inputs, df_use_key=True)

# 3. consistently up- and downregulated genes per set of comparisons
dnreg = dict()
upreg = dict()
for sg, res in [('MGUS', mgus_res), ('LRSMM', lrsmm_res), ('IRSMM', irsmm_res), ('HRSMM', hrsmm_res), ('MM', mm_res),
                ('TvN', tvn_res)]:
    dnreg[sg] = get_consensus(res, 'dnreg')
    upreg[sg] = get_consensus(res, 'upreg')
for sg in dnreg.keys():
    write_list(sorted(dnreg[sg]), LISTS / 'bm_{}_{}_{}genes.txt'.format(sg, 'dnreg', len(dnreg[sg])))
    write_list(sorted(upreg[sg]), LISTS / 'bm_{}_{}_{}genes.txt'.format(sg, 'upreg', len(upreg[sg])))

# 4. early genes: consistent at every stage and in the comparisons with all normal PCs
#    (membership columns MGUS, LRSMM, IRSMM, HRSMM, MM, TvN)
up_df = upsetplot.from_contents(upreg)
down_df = upsetplot.from_contents(dnreg)
early_upreg = sorted(up_df.loc[(True, True, True, True, True, True), 'id'])
early_dnreg = sorted(down_df.loc[(True, True, True, True, True, True), 'id'])
write_list(sorted(early_upreg), LISTS / 'early_upreg_111genes.txt')
write_list(sorted(early_dnreg), LISTS / 'early_downreg_36genes.txt')

# 5. SMM-onward genes: not at MGUS; at MM, in SMM vs all normal PCs and in at least two of the three risk groups
#    (membership columns MGUS, LRSMM, IRSMM, HRSMM, MM, SMM|TvN)
_ = dnreg.pop('TvN')
_ = upreg.pop('TvN')
dnreg['SMM|TvN'] = get_consensus(tvn_res[['BM:SMM']], 'dnreg')
upreg['SMM|TvN'] = get_consensus(tvn_res[['BM:SMM']], 'upreg')

up_df = upsetplot.from_contents(upreg)
middle_upreg = sorted(up_df.loc[(False, True, True, True, True, True), 'id'])            # all three risk groups
middle_upreg.extend(sorted(up_df.loc[(False, False, True, True, True, True), 'id']))     # not LRSMM
middle_upreg.extend(sorted(up_df.loc[(False, True, False, True, True, True), 'id']))     # not IRSMM
middle_upreg.extend(sorted(up_df.loc[(False, True, True, False, True, True), 'id']))     # not HRSMM

down_df = upsetplot.from_contents(dnreg)
middle_dnreg = sorted(down_df.loc[(False, True, True, True, True, True), 'id'])
middle_dnreg.extend(sorted(down_df.loc[(False, False, True, True, True, True), 'id']))
middle_dnreg.extend(sorted(down_df.loc[(False, True, False, True, True, True), 'id']))
middle_dnreg.extend(sorted(down_df.loc[(False, True, True, False, True, True), 'id']))

write_list(sorted(middle_upreg), LISTS / 'smm-on_upreg_116genes.txt')
write_list(sorted(middle_dnreg), LISTS / 'smm-on_downreg_66genes.txt')

# 6. late genes: at MM and in MM vs all normal PCs only (membership columns MGUS, LRSMM, IRSMM, HRSMM, MM, MM|TvN)
_ = dnreg.pop('SMM|TvN')
_ = upreg.pop('SMM|TvN')
dnreg['MM|TvN'] = get_consensus(tvn_res[['BM:MM']], 'dnreg')
upreg['MM|TvN'] = get_consensus(tvn_res[['BM:MM']], 'upreg')

up_df = upsetplot.from_contents(upreg)
late_upreg = sorted(up_df.loc[(False, False, False, False, True, True), 'id'])
down_df = upsetplot.from_contents(dnreg)
late_dnreg = sorted(down_df.loc[(False, False, False, False, True, True), 'id'])

write_list(sorted(late_upreg), LISTS / 'late_upreg_74genes.txt')
write_list(sorted(late_dnreg), LISTS / 'late_downreg_74genes.txt')
