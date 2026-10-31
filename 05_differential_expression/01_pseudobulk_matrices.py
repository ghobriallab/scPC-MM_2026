#!/usr/bin/env python3
"""Pseudobulk count matrices and sample tables of the pseudobulk differential expression analyses (Methods, Differential
expression analysis, pseudobulk level).

Purpose:      Sums the UMI counts of the 5' PCs of each group with at least 30 cells: the malignant PCs of each
              malignant population (tumor_id) and the normal PCs of each patient, per tissue (BM, PB) and disease stage
              (SMM split by risk group, SMM_stage_lenient), with healthy donors apart. Writes one matrix of all groups
              and the sample tables used to select groups:
              one row per patient, tissue and stage; the same with the cytogenetic group of the patient (Multi when
              its malignant populations have two groups) and BM_PB_set (Pair when the patient has as many BM as PB
              rows at a stage).

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv: gex_id (index), assay_type, disease_status (HD =
              healthy donor), SMM_stage_lenient, sample_type, redcap_id, Sex.
              data/h5/agg/plasma-cell-GEX5-counts-250207.h5ad: UMI counts of all PCs (obs gex_id, sample_type,
              redcap_id, assay_type, annotation, tumor_id).
              data/annotation/plasma-cell-annotation-final-240327.csv: per-cell annotation, columns missing from the
              AnnData obs are added.
              data/metadata/cytogenetics-final-240327.csv: redcap_id, cytogenetics (one row per malignant population).

Outputs:      data/pseudobulk/matrix/parts/{BM,PB}_{pat_tumor,pat_normal,donor}_GEX5.csv and
              data/pseudobulk/matrix/pseudobulk-expr-matrix-GEX5.csv: summed counts, genes x groups; group names
              <tissue>.<stage>.<record id>.normal or <tissue>.<stage>.<record id>.tumor.<k>.
              data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240329.csv: sample_type, disease_status, disease_categs,
              redcap_id, assay_type, Sex.
              data/pseudobulk/matrix/pseudobulk-metadata-GEX5-240907.csv: the same with cytogenetics and BM_PB_set.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)

MATRIX = data("pseudobulk", "matrix")
PARTS = MATRIX / "parts"
PARTS.mkdir(parents=True, exist_ok=True)


def create_pseudobulk(adata, group_col, output_file, min_cell=30):
    """Summed counts of each group with at least min_cell cells; '/', '-' and ':' in names become '_', '_' and '.'."""
    n_cell = adata.obs[group_col].value_counts()
    n_cell = n_cell[n_cell >= min_cell]
    X_sum = pd.DataFrame(index=adata.var_names, columns=n_cell.index)
    for pid, df in adata.obs.groupby(group_col):
        if pid not in n_cell.index:
            continue
        X_sum.loc[:, pid] = np.array(adata[df.index].X.sum(axis=0)).reshape(-1)
    X_sum.columns = [col.replace('/', '_').replace('-', '_').replace(':', '.') for col in X_sum.columns]
    X_sum.to_csv(output_file)


def join_parts(input_files, output_file):
    """Joins part matrices on their genes, in the order given."""
    mat = pd.read_csv(input_files[0], index_col=0)
    for input_file in input_files[1:]:
        mat = mat.join(pd.read_csv(input_file, index_col=0))
    mat.to_csv(output_file)


# 1. sample table of the 5' libraries; disease_categs = disease stage, with SMM replaced by its risk group
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
meta = meta[meta['assay_type'] == 'GEX5']
meta['disease_categs'] = list(meta['disease_status'])
has_risk = ~meta['SMM_stage_lenient'].isna()
meta.loc[has_risk, 'disease_categs'] = meta.loc[has_risk, 'SMM_stage_lenient']
donor_ids = list(meta.loc[meta['disease_status'] == 'HD', 'redcap_id'])

# 2. UMI counts of the 5' PCs with their annotation; pseudobulk group of each cell:
#    <tissue>:<stage>:<malignant population> for malignant PCs, <tissue>:<stage>:<record id>:normal for normal PCs
adata = sc.read(data("h5", "agg", "plasma-cell-GEX5-counts-250207.h5ad"))
annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
attach = list(set(annot.columns).difference(set(adata.obs.columns)))
adata.obs = adata.obs.join(annot[attach])
adata = adata[adata.obs['assay_type'] == 'GEX5']
adata.obs['group_id'] = list(adata.obs['tumor_id'])
pos = adata.obs['tumor_id'].isna()
adata.obs.loc[pos, 'group_id'] = np.array(adata.obs.loc[pos, 'redcap_id']) + ':normal'
adata.obs['group_id'] = (np.array(adata.obs['sample_type']) + ':'
                         + np.array(meta.loc[adata.obs['gex_id'], 'disease_categs']) + ':'
                         + np.array(adata.obs['group_id']))

# 3. matrices of all groups: malignant PCs, normal PCs of patients, normal PCs of healthy donors, per tissue
for tissue in ['BM', 'PB']:
    in_tissue = adata.obs['sample_type'] == tissue
    create_pseudobulk(adata[in_tissue & (adata.obs['annotation'] == 'tumor')], 'group_id',
                      PARTS / (tissue + '_pat_tumor_GEX5.csv'))
    create_pseudobulk(adata[in_tissue & (adata.obs['annotation'] == 'normal')
                            & (~adata.obs['redcap_id'].isin(donor_ids))],
                      'group_id', PARTS / (tissue + '_pat_normal_GEX5.csv'))
    create_pseudobulk(adata[in_tissue & (adata.obs['annotation'] == 'normal')
                            & (adata.obs['redcap_id'].isin(donor_ids))],
                      'group_id', PARTS / (tissue + '_donor_GEX5.csv'))
join_parts([PARTS / f for f in ['BM_donor_GEX5.csv', 'BM_pat_normal_GEX5.csv', 'BM_pat_tumor_GEX5.csv',
                                'PB_donor_GEX5.csv', 'PB_pat_normal_GEX5.csv', 'PB_pat_tumor_GEX5.csv']],
           MATRIX / 'pseudobulk-expr-matrix-GEX5.csv')

# 4. sample table of the pseudobulks: one row per patient, tissue and stage ('/' in record ids written as '_', as in
#    the group names)
de_meta = meta[['sample_type', 'disease_status', 'disease_categs', 'redcap_id', 'assay_type', 'Sex']].drop_duplicates()
de_meta['redcap_id'] = [x.replace('/', '_') for x in de_meta['redcap_id']]
de_meta.to_csv(MATRIX / 'pseudobulk-metadata-GEX5-240329.csv', index=False)

# 5. the same with the cytogenetic group of each patient (Multi when its malignant populations have two groups) and
#    BM_PB_set: Pair when the patient has as many BM as PB rows at that stage, otherwise Single
de_meta = pd.read_csv(MATRIX / 'pseudobulk-metadata-GEX5-240329.csv')
cytogenetics = pd.read_csv(data("metadata", "cytogenetics-final-240327.csv"), index_col=0)
cytogenetics['de_redcap_id'] = [x.replace('/', '_') for x in cytogenetics['redcap_id']]
de_meta['cytogenetics'] = np.nan
for i, row in de_meta.iterrows():
    c = set(sorted(cytogenetics.loc[cytogenetics['de_redcap_id'] == row['redcap_id'], 'cytogenetics']))
    if len(c) == 0:
        continue
    elif len(c) > 1:
        de_meta.loc[i, 'cytogenetics'] = 'Multi'
    else:
        de_meta.loc[i, 'cytogenetics'] = sorted(c)[0]
de_meta['BM_PB_set'] = 'Single'
for _, df in de_meta.groupby(['redcap_id', 'disease_categs']):
    if df.shape[0] == 1:
        continue
    if sum(df['sample_type'] == 'BM') == sum(df['sample_type'] == 'PB'):
        de_meta.loc[df.index, 'BM_PB_set'] = 'Pair'
de_meta.to_csv(MATRIX / 'pseudobulk-metadata-GEX5-240907.csv', index=False)
