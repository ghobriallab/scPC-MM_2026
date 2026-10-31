#!/usr/bin/env python3
"""Within-patient differential expression between malignant and normal BM PCs, per patient, disease stage and malignant
population (Methods, Differential expression analysis, single-cell level; Fig. 2A).

Purpose:      For each patient and stage, compares the malignant BM PCs of each malignant population (clone_id) with
              the normal BM PCs of the same stage: two-sided Wilcoxon rank-sum test (scanpy rank_genes_groups,
              Benjamini-Hochberg adjusted P). Comparisons with fewer than 30 malignant or 30 normal PCs are skipped.
              Stages from disease_status (MGUS, MM); SMM patients are compared by risk group (SMM_stage_lenient; the
              IMWG 2/20/20 labels are applied by 15_tumor_de_imwg_tables.py). Runs the patients given as arguments (redcap_id values)
              or, without arguments, every patient of the sample table (disease_status other than HD) that has a
              per-patient AnnData object, one after the other. Runs with PYTHONHASHSEED=0 (the script restarts
              itself with it when unset).

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv: gex_id (index), redcap_id, disease_status,
              SMM_stage_lenient.
              data/annotation/plasma-cell-annotation-final-240327.csv: cell (index), redcap_id, annotation (tumor,
              normal), clone_id.
              data/h5/pat/clustered/annot/<redcap_id>.h5ad ('/' in the id written as '-'): PCs of one patient (obs
              redcap_id, gex_id, sample_type).

Outputs:      data/scrna_de/de_files/tumor_vs_normal/<redcap_id>_BM_<stage>_tumor_<clone>.wilcox.csv: scanpy
              rank_genes_groups table (names, scores, logfoldchanges, pvals, pvals_adj).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

if not os.getenv('PYTHONHASHSEED'):                       # restart with a set hash seed
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)

OUT = data("scrna_de", "de_files", "tumor_vs_normal")


def patient_file(redcap_id):
    """Per-patient AnnData object ('/' in the record id written as '-')."""
    return data("h5", "pat", "clustered", "annot", redcap_id.replace('/', '-') + '.h5ad')


def compute_de(adata, redcap_id, stage_column="disease_status"):
    """Wilcoxon test of each malignant population vs the normal PCs of the same tissue and stage."""
    adata.obs['attr'] = np.nan
    adata.obs['attr'] = adata.obs.apply(lambda x: ':'.join([x['sample_type'], x[stage_column]]), axis=1)
    adata.obs['attr'] = pd.Categorical(adata.obs['attr'])

    # group = <tissue>:<stage>:<tumor | normal>, with :<clone> for malignant populations
    adata.obs['group'] = np.nan
    adata.obs['group'] = adata.obs.apply(lambda x: ':'.join([x['sample_type'], x[stage_column], x['annotation']]),
                                         axis=1)
    adata.obs['group'] = adata.obs.apply(lambda x: ':'.join([x['group'], str(int(x['clone_id']))])
                                         if str(x['clone_id']) != 'nan' else x['group'], axis=1)
    adata.obs['group'] = pd.Categorical(adata.obs['group'])

    for attr in adata.obs['attr'].unique():
        tumor_names = list(adata.obs.loc[(adata.obs['attr'] == attr) & (adata.obs['annotation'] == 'tumor'),
                                         'group'].unique())
        normal_name = attr + ':normal'
        normal_cells = sum(adata.obs['group'] == normal_name)
        out_dir = OUT
        for tumor_name in tumor_names:
            tumor_cells = sum(adata.obs['group'] == tumor_name)
            if tumor_cells < 30 or normal_cells < 30:
                continue
            tvn_id = redcap_id.replace('/', '-') + "_" + tumor_name.replace(':', '_') + '.wilcox'
            sc.tl.rank_genes_groups(adata, 'group', key_added=tvn_id, groups=[tumor_name], reference=normal_name,
                                    method='wilcoxon')
            df = sc.get.rank_genes_groups_df(adata, key=tvn_id, group=tumor_name)
            df.to_csv(out_dir / (tvn_id + '.csv'), index=False)


def run_patient(path, meta, annot):
    """All comparisons of one patient."""
    adata = sc.read_h5ad(path)
    adata = adata[adata.obs['sample_type'] == 'BM'].copy()                 # BM only (Fig. 2A)
    redcap_id = list(adata.obs['redcap_id'].unique())[0]
    pat_annot = annot[annot['redcap_id'] == redcap_id]
    adata.obs['annotation'] = pat_annot.loc[adata.obs.index, 'annotation']
    adata.obs['clone_id'] = pat_annot.loc[adata.obs.index, 'clone_id']
    # stage: MGUS, MM, or the SMM risk group
    meta['disease_categ'] = list(meta['disease_status'])
    has_risk = ~meta['SMM_stage_lenient'].isna()
    meta.loc[has_risk, 'disease_categ'] = meta.loc[has_risk, 'SMM_stage_lenient']
    adata.obs['disease_categ'] = list(meta.loc[adata.obs['gex_id'], 'disease_categ'])
    adata = adata[adata.obs['disease_categ'] != 'SMM'].copy()             # SMM without a risk group: not compared
    compute_de(adata, redcap_id, stage_column="disease_categ")


def main(patients):
    meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
    annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
    if not patients:
        patients = sorted(meta.loc[meta['disease_status'] != 'HD', 'redcap_id'].dropna().unique())
        patients = [x for x in patients if patient_file(x).exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    for redcap_id in patients:
        run_patient(patient_file(redcap_id), meta, annot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wilcoxon tests of malignant vs normal PCs per patient and stage")
    parser.add_argument("redcap_id", nargs="*", help="patients to run (default: every patient with a per-patient file)")
    main(parser.parse_args().redcap_id)
