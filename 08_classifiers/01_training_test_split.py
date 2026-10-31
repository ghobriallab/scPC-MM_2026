#!/usr/bin/env python3
"""Training and test sets of the malignancy and MGUS-MM classifiers: 75/25 split of the eligible individuals within each
disease stage (Methods, classifier development).

Purpose:      Builds the per-cell feature table of all PCs and writes the feature-column lists that every classifier step
              reads. Features: mean log-normalized expression ("mean_expr:") and AUCell score ("auc:") of the 16
              signatures and of the six stage-wise DE gene sets (*_UP, *_DN), SenMayo senescence scores, CNV-like scores
              of five driver arms (mean log fold change and scanpy score), expression of nine translocation partner genes;
              column lists of the NF-kB index and RAS signature scores are written too. Disease stage: SMM replaced by its
              2/20/20 risk group (lenient call), HD renamed Donor. Cells: BM PCs of the down-sampled lists (at most 1,000
              malignant and 1,000 normal PCs per individual and tissue). Unit: individual x disease stage. Eligible units:
              >= 500 malignant PCs (patients) or >= 500 normal PCs (donors). Within each disease stage, eligible units are
              shuffled (Python random, seed 42) and the first 75% (rounded down) go to training, the rest to test; units
              under the cut-off are kept apart. Also writes the mean feature values of the malignant PCs of each unit
              (pseudobulk). Run with PYTHONHASHSEED=0.

Inputs:       data/signature/score/cell-level-module-gene-expr-auc.pkl (mean_expr:/auc: columns of the signatures and DE
              sets; sample_type, redcap_id, catissue_id, gex_id), cell-level-senescence-expr-auc.pkl,
              cell-level-copy-number-full-gene-expr.pkl (mean_expr:<arm>_full_log_fc, scanpy:<arm>_full_score),
              cell-level-cytogenetics-gene-expr.pkl (nine genes), nfkb-index-cell-level-score.pkl and
              lin-etal-ras-signature-cell-level-score.pkl (column names only).
              data/annotation/plasma-cell-annotation-final-240327.csv: annotation (tumor/normal), pct_counts_ig,
              pct_counts_mt, phase, G2M_score, S_score, total_counts, tumor_id.
              data/metadata/cytogenetics-final-240327.csv: cytogenetics, hrd_status (by tumor_id).
              data/metadata/tumor-cohort-metadata-final-240325.csv: disease_status, SMM_stage_lenient (by gex_id).
              data/barcode/downsample/{BM,PB}_{donor,normal,tumor}_1000cell.txt: down-sampled cell lists.

Outputs:      data/classifier/datasets/feature_columns/*.txt: the 13 feature-column lists.
              data/classifier/datasets/sample_info/train_samples.csv, test_samples.csv, below_cutoff_samples.csv:
              redcap_id, disease_status and malignant (tumor) and normal PC counts of each unit.
              data/classifier/datasets/sc-train-expr_134samples.pkl, sc-test-expr_47samples.pkl,
              sc-below-cutoff-extra-expr_42samples.pkl: per-cell features and annotation of each set.
              data/classifier/datasets/pseudo-mean-tumor-train-expr_134samples.pkl, pseudo-mean-tumor-test-expr_47samples.pkl,
              pseudo-mean-tumor-below-cutoff-extra-expr_42samples.pkl: mean malignant-PC features per unit.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

import random
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

DATASETS = ("classifier", "datasets")


def out(*parts):
    """an output file in the processed-data tree; its folder is created if needed"""
    p = data(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_list(elems, name):
    """one feature column per line"""
    with open(out(*DATASETS, "feature_columns", name), "w") as fo:
        for elem in elems:
            fo.write(elem + "\n")


# ---- 1. feature-column lists ----------------------------------------------------------------------------------------------
# signatures (16) and stage-wise DE gene sets (*_UP, *_DN), mean expression and AUCell
expr = pd.read_pickle(data("signature", "score", "cell-level-module-gene-expr-auc.pkl"))

feats = list(expr.columns[expr.columns.str.startswith("mean_expr:")])
mean_module_cols = [i for i in feats if not i.endswith("_UP") and not i.endswith("_DN")]
write_list(mean_module_cols, "mean_expr_16_gene_modules.txt")
mean_de_cols = [i for i in feats if i.endswith("_UP") or i.endswith("_DN")]
write_list(mean_de_cols, "mean_expr_6_de_gene_sets.txt")

feats = list(expr.columns[expr.columns.str.startswith("auc:")])
auc_module_cols = [i for i in feats if not i.endswith("_UP") and not i.endswith("_DN")]
write_list(auc_module_cols, "auc_16_gene_modules.txt")
auc_de_cols = [i for i in feats if i.endswith("_UP") or i.endswith("_DN")]
write_list(auc_de_cols, "auc_6_de_gene_sets.txt")

# SenMayo senescence score
aging = pd.read_pickle(data("signature", "score", "cell-level-senescence-expr-auc.pkl"))
mean_aging_cols = list(aging.columns[aging.columns.str.startswith("mean_expr:")])
write_list(mean_aging_cols, "mean_expr_1_aging_signature.txt")
auc_aging_cols = list(aging.columns[aging.columns.str.startswith("auc:")])
write_list(auc_aging_cols, "auc_1_aging_signature.txt")

# CNV-like scores of the driver arms
driver_arms = pd.read_pickle(data("signature", "score", "cell-level-copy-number-full-gene-expr.pkl"))
mean_cnv_arms = list(driver_arms.columns[driver_arms.columns.str.startswith("mean_expr:") &
                                         driver_arms.columns.str.endswith("_log_fc")])
write_list(mean_cnv_arms, "mean_expr_fc_driver_cnv_arms.txt")
scanpy_cnv_arms = list(driver_arms.columns[driver_arms.columns.str.startswith("scanpy:")])
write_list(scanpy_cnv_arms, "scanpy_score_driver_cnv_arms.txt")

# expression of the translocation partner genes
cyto_genes = pd.read_pickle(data("signature", "score", "cell-level-cytogenetics-gene-expr.pkl"))
cyto_gene_cols = list(cyto_genes.columns)
write_list(cyto_gene_cols, "cytogenetics_genes.txt")

# NF-kB index and RAS signature scores (column lists only)
nfkb_index = pd.read_pickle(data("signature", "score", "nfkb-index-cell-level-score.pkl"))
write_list(list(nfkb_index.columns[nfkb_index.columns.str.startswith("mean_expr:")]), "mean_expr_1_nfkb_index.txt")
write_list(list(nfkb_index.columns[nfkb_index.columns.str.startswith("auc:")]), "auc_1_nfkb_index.txt")
ras_scores = pd.read_pickle(data("signature", "score", "lin-etal-ras-signature-cell-level-score.pkl"))
write_list(list(ras_scores.columns[ras_scores.columns.str.startswith("mean_expr:")]), "mean_expr_2_ras_signature.txt")
write_list(list(ras_scores.columns[ras_scores.columns.str.startswith("auc:")]), "auc_2_ras_signature.txt")
del nfkb_index, ras_scores

# ---- 2. per-cell feature table with annotation and disease stage ------------------------------------------------------
annot_columns = ["annotation", "pct_counts_ig", "pct_counts_mt", "phase", "G2M_score", "S_score", "total_counts", "tumor_id"]
annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
annot = annot[annot_columns]
cytogenetics = pd.read_csv(data("metadata", "cytogenetics-final-240327.csv"), index_col=0)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
smm = ~meta["SMM_stage_lenient"].isna()
meta.loc[smm, "disease_status"] = meta.loc[smm, "SMM_stage_lenient"]          # SMM -> LRSMM, IRSMM or HRSMM

mat = expr[mean_module_cols + mean_de_cols + auc_module_cols + auc_de_cols]
mat = mat.join(aging[mean_aging_cols + auc_aging_cols])
mat = mat.join(driver_arms[mean_cnv_arms + scanpy_cnv_arms])
mat = mat.join(cyto_genes[cyto_gene_cols])
mat = mat.join(expr[["sample_type", "redcap_id", "catissue_id", "gex_id"]])
expr = mat
del mat, aging, driver_arms, cyto_genes

expr = expr.join(annot).join(cytogenetics[["cytogenetics", "hrd_status"]], on="tumor_id")
expr["cytogenetics"] = pd.Categorical(expr["cytogenetics"],
                                      categories=["t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)", "t(12;14)",
                                                  "t(14;16)", "t(14;20)", "HRD", "Other"])
expr["disease_status"] = list(meta.loc[expr["gex_id"], "disease_status"])
expr.loc[expr["disease_status"] == "HD", "disease_status"] = "Donor"

# ---- 3. units (individual x disease stage) of the down-sampled BM cells -----------------------------------------------
barcode_lists = ["BM_donor_1000cell.txt", "BM_normal_1000cell.txt", "BM_tumor_1000cell.txt",
                 "PB_donor_1000cell.txt", "PB_normal_1000cell.txt", "PB_tumor_1000cell.txt"]
cell_barcodes = []
for name in barcode_lists:
    cell_barcodes.extend([x.rstrip("\n") for x in open(data("barcode", "downsample", name))])

subset_expr = expr[expr.index.isin(cell_barcodes) & (expr["sample_type"] == "BM")].copy()

cell_counts = {"redcap_id": [], "disease_status": [], "tumor": [], "normal": []}
for x, df in subset_expr.groupby(["redcap_id", "disease_status"]):
    n_tumor = sum(df["annotation"] == "tumor")
    n_normal = df.shape[0] - n_tumor
    cell_counts["redcap_id"].append(x[0])
    cell_counts["disease_status"].append(x[1])
    cell_counts["tumor"].append(n_tumor)
    cell_counts["normal"].append(n_normal)
cell_counts = pd.DataFrame(cell_counts)

# eligible units per disease stage and the number that goes to training (75%, rounded down)
tumor_stats = cell_counts[cell_counts["tumor"] >= 500].value_counts("disease_status")
donor_stats = cell_counts[(cell_counts["disease_status"] == "Donor") & (cell_counts["normal"] >= 500)].value_counts("disease_status")
train_cutoff = dict(tumor_stats * 0.75)
train_cutoff.update(dict(donor_stats * 0.75))
train_cutoff = {ds: int(train_cutoff[ds]) for ds in train_cutoff}
print("training units per disease stage:", train_cutoff)

# ---- 4. random split of the eligible units ------------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)

min_cell = 500

train_df = []
test_df = []
leftover_df = []
for x, df in cell_counts.groupby(["disease_status"]):
    if x == "Donor":
        samp = df[df["normal"] >= min_cell].copy().reset_index(drop=True)
        leftover_df.append(df[df["normal"] < min_cell].copy())
    else:
        samp = df[df["tumor"] >= min_cell].copy().reset_index(drop=True)
        leftover_df.append(df[df["tumor"] < min_cell].copy())

    idx = list(samp.index)
    random.shuffle(idx)

    train_df.append(samp.loc[idx[:train_cutoff[x]]].copy())
    test_df.append(samp.loc[idx[train_cutoff[x]:]].copy())

train_df = pd.concat(train_df).reset_index(drop=True)
test_df = pd.concat(test_df).reset_index(drop=True)
leftover_df = pd.concat(leftover_df).reset_index(drop=True)

train_df.to_csv(out(*DATASETS, "sample_info", "train_samples.csv"))
test_df.to_csv(out(*DATASETS, "sample_info", "test_samples.csv"))
leftover_df.to_csv(out(*DATASETS, "sample_info", "below_cutoff_samples.csv"))


# ---- 5. cells of each set ---------------------------------------------------------------------------------------------
def check_table(x, key):
    """whether the unit (individual, disease stage) is listed in table x"""
    flag = False
    if key[0] in list(x["redcap_id"]):
        if key[1] in list(x.loc[key[0] == x["redcap_id"], "disease_status"]):
            flag = True
    return flag


train_expr = []
test_expr = []
leftover_expr = []
for key, df in subset_expr.groupby(["redcap_id", "disease_status"]):
    if check_table(train_df, key):
        train_expr.append(df.copy())
    elif check_table(test_df, key):
        test_expr.append(df.copy())
    elif check_table(leftover_df, key):
        leftover_expr.append(df.copy())
    else:
        print("unit missing from the sample tables")

train_expr = pd.concat(train_expr)
test_expr = pd.concat(test_expr)
leftover_expr = pd.concat(leftover_expr)
print("units: %d training, %d test, %d under the cut-off" % (train_df.shape[0], test_df.shape[0], leftover_df.shape[0]))

train_expr.to_pickle(out(*DATASETS, "sc-train-expr_134samples.pkl"))
test_expr.to_pickle(out(*DATASETS, "sc-test-expr_47samples.pkl"))
leftover_expr.to_pickle(out(*DATASETS, "sc-below-cutoff-extra-expr_42samples.pkl"))


# ---- 6. mean malignant-PC features per unit ----------------------------------------------------------------------------
def create_tumor_pseudobulk(mat, cols):
    mat["group_id"] = np.array(mat["redcap_id"]) + ":" + np.array(mat["disease_status"])
    p = pd.DataFrame()
    for group_id, x in mat.groupby("group_id"):
        p[group_id] = x[cols].mean()
    p = p.T
    p["redcap_id"] = [i.split(":")[0] for i in p.index]
    p["disease_status"] = [i.split(":")[1] for i in p.index]
    return p


featall_names = (mean_module_cols + mean_de_cols + auc_module_cols + auc_de_cols + mean_aging_cols + auc_aging_cols +
                 mean_cnv_arms + scanpy_cnv_arms + cyto_gene_cols)

create_tumor_pseudobulk(train_expr[train_expr["annotation"] == "tumor"], featall_names).to_pickle(
    out(*DATASETS, "pseudo-mean-tumor-train-expr_134samples.pkl"))
create_tumor_pseudobulk(test_expr[test_expr["annotation"] == "tumor"], featall_names).to_pickle(
    out(*DATASETS, "pseudo-mean-tumor-test-expr_47samples.pkl"))
create_tumor_pseudobulk(leftover_expr[leftover_expr["annotation"] == "tumor"], featall_names).to_pickle(
    out(*DATASETS, "pseudo-mean-tumor-below-cutoff-extra-expr_42samples.pkl"))
print("done")
