#!/usr/bin/env python3
"""CNV-arm features of the cells of the public single-cell datasets, inputs of the MST score (Methods, calculation of
malignancy and MST scores on external single-cell data).

Purpose:      For every cell: mean log-normalized expression of the genes of chromosome arms 1p, 1q, 13q, 14q and 17p
              minus the mean over the genes of the control chromosomes 2, 4, 10, 12 and 20 (genes measured in the
              dataset; zeros included). One output per dataset: the datasets named on the command line, or all of them.
              The script restarts itself with PYTHONHASHSEED=0 when that variable is not set.

Inputs:       data/h5/external/<dataset folder>/log1p_count.h5ad: log-normalized counts (X) of each dataset script.
              data/reference/refdata-gex-GRCh38-2020-A/genes/genes.bed: one gene per row, no header; column 1 chromosome,
              column 4 gene name, column 7 chromosome arm (chr1p, chr1q, ...).

Outputs:      data/signature/external/raw/cnv_score/<dataset>_cnv_score.pkl: per cell, mean_expr:<arm>_full_log_fc for
              1p, 1q, 13q, 14q and 17p.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

if not os.getenv("PYTHONHASHSEED"):                                     # restart with PYTHONHASHSEED=0
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse

import numpy as np
import pandas as pd
import scanpy as sc

DATASETS = {  # output name: folder under data/h5/external
    "boiarsky_natcommun2022": "Boiarsky_NatCommun2022",
    "ledergor_natmedicine2018": "CA3Data_Ledergor2018",
    "dang_cancercell2023": "Dang_CancerCell2023",
}


def create_gene_list(gene_list, var_space):
    return list(set(gene_list).intersection(set(var_space)))


def out(*parts):
    """an output file in the processed-data tree; its folder is created if needed"""
    p = data(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main(dataset):
    a = sc.read(data("h5", "external", DATASETS[dataset], "log1p_count.h5ad"))
    var_space = list(a.var_names)
    gene_bed = pd.read_csv(data("reference", "refdata-gex-GRCh38-2020-A", "genes", "genes.bed"), sep="\t", header=None)

    # measured genes of the control chromosomes and of each arm
    control = gene_bed.loc[gene_bed[0].isin(["chr2", "chr4", "chr10", "chr12", "chr20"]), 3].values
    control_genes = create_gene_list(list(control), var_space)
    chr1p_genes = create_gene_list(list(gene_bed.loc[gene_bed[6] == "chr1p", 3].values), var_space)
    chr1q_genes = create_gene_list(list(gene_bed.loc[gene_bed[6] == "chr1q", 3].values), var_space)
    chr13q_genes = create_gene_list(list(gene_bed.loc[gene_bed[6] == "chr13q", 3].values), var_space)
    chr14q_genes = create_gene_list(list(gene_bed.loc[gene_bed[6] == "chr14q", 3].values), var_space)
    chr17p_genes = create_gene_list(list(gene_bed.loc[gene_bed[6] == "chr17p", 3].values), var_space)

    # mean expression per cell, then arm minus control
    a.obs["mean_expr:control_full_genes"] = np.array(a[:, control_genes].X.mean(axis=1)).reshape(-1)
    a.obs["mean_expr:1p_full_genes"] = np.array(a[:, chr1p_genes].X.mean(axis=1)).reshape(-1)
    a.obs["mean_expr:1q_full_genes"] = np.array(a[:, chr1q_genes].X.mean(axis=1)).reshape(-1)
    a.obs["mean_expr:13q_full_genes"] = np.array(a[:, chr13q_genes].X.mean(axis=1)).reshape(-1)
    a.obs["mean_expr:14q_full_genes"] = np.array(a[:, chr14q_genes].X.mean(axis=1)).reshape(-1)
    a.obs["mean_expr:17p_full_genes"] = np.array(a[:, chr17p_genes].X.mean(axis=1)).reshape(-1)

    a.obs["mean_expr:1p_full_log_fc"] = a.obs["mean_expr:1p_full_genes"] - a.obs["mean_expr:control_full_genes"]
    a.obs["mean_expr:1q_full_log_fc"] = a.obs["mean_expr:1q_full_genes"] - a.obs["mean_expr:control_full_genes"]
    a.obs["mean_expr:13q_full_log_fc"] = a.obs["mean_expr:13q_full_genes"] - a.obs["mean_expr:control_full_genes"]
    a.obs["mean_expr:14q_full_log_fc"] = a.obs["mean_expr:14q_full_genes"] - a.obs["mean_expr:control_full_genes"]
    a.obs["mean_expr:17p_full_log_fc"] = a.obs["mean_expr:17p_full_genes"] - a.obs["mean_expr:control_full_genes"]

    log_fc_columns = ["mean_expr:1p_full_log_fc", "mean_expr:1q_full_log_fc", "mean_expr:13q_full_log_fc",
                      "mean_expr:14q_full_log_fc", "mean_expr:17p_full_log_fc"]
    expr = sc.get.obs_df(a, keys=log_fc_columns)
    expr.to_pickle(out("signature", "external", "raw", "cnv_score", "{}_cnv_score.pkl".format(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNV-arm features of the cells of the public single-cell datasets")
    parser.add_argument("datasets", nargs="*", help="datasets to process (default: all): " + ", ".join(DATASETS))
    args = parser.parse_args()
    for d in args.datasets:
        if d not in DATASETS:
            parser.error("unknown dataset: " + d)
    for d in args.datasets or list(DATASETS):
        main(d)
