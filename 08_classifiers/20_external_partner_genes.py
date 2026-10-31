#!/usr/bin/env python3
"""Expression of the nine translocation partner genes in the cells of the public single-cell datasets, inputs of the
malignancy score (Methods, calculation of malignancy and MST scores on external single-cell data).

Purpose:      For every cell: log-normalized expression of the genes of the classifier's partner-gene list (CCND1, CCND2,
              CCND3, MAF, MAFA, MAFB, MYC, NSD2, FGFR3). One output per dataset: the datasets named on the command line,
              or all of them. Seeds 42 (Python random, numpy). The script restarts itself with PYTHONHASHSEED=0 when that
              variable is not set.

Inputs:       data/h5/external/<dataset folder>/log1p_count.h5ad: log-normalized counts (X) of each dataset script.
              data/classifier/datasets/feature_columns/cytogenetics_genes.txt: the nine genes.

Outputs:      data/signature/external/raw/cyto_gene/<dataset>_cyto.pkl: per cell, one column per gene.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import SEED, data

if not os.getenv("PYTHONHASHSEED"):                                     # restart with PYTHONHASHSEED=0
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import random

import numpy as np
import scanpy as sc

random.seed(SEED)
np.random.seed(SEED)

DATASETS = {  # output name: folder under data/h5/external
    "boiarsky_natcommun2022": "Boiarsky_NatCommun2022",
    "ledergor_natmedicine2018": "CA3Data_Ledergor2018",
    "dang_cancercell2023": "Dang_CancerCell2023",
}


def load_list(input_file):
    return [x.rstrip() for x in open(input_file)]


def out(*parts):
    """an output file in the processed-data tree; its folder is created if needed"""
    p = data(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main(dataset):
    a = sc.read(data("h5", "external", DATASETS[dataset], "log1p_count.h5ad"))
    cyto_genes = load_list(data("classifier", "datasets", "feature_columns", "cytogenetics_genes.txt"))
    cyto_expr = sc.get.obs_df(a, keys=cyto_genes)
    cyto_expr.to_pickle(out("signature", "external", "raw", "cyto_gene", "{}_cyto.pkl".format(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="partner-gene expression of the cells of the public single-cell datasets")
    parser.add_argument("datasets", nargs="*", help="datasets to process (default: all): " + ", ".join(DATASETS))
    args = parser.parse_args()
    for d in args.datasets:
        if d not in DATASETS:
            parser.error("unknown dataset: " + d)
    for d in args.datasets or list(DATASETS):
        main(d)
