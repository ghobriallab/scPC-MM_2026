#!/usr/bin/env python3
"""Signature features of the cells of the public single-cell datasets: mean expression and AUCell scores, inputs of the
malignancy and MST scores (Methods, calculation of malignancy and MST scores on external single-cell data).

Purpose:      For every cell: mean log-normalized expression and AUCell score (pySCENIC aucell, normalize=True, default
              AUC threshold 0.05) of each of the 16 signatures, the six stage-wise DE gene sets and the SenMayo senescence
              set, over the genes of each set measured in the dataset (missing genes are printed). One output per
              dataset: the datasets named on the command line, or all of them. The script restarts itself with
              PYTHONHASHSEED=0 when that variable is not set.

Inputs:       data/h5/external/<dataset folder>/log1p_count.h5ad: log-normalized counts (X) of each dataset script.
              data/signature/hotspot/consensus-modules_genes.csv: gene (index), context (signature).
              data/signature/gene_list/cross_tvn_de/: early_upreg_111genes.txt, smm-on_upreg_116genes.txt,
              late_upreg_74genes.txt, early_downreg_36genes.txt, late_downreg_74genes.txt, smm-on_downreg_66genes.txt.
              data/pathway/external/senmayo_Saul_etal_NatCommun_2022.txt: SenMayo genes.

Outputs:      data/signature/external/raw/expr_score/<dataset>_expr_score.pkl: per cell, mean_expr:<set> and auc:<set>.

Dependencies: Python 3.8 with requirements.txt (pandas, scanpy); pySCENIC (aucell) and ctxcore; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

if not os.getenv("PYTHONHASHSEED"):                                     # restart with PYTHONHASHSEED=0
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse

import pandas as pd
import scanpy as sc
from ctxcore.genesig import GeneSignature
from pyscenic.aucell import aucell

DATASETS = {  # output name: folder under data/h5/external
    "boiarsky_natcommun2022": "Boiarsky_NatCommun2022",
    "ledergor_natmedicine2018": "CA3Data_Ledergor2018",
    "dang_cancercell2023": "Dang_CancerCell2023",
}
GENE_LISTS = data("signature", "gene_list", "cross_tvn_de")


def load_gene_list(input_file):
    return [x.rstrip() for x in open(input_file)]


def out(*parts):
    """an output file in the processed-data tree; its folder is created if needed"""
    p = data(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main(dataset):
    a = sc.read(data("h5", "external", DATASETS[dataset], "log1p_count.h5ad"))
    gene_module = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
    gene_lists = {
        "MGUS_UP": load_gene_list(GENE_LISTS / "early_upreg_111genes.txt"),
        "SMM_UP": load_gene_list(GENE_LISTS / "smm-on_upreg_116genes.txt"),
        "MM_UP": load_gene_list(GENE_LISTS / "late_upreg_74genes.txt"),
        "MGUS_DN": load_gene_list(GENE_LISTS / "early_downreg_36genes.txt"),
        "SMM_DN": load_gene_list(GENE_LISTS / "late_downreg_74genes.txt"),
        "MM_DN": load_gene_list(GENE_LISTS / "smm-on_downreg_66genes.txt"),
        "SenMayo": load_gene_list(data("pathway", "external", "senmayo_Saul_etal_NatCommun_2022.txt")),
    }

    # mean expression over the measured genes of each set, and the gene signatures for AUCell
    gene_sigs = []
    expr_score = pd.DataFrame(index=a.obs_names)
    input_var_space = set(a.var_names)

    # first the 16 signatures
    for name, members in gene_module.groupby("context"):
        genes = list(members.index)
        valid_genes = list(input_var_space.intersection(set(genes)))
        missing_genes = set(genes).difference(input_var_space)
        print("{}: {} = missed: {}".format(dataset, name, ",".join(missing_genes)))
        gene_sigs.append(GeneSignature(name, valid_genes))
        x = sc.get.obs_df(a, keys=valid_genes)
        expr_score["mean_expr:{}".format(name)] = x.mean(axis=1)

    # then the stage-wise DE gene sets and SenMayo
    for name in gene_lists:
        genes = gene_lists[name]
        valid_genes = list(input_var_space.intersection(set(genes)))
        missing_genes = set(genes).difference(input_var_space)
        print("{}: {} = missed: {}".format(dataset, name, ",".join(missing_genes)))
        gene_sigs.append(GeneSignature(name, valid_genes))
        x = sc.get.obs_df(a, keys=valid_genes)
        expr_score["mean_expr:{}".format(name)] = x.mean(axis=1)

    # AUCell on the dense log-normalized matrix (cells x genes)
    exp_mtx = pd.DataFrame(a.X.toarray(), index=a.obs_names, columns=a.var_names)
    auc_score = aucell(exp_mtx, gene_sigs, normalize=True)
    auc_score = auc_score.rename(columns={c: "auc:{}".format(c) for c in auc_score.columns})

    expr_score = expr_score.join(auc_score)
    expr_score.to_pickle(out("signature", "external", "raw", "expr_score", "{}_expr_score.pkl".format(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="signature features of the cells of the public single-cell datasets")
    parser.add_argument("datasets", nargs="*", help="datasets to process (default: all): " + ", ".join(DATASETS))
    args = parser.parse_args()
    for d in args.datasets:
        if d not in DATASETS:
            parser.error("unknown dataset: " + d)
    for d in args.datasets or list(DATASETS):
        main(d)
