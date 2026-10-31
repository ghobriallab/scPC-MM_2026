#!/usr/bin/env python3
"""Per-cell activity of the 16 signatures and of six stage-specific DEG lists, one individual at a time: mean and sum of
the log-normalized expression of their genes, and AUCell scores (Methods, signature activity).

Purpose:      For the cells of one individual (redcap_id) of the expression object: mean and sum of the log-normalized
              expression of the genes of each signature (consensus signature table) and of each of six stage-specific DEG
              lists, then AUCell on the cells x genes matrix of that individual with the same 22 gene sets (pySCENIC
              aucell, normalize=True, other arguments at their defaults: AUC threshold 0.05 and seed None, so ties in the
              gene rankings are broken at random). DEG list names: MGUS_UP, SMM_UP, MM_UP and MGUS_DN; SMM_DN is the late
              down-regulated list and MM_DN the list down-regulated from SMM on. One table per individual. Without
              arguments, every redcap_id of the expression object is scored in turn; with redcap_id arguments, only those
              (one job per individual can run in parallel). Runs with PYTHONHASHSEED=0 (restarts itself with it if unset).

Inputs:       data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad: X = log-normalized expression; obs redcap_id.
              data/signature/hotspot/consensus-modules_genes.csv (05_consensus_signatures.py): index = gene; context.
              data/signature/gene_list/cross_tvn_de/: early_upreg_111genes.txt, smm-on_upreg_116genes.txt,
              late_upreg_74genes.txt, early_downreg_36genes.txt, late_downreg_74genes.txt, smm-on_downreg_66genes.txt.

Outputs:      data/signature/score/cohort/<redcap_id>.cohort.score.csv: one row per cell; mean_expr:<set>,
              sum_expr:<set> and auc:<set> for the 22 gene sets ("/" in the ID replaced by "-").

Dependencies: Python 3.8 with requirements.txt (pandas, scanpy, anndata), pySCENIC and ctxcore (aucell, GeneSignature);
              config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

if not os.getenv("PYTHONHASHSEED"):
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse

import pandas as pd
import scanpy as sc
from ctxcore.genesig import GeneSignature
from pyscenic.aucell import aucell

DE = data("signature", "gene_list", "cross_tvn_de")
OUT = data("signature", "score", "cohort")


def load_gene_list(input_file):
    return [x.rstrip() for x in open(input_file)]


def score_cells(a, gene_module, gene_lists):
    """mean and sum of expression and AUCell score of each gene set, for the cells of a"""
    gene_sigs = []
    expr_score = pd.DataFrame(index=a.obs_names)
    # signatures first
    for name, members in gene_module.groupby("context"):
        genes = list(members.index)
        gene_sigs.append(GeneSignature(name, genes))
        x = sc.get.obs_df(a, keys=genes)
        expr_score["mean_expr:{}".format(name)] = x.mean(axis=1)
        expr_score["sum_expr:{}".format(name)] = x.sum(axis=1)
    # then the stage-specific DEG lists
    for name in gene_lists:
        genes = gene_lists[name]
        gene_sigs.append(GeneSignature(name, genes))
        x = sc.get.obs_df(a, keys=genes)
        expr_score["mean_expr:{}".format(name)] = x.mean(axis=1)
        expr_score["sum_expr:{}".format(name)] = x.sum(axis=1)
    # AUCell on the cells x genes matrix
    exp_mtx = pd.DataFrame(a.X.toarray(), index=a.obs_names, columns=a.var_names)
    auc_score = aucell(exp_mtx, gene_sigs, normalize=True)
    auc_score = auc_score.rename(columns={c: "auc:{}".format(c) for c in auc_score.columns})
    return expr_score.join(auc_score)


def main(patients):
    a = sc.read(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad"))
    gene_module = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
    gene_lists = {
        "MGUS_UP": load_gene_list(DE / "early_upreg_111genes.txt"),
        "SMM_UP": load_gene_list(DE / "smm-on_upreg_116genes.txt"),
        "MM_UP": load_gene_list(DE / "late_upreg_74genes.txt"),
        "MGUS_DN": load_gene_list(DE / "early_downreg_36genes.txt"),
        "SMM_DN": load_gene_list(DE / "late_downreg_74genes.txt"),
        "MM_DN": load_gene_list(DE / "smm-on_downreg_66genes.txt"),
    }
    os.makedirs(OUT, exist_ok=True)
    for pid in patients or sorted(a.obs["redcap_id"].astype(str).unique()):
        cells = a[a.obs["redcap_id"] == pid]
        score_cells(cells, gene_module, gene_lists).to_csv(OUT / (pid.replace("/", "-") + ".cohort.score.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="signature scores of the cells of each individual: mean and sum of "
                                                 "expression, and AUCell")
    parser.add_argument("redcap_id", nargs="*", help="individuals to score (default: every individual of the object)")
    main(parser.parse_args().redcap_id)
