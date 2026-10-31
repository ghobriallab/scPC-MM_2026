#!/usr/bin/env python3
"""Per-cell activity of the trimmed signatures (top 5, 10, 15 and 20 genes of each of the 16 signatures), one individual
at a time: mean log-normalized expression of their genes and AUCell scores (classifier feature sets).

Purpose:      For the PCs of one individual, from its own expression object: mean log-normalized expression of the genes
              of each of the 64 trimmed signatures (MTS5, MTS10, MTS15, MTS20) and AUCell on the cells x genes matrix
              (pySCENIC aucell, normalize=True, other arguments at their defaults: AUC threshold 0.05 and seed None, so ties
              in the gene rankings are broken at random). Without arguments, every redcap_id of the obs of
              data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad is scored in turn; with redcap_id arguments, only those (one
              job per individual can run in parallel). Runs with PYTHONHASHSEED=0 (restarts itself with it if unset).

Inputs:       data/h5/agg/pat/<redcap_id>.h5ad: log-normalized expression of the PCs of one individual ("/" in the ID
              replaced by "-").
              data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad: obs redcap_id (the individuals to score by default).
              data/signature/hotspot/trimmed-consensus-modules_genes.csv (07_aggregate_scores_and_trim_genes.py):
              index = gene; context = MTS<n>:<signature>.

Outputs:      data/signature/score/cohort_trimmed/<redcap_id>.trimmed-out.csv: one row per cell; mean_expr:<set> and
              auc:<set> for the 64 trimmed signatures.

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

OUT = data("signature", "score", "cohort_trimmed")


def score_cells(a, gene_lists):
    """mean expression and AUCell score of each trimmed signature, for the cells of a"""
    gene_sigs = []
    expr_score = pd.DataFrame(index=a.obs_names)
    for name in gene_lists:
        genes = gene_lists[name]
        gene_sigs.append(GeneSignature(name, genes))
        x = sc.get.obs_df(a, keys=genes)
        expr_score["mean_expr:{}".format(name)] = x.mean(axis=1)
    exp_mtx = pd.DataFrame(a.X.toarray(), index=a.obs_names, columns=a.var_names)
    auc_score = aucell(exp_mtx, gene_sigs, normalize=True)
    auc_score = auc_score.rename(columns={c: "auc:{}".format(c) for c in auc_score.columns})
    return expr_score.join(auc_score)


def main(patients):
    gene_module = pd.read_csv(data("signature", "hotspot", "trimmed-consensus-modules_genes.csv"), index_col=0)
    gene_lists = {}
    for context, df in gene_module.groupby("context"):
        gene_lists[context] = list(df.index.unique())
    if not patients:
        obs = sc.read(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad"), backed="r").obs
        patients = sorted(obs["redcap_id"].astype(str).unique())
    os.makedirs(OUT, exist_ok=True)
    for pid in patients:
        name = pid.replace("/", "-")
        a = sc.read(data("h5", "agg", "pat", name + ".h5ad"))
        score_cells(a, gene_lists).to_csv(OUT / (name + ".trimmed-out.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trimmed signature scores of the cells of each individual: mean "
                                                 "expression and AUCell")
    parser.add_argument("redcap_id", nargs="*", help="individuals to score (default: every individual of the object)")
    main(parser.parse_args().redcap_id)
