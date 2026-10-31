#!/usr/bin/env python3
"""Per-cell activity of the published gene sets used by the classifiers, one individual at a time: mean and sum of the
log-normalized expression of their genes, and AUCell scores (Methods, Development of malignancy and MST scores).

Purpose:      For the PCs of one individual and each gene set: mean and sum of the log-normalized expression of its genes,
              then AUCell on the cells x genes matrix (pySCENIC aucell, normalize=True, other arguments at their
              defaults: AUC threshold 0.05 and seed None, so ties in the gene rankings are broken at random). Gene sets:
              senmayo (SenMayo senescence genes, a classifier feature), nfkb (NF-kB index, the sensitivity analysis of
              the MGUS-MM classifier) and ras (KRAS and NRAS signatures, candidate classifier features). senmayo is
              scored on the cells of the individual in the full annotated object (individuals: every redcap_id of its
              obs); the others on the individual's own object (individuals: every redcap_id of the obs of
              plasma-cell-GEX5-log1p-250207.h5ad). Usage: 12_published_signature_scores.py [study [redcap_id ...]];
              without arguments every study and every individual is scored in turn. Runs with PYTHONHASHSEED=0
              (restarts itself with it if unset).

Inputs:       data/h5/agg/annotated-cell-log1p-240327.h5ad (senmayo) and data/h5/agg/pat/<redcap_id>.h5ad (nfkb, ras;
              "/" in the ID replaced by "-"): log-normalized expression; obs redcap_id.
              data/h5/agg/plasma-cell-GEX5-log1p-250207.h5ad: obs redcap_id (individuals of the per-individual objects).
              data/pathway/external/: senmayo_Saul_etal_NatCommun_2022.txt, nfkb_index_Annunziata_etal_CancerCell2007.txt
              (one gene per line), ras_signature_Lin_etal_Blood2019.csv (index = gene, column context; written by
              11_published_gene_lists.py).

Outputs:      data/signature/score/<folder>/<redcap_id><suffix>: one row per cell; mean_expr:<set>, sum_expr:<set>,
              auc:<set>. Folder and suffix: aging .cohort.aging.csv, nfkb .out.csv, ras_signature .ras-out.csv.

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

EXT = data("pathway", "external")


def load_gene_list(input_file):
    return [x.rstrip() for x in open(input_file)]


def context_sets(file_name):
    """one gene set per value of context; index = gene"""
    sigs = pd.read_csv(EXT / file_name, index_col=0)
    gene_lists = {}
    for context, df in sigs.groupby("context"):
        gene_lists[context] = list(df.index.unique())
    return gene_lists


def senmayo_sets():
    return {"SenMayo": load_gene_list(EXT / "senmayo_Saul_etal_NatCommun_2022.txt")}


def nfkb_sets():
    return {"NFkB_index": load_gene_list(EXT / "nfkb_index_Annunziata_etal_CancerCell2007.txt")}


# study: (expression input, gene sets, output folder, output suffix)
STUDIES = {
    "senmayo": ("cohort", senmayo_sets, "aging", ".cohort.aging.csv"),
    "nfkb": ("individual", nfkb_sets, "nfkb", ".out.csv"),
    "ras": ("individual", lambda: context_sets("ras_signature_Lin_etal_Blood2019.csv"), "ras_signature", ".ras-out.csv"),
}


def score_cells(a, gene_lists):
    """mean and sum of expression and AUCell score of each gene set, for the cells of a"""
    gene_sigs = []
    expr_score = pd.DataFrame(index=a.obs_names)
    for name in gene_lists:
        genes = gene_lists[name]
        gene_sigs.append(GeneSignature(name, genes))
        x = sc.get.obs_df(a, keys=genes)
        expr_score["mean_expr:{}".format(name)] = x.mean(axis=1)
        expr_score["sum_expr:{}".format(name)] = x.sum(axis=1)
    exp_mtx = pd.DataFrame(a.X.toarray(), index=a.obs_names, columns=a.var_names)
    auc_score = aucell(exp_mtx, gene_sigs, normalize=True)
    auc_score = auc_score.rename(columns={c: "auc:{}".format(c) for c in auc_score.columns})
    return expr_score.join(auc_score)


def run(study, patients):
    source, gene_sets, folder, suffix = STUDIES[study]
    gene_lists = gene_sets()
    out = data("signature", "score", folder)
    os.makedirs(out, exist_ok=True)
    if source == "cohort":
        # cells of each individual in the full annotated object
        a = sc.read(data("h5", "agg", "annotated-cell-log1p-240327.h5ad"))
        for pid in patients or sorted(a.obs["redcap_id"].astype(str).unique()):
            cells = a[a.obs["redcap_id"] == pid]
            score_cells(cells, gene_lists).to_csv(out / (pid.replace("/", "-") + suffix))
    else:
        # each individual's own object
        if not patients:
            obs = sc.read(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad"), backed="r").obs
            patients = sorted(obs["redcap_id"].astype(str).unique())
        for pid in patients:
            name = pid.replace("/", "-")
            a = sc.read(data("h5", "agg", "pat", name + ".h5ad"))
            score_cells(a, gene_lists).to_csv(out / (name + suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="published signature scores of the cells of each individual: mean and "
                                                 "sum of expression, and AUCell")
    parser.add_argument("study", nargs="?", choices=list(STUDIES), help="study to score (default: all)")
    parser.add_argument("redcap_id", nargs="*", help="individuals to score (default: all)")
    args = parser.parse_args()
    for study in [args.study] if args.study else list(STUDIES):
        run(study, args.redcap_id)
