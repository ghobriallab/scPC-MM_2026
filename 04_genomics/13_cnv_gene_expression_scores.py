#!/usr/bin/env python3
"""Copy-number-like expression scores of the driver arms 1p, 1q, 13q, 14q and 17p in every cell, used as classifier
features, and the expression of the translocation partner genes (Methods, Development of malignancy and MST scores).

Purpose:      Gene sets from 46 BM pseudobulk DESeq2 tables (tumor vs matched normal 6, stage tumor vs other normal 30,
              tumor vs tumor 10; files selected by name, pooled SMM comparisons left out). Control genes: genes of
              chromosomes 2, 4, 10, 12 and 20 tested in all 46 and never differentially expressed (padj <= 0.01 and
              |log2FC| >= 1). Arm genes: genes of the arm (genes.bed column 7) tested in >= 3 of these 46 and the 16
              normal vs normal and normal vs donor tables. In every cell (log1p expression): scanpy
              score_genes of each arm's genes against the control genes (gene_pool), mean expression of each gene set
              and its difference to the control mean (log fold change). The same with all genes of each arm and of the
              control chromosomes (full scores). Also the expression of CCND1, CCND2, CCND3, MAF, MAFA, MAFB, MYC, NSD2
              and FGFR3. Must be run with PYTHONHASHSEED=0.

Inputs:       data/pseudobulk/result/GEX5_only/{tumor_vs_normal, Stage_tumor_vs_Other_normal, tumor_vs_tumor,
              normal_vs_normal, normal_vs_donor}/BM_* (DESeq2 tables: gene index, log2FoldChange, padj).
              data/reference/refdata-gex-GRCh38-2020-A/genes/genes.bed: chromosome (column 1), gene (column 4), arm
              (column 7).
              data/h5/agg/annotated-cell-log1p-240327.h5ad: log1p expression of all cells.

Outputs:      data/signature/gene_list/expr_cnv/: control_gene_list.txt, del1p_gene_list.txt, amp1q_gene_list.txt,
              del13q_gene_list.txt, del14q_gene_list.txt, del17p_gene_list.txt.
              data/signature/score/cell-level-copy-number-gene-expr.pkl: scanpy:<arm>_score, mean_expr:<arm>_log_fc,
              mean_expr:<arm>_genes and mean_expr:control_genes per cell.
              data/signature/score/cell-level-copy-number-full-gene-expr.pkl: the same with all genes (_full_).
              data/signature/score/cell-level-cytogenetics-gene-expr.pkl: expression of the nine partner genes per cell.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy); ls; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import shlex
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc

DE = data("pseudobulk", "result", "GEX5_only")
GENE_LISTS = data("signature", "gene_list", "expr_cnv")
GENE_LISTS.mkdir(parents=True, exist_ok=True)


def ls(folder, pattern):
    """Files of a DE folder matching a shell pattern, as listed by ls."""
    out = subprocess.check_output("ls %s/%s" % (shlex.quote(str(DE / folder)), pattern), shell=True)
    return out.decode("utf-8").rstrip("\n").split("\n")


# ---- 1. DE tables ----------------------------------------------------------------------------------------------------
# tumor vs normal
de_tn_match = ls("tumor_vs_normal", "BM_[MLIHt]*")
# tumor vs other normal
de_tn_stage = [i for i in ls("Stage_tumor_vs_Other_normal", "BM_*") if not "_SMM_" in i]
# tumor vs tumor
de_tt = [i for i in ls("tumor_vs_tumor", "BM_[MLIH]*") if not "_SMM_" in i]
de_files = de_tn_match + de_tn_stage + de_tt

# normal vs normal and normal vs donor (arm genes only)
de_nn = [i for i in ls("normal_vs_normal", "BM_[MLIHn]*") if not "_SMM_" in i]
de_nd = [i for i in ls("normal_vs_donor", "BM_[MLIHn]*") if not "_SMM_" in i]

# ---- 2. control and arm gene lists -----------------------------------------------------------------------------------
gene_bed = pd.read_csv(data("reference", "refdata-gex-GRCh38-2020-A", "genes", "genes.bed"), sep="\t", header=None)


def create_gene_list(gene_bed, chroms, de_files, fdr_cutoff=0.01, lfc_cutoff=1):
    """Per gene of the given chromosomes: number of DE tables testing it (analyzed) and calling it DE."""
    chr_genes = gene_bed.loc[gene_bed[0].isin(chroms), [0, 3]].drop_duplicates()
    chr_genes = chr_genes.rename(columns={0: "chr", 3: "gene"}).set_index("gene")
    chr_genes["analyzed"] = 0
    chr_genes["DE"] = 0

    for in_file in de_files:
        de_df = pd.read_csv(in_file, index_col=0)
        de_df = de_df[de_df.index.isin(chr_genes.index)]
        # genes tested
        chr_genes.loc[chr_genes.index.isin(de_df.index), "analyzed"] += 1
        # DE genes
        de_genes = de_df[(np.abs(de_df["log2FoldChange"]) >= lfc_cutoff) & (de_df["padj"] <= 0.01)].index
        chr_genes.loc[chr_genes.index.isin(de_genes), "DE"] += 1

    return chr_genes


def write_gene_list(genes, output_file):
    fo = open(output_file, "w")
    for g in sorted(genes):
        fo.write(g + "\n")
    fo.close()


# control genes: tested in all 46 comparisons and never DE
control_chr_genes = create_gene_list(gene_bed, ["chr2", "chr4", "chr10", "chr12", "chr20"], de_files)
control_genes = list(control_chr_genes[(control_chr_genes["DE"] == 0) & (control_chr_genes["analyzed"] == 46)].index)
write_gene_list(control_genes, GENE_LISTS / "control_gene_list.txt")

# arm genes: tested in at least 3 comparisons
analyzed_cutoff = 3

df = gene_bed[gene_bed[6] == "chr1p"].copy()
chr1p_genes_df = create_gene_list(df, ["chr1"], de_files + de_nn + de_nd)
chr1p_genes = list(chr1p_genes_df[chr1p_genes_df["analyzed"] >= analyzed_cutoff].index)
print("Del1p: usable={} / all={}".format(len(chr1p_genes), len(df[3].unique())))
write_gene_list(chr1p_genes, GENE_LISTS / "del1p_gene_list.txt")

df = gene_bed[gene_bed[6] == "chr1q"].copy()
chr1q_genes_df = create_gene_list(df, ["chr1"], de_files + de_nn + de_nd)
chr1q_genes = list(chr1q_genes_df[chr1q_genes_df["analyzed"] >= analyzed_cutoff].index)
print("Amp1q: usable={} / all={}".format(len(chr1q_genes), len(df[3].unique())))
write_gene_list(chr1q_genes, GENE_LISTS / "amp1q_gene_list.txt")

df = gene_bed[gene_bed[6] == "chr13q"].copy()
chr13q_genes_df = create_gene_list(df, ["chr13"], de_files + de_nn + de_nd)
chr13q_genes = list(chr13q_genes_df[chr13q_genes_df["analyzed"] >= analyzed_cutoff].index)
print("Del13q: usable={} / all={}".format(len(chr13q_genes), len(df[3].unique())))
write_gene_list(chr13q_genes, GENE_LISTS / "del13q_gene_list.txt")

df = gene_bed[gene_bed[6] == "chr14q"].copy()
chr14q_genes_df = create_gene_list(df, ["chr14"], de_files + de_nn + de_nd)
chr14q_genes = list(chr14q_genes_df[chr14q_genes_df["analyzed"] >= analyzed_cutoff].index)
print("Del14q: usable={} / all={}".format(len(chr14q_genes), len(df[3].unique())))
write_gene_list(chr14q_genes, GENE_LISTS / "del14q_gene_list.txt")

df = gene_bed[gene_bed[6] == "chr17p"].copy()
chr17p_genes_df = create_gene_list(df, ["chr17"], de_files + de_nn + de_nd)
chr17p_genes = list(chr17p_genes_df[chr17p_genes_df["analyzed"] >= analyzed_cutoff].index)
print("Del17p: usable={} / all={}".format(len(chr17p_genes), len(df[3].unique())))
write_gene_list(chr17p_genes, GENE_LISTS / "del17p_gene_list.txt")

# ---- 3. scores from the selected genes -------------------------------------------------------------------------------
adata = sc.read(str(data("h5", "agg", "annotated-cell-log1p-240327.h5ad")))

sc.tl.score_genes(adata, chr1p_genes, score_name="scanpy:1p_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr1q_genes, score_name="scanpy:1q_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr13q_genes, score_name="scanpy:13q_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr14q_genes, score_name="scanpy:14q_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr17p_genes, score_name="scanpy:17p_score", use_raw=False, gene_pool=control_genes)

scanpy_scores = ["scanpy:1p_score", "scanpy:1q_score", "scanpy:13q_score", "scanpy:14q_score", "scanpy:17p_score"]

expr = sc.get.obs_df(adata, keys=control_genes + chr1p_genes + chr1q_genes + chr13q_genes + chr14q_genes +
                     chr17p_genes + scanpy_scores)

# mean expression of each gene set
expr["mean_expr:control_genes"] = expr[control_genes].mean(axis=1)
expr["mean_expr:1p_genes"] = expr[chr1p_genes].mean(axis=1)
expr["mean_expr:1q_genes"] = expr[chr1q_genes].mean(axis=1)
expr["mean_expr:13q_genes"] = expr[chr13q_genes].mean(axis=1)
expr["mean_expr:14q_genes"] = expr[chr14q_genes].mean(axis=1)
expr["mean_expr:17p_genes"] = expr[chr17p_genes].mean(axis=1)

# difference to the control genes (log fold change)
expr["mean_expr:1p_log_fc"] = expr["mean_expr:1p_genes"] - expr["mean_expr:control_genes"]
expr["mean_expr:1q_log_fc"] = expr["mean_expr:1q_genes"] - expr["mean_expr:control_genes"]
expr["mean_expr:13q_log_fc"] = expr["mean_expr:13q_genes"] - expr["mean_expr:control_genes"]
expr["mean_expr:14q_log_fc"] = expr["mean_expr:14q_genes"] - expr["mean_expr:control_genes"]
expr["mean_expr:17p_log_fc"] = expr["mean_expr:17p_genes"] - expr["mean_expr:control_genes"]

mean_expr_columns = ["mean_expr:1p_genes", "mean_expr:1q_genes", "mean_expr:13q_genes",
                     "mean_expr:14q_genes", "mean_expr:17p_genes", "mean_expr:control_genes"]

log_fc_columns = ["mean_expr:1p_log_fc", "mean_expr:1q_log_fc", "mean_expr:13q_log_fc",
                  "mean_expr:14q_log_fc", "mean_expr:17p_log_fc"]

expr = expr[scanpy_scores + log_fc_columns + mean_expr_columns]
expr.to_pickle(data("signature", "score", "cell-level-copy-number-gene-expr.pkl"))

# ---- 4. scores from all genes of each arm ----------------------------------------------------------------------------
control_genes = list(gene_bed.loc[gene_bed[0].isin(["chr2", "chr4", "chr10", "chr12", "chr20"]), 3])

chr1p_genes = list(gene_bed.loc[gene_bed[6] == "chr1p", 3])
chr1q_genes = list(gene_bed.loc[gene_bed[6] == "chr1q", 3])
chr13q_genes = list(gene_bed.loc[gene_bed[6] == "chr13q", 3])
chr14q_genes = list(gene_bed.loc[gene_bed[6] == "chr14q", 3])
chr17p_genes = list(gene_bed.loc[gene_bed[6] == "chr17p", 3])

sc.tl.score_genes(adata, chr1p_genes, score_name="scanpy:1p_full_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr1q_genes, score_name="scanpy:1q_full_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr13q_genes, score_name="scanpy:13q_full_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr14q_genes, score_name="scanpy:14q_full_score", use_raw=False, gene_pool=control_genes)
sc.tl.score_genes(adata, chr17p_genes, score_name="scanpy:17p_full_score", use_raw=False, gene_pool=control_genes)

adata.obs["mean_expr:control_full_genes"] = np.array(adata[:, control_genes].X.mean(axis=1)).reshape(-1)
adata.obs["mean_expr:1p_full_genes"] = np.array(adata[:, chr1p_genes].X.mean(axis=1)).reshape(-1)
adata.obs["mean_expr:1q_full_genes"] = np.array(adata[:, chr1q_genes].X.mean(axis=1)).reshape(-1)
adata.obs["mean_expr:13q_full_genes"] = np.array(adata[:, chr13q_genes].X.mean(axis=1)).reshape(-1)
adata.obs["mean_expr:14q_full_genes"] = np.array(adata[:, chr14q_genes].X.mean(axis=1)).reshape(-1)
adata.obs["mean_expr:17p_full_genes"] = np.array(adata[:, chr17p_genes].X.mean(axis=1)).reshape(-1)

adata.obs["mean_expr:1p_full_log_fc"] = adata.obs["mean_expr:1p_full_genes"] - adata.obs["mean_expr:control_full_genes"]
adata.obs["mean_expr:1q_full_log_fc"] = adata.obs["mean_expr:1q_full_genes"] - adata.obs["mean_expr:control_full_genes"]
adata.obs["mean_expr:13q_full_log_fc"] = adata.obs["mean_expr:13q_full_genes"] - \
    adata.obs["mean_expr:control_full_genes"]
adata.obs["mean_expr:14q_full_log_fc"] = adata.obs["mean_expr:14q_full_genes"] - \
    adata.obs["mean_expr:control_full_genes"]
adata.obs["mean_expr:17p_full_log_fc"] = adata.obs["mean_expr:17p_full_genes"] - \
    adata.obs["mean_expr:control_full_genes"]

scanpy_scores = ["scanpy:1p_full_score", "scanpy:1q_full_score", "scanpy:13q_full_score",
                 "scanpy:14q_full_score", "scanpy:17p_full_score"]

mean_expr_columns = ["mean_expr:1p_full_genes", "mean_expr:1q_full_genes", "mean_expr:13q_full_genes",
                     "mean_expr:14q_full_genes", "mean_expr:17p_full_genes", "mean_expr:control_full_genes"]

log_fc_columns = ["mean_expr:1p_full_log_fc", "mean_expr:1q_full_log_fc", "mean_expr:13q_full_log_fc",
                  "mean_expr:14q_full_log_fc", "mean_expr:17p_full_log_fc"]

expr = sc.get.obs_df(adata, keys=scanpy_scores + log_fc_columns + mean_expr_columns)
expr.to_pickle(data("signature", "score", "cell-level-copy-number-full-gene-expr.pkl"))

# ---- 5. expression of the translocation partner genes ----------------------------------------------------------------
cyto_expr = sc.get.obs_df(adata, keys=["CCND1", "CCND2", "CCND3", "MAF", "MAFA", "MAFB", "MYC", "NSD2", "FGFR3"])
cyto_expr.to_pickle(data("signature", "score", "cell-level-cytogenetics-gene-expr.pkl"))
