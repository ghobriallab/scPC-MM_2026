#!/usr/bin/env python3
"""Gene lists of published myeloma signatures matched to the gene names of the expression data (CoMMpass RNA subtypes,
Zhan et al. subgroups, Boiarsky et al. programs, GEP70 and GEP5 risk scores, RAS signatures) and the table of the six
stage-specific DEG lists (Methods, scores of published signatures).

Purpose:      Stage DEG table: the six DEG lists with their names (here MM_DN is the late down-regulated list and SMM_DN
              the list down-regulated from SMM on). Zhan et al. 2006: genes with a curated single-cell name marked
              'match'. CoMMpass RNA subtypes (Skerget et al. 2024): genes named as in the expression data ('match') or
              matched by Ensembl ID through the 10x reference ('alias'); subtype names without spaces. Boiarsky et al.
              2022: genes absent from the expression data renamed through org.Hs.eg.db (symbol, then alias, to Ensembl ID)
              and the 10x reference, genes still absent dropped. GEP70 (Shaughnessy et al. 2007): Ensembl IDs converted to
              10x gene names; GEP5 (Heuck et al. 2014) as published. Zhan subgroups for scoring: genes with a single-cell
              name. RAS signatures (Lin et al. 2019): genes whose KRAS (or NRAS) classifier weight exceeds 3 times the
              standard deviation of that weight across genes, named through the 10x reference.

Inputs:       data/signature/gene_list/cross_tvn_de/: early_downreg_36genes.txt, early_upreg_111genes.txt,
              late_downreg_89genes.txt, late_upreg_84genes.txt, smm-on_downreg_45genes.txt, smm-on_upreg_98genes.txt.
              data/pathway/external/raw/: zhan_etal_Blood2006.csv (Gene, Subgroup, scRNA_ref),
              CoMMPass_12RnaSubtypes_NatGenetics2024.csv (index = Ensembl ID; Gene, Coefficient, RNA_Subtype_Name,
              RNA_Subtype_ID), myeloma_nmf_Boiarsky_etal_NatCommun2022.csv (index = gene), GEP_scores/
              GEP70_risk_score_Shaughnessy_etal_Blood2007.csv (gencode_id) and GEP5_risk_score_Heuck_etal_Leukemia2014.txt,
              Lin_Blood2019/ras_classifier_genes.csv (index = Ensembl ID; KRAS, NRAS weights).
              data/h5/agg/annotated-cell-log1p-240327.h5ad (gene names) and plasma-cell-GEX5-log1p-250207-var-names.txt.
              data/reference/refdata-gex-GRCh38-2020-A/genes/genes.bed (columns 4, 5: gene name, Ensembl ID) and
              data/reference/orgHsegdb-gene-map.txt (ENSEMBL, SYMBOL, ALIAS).

Outputs:      data/signature/gene_list/cross_tvn_de/disease_stage_stepwise_DEs.csv: index = gene; context.
              data/pathway/external/: zhan_etal_Blood2006.csv, CoMMPass_12RnaSubtypes_NatGenetics2024.csv,
              myeloma_nmf_Boiarsky_etal_NatCommun2022.csv, GEP_risk_score_Heuck-Shaughnessy_etal_Leukemia2014-Blood2007.csv,
              microarray_signature_Zhan_etal_Blood2006.csv, ras_signature_Lin_etal_Blood2019.csv.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy, anndata); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)
DE = data("signature", "gene_list", "cross_tvn_de")
EXT = data("pathway", "external")
RAW = EXT / "raw"
BED = data("reference", "refdata-gex-GRCh38-2020-A", "genes", "genes.bed")


def load_gene_list(input_file):
    return [x.rstrip() for x in open(input_file)]


# ---- stage-specific DEG lists in one table ------------------------------------------------------------------------------
input_paths = {
    "MGUS_DN": DE / "early_downreg_36genes.txt",
    "MGUS_UP": DE / "early_upreg_111genes.txt",
    "MM_DN": DE / "late_downreg_89genes.txt",
    "MM_UP": DE / "late_upreg_84genes.txt",
    "SMM_DN": DE / "smm-on_downreg_45genes.txt",
    "SMM_UP": DE / "smm-on_upreg_98genes.txt",
}
de_genes = {"gene": [], "context": []}
for gene_type in input_paths:
    genes = load_gene_list(input_paths[gene_type])
    de_genes["gene"].extend(genes)
    de_genes["context"].extend([gene_type for i in range(len(genes))])
pd.DataFrame(de_genes).set_index("gene").to_csv(DE / "disease_stage_stepwise_DEs.csv")

# ---- Zhan et al. subgroups: curated single-cell gene names ------------------------------------------------------------------
zhan_genes = pd.read_csv(RAW / "zhan_etal_Blood2006.csv")
zhan_genes.rename(columns={"Gene": "orig_gene_name", "scRNA_ref": "scRNA_gene_name"}, inplace=True)
zhan_genes["gene"] = list(zhan_genes["scRNA_gene_name"])
zhan_genes["scRNA_status"] = "n/a"
zhan_genes.loc[~zhan_genes["scRNA_gene_name"].isna(), "scRNA_status"] = "match"
zhan_genes[["gene", "Subgroup", "scRNA_status", "scRNA_gene_name", "orig_gene_name"]].to_csv(EXT / "zhan_etal_Blood2006.csv",
                                                                                              index=False)

# ---- CoMMpass RNA subtypes: gene names of the expression data, directly or by Ensembl ID -----------------------------------
a = sc.read(data("h5", "agg", "annotated-cell-log1p-240327.h5ad"), backed="r")
commpass_genes = pd.read_csv(RAW / "CoMMPass_12RnaSubtypes_NatGenetics2024.csv", index_col=0)
commpass_genes["scRNA_status"] = "n/a"
overlap_genes = a.var_names.intersection(set(commpass_genes["Gene"].unique()))
commpass_genes.loc[commpass_genes["Gene"].isin(overlap_genes), "scRNA_status"] = "match"
gene_bed = pd.read_csv(BED, sep="\t", header=None)
scrna_ids = gene_bed[[3, 4]].rename(columns={3: "gene", 4: "ENSG"}).drop_duplicates().set_index("ENSG")
commpass_genes["scRNA_gene_name"] = np.nan
overlap_ensg = commpass_genes.index.unique().intersection(scrna_ids.index)
for g in overlap_ensg:
    commpass_genes.loc[g, "scRNA_gene_name"] = scrna_ids.loc[g, "gene"]
commpass_genes.loc[~commpass_genes["scRNA_gene_name"].isna() & (commpass_genes["scRNA_status"] == "n/a"), "scRNA_status"] = "alias"
commpass_genes.rename(columns={"Gene": "orig_gene_name"}, inplace=True)
commpass_genes["gene"] = list(commpass_genes["scRNA_gene_name"])
commpass_genes = commpass_genes[["gene", "RNA_Subtype_Name", "RNA_Subtype_ID", "Coefficient", "scRNA_status", "scRNA_gene_name",
                                 "orig_gene_name"]]
commpass_genes["RNA_Subtype_Name"] = commpass_genes["RNA_Subtype_Name"].str.replace(" ", "_").str.replace(",_", ",")
commpass_genes.to_csv(EXT / "CoMMPass_12RnaSubtypes_NatGenetics2024.csv")
del a

# ---- Boiarsky et al. programs: absent genes renamed through org.Hs.eg.db and the 10x reference -----------------------------
sig = pd.read_csv(RAW / "myeloma_nmf_Boiarsky_etal_NatCommun2022.csv", index_col=0)
genes = set([g.rstrip() for g in open(data("h5", "agg", "plasma-cell-GEX5-log1p-250207-var-names.txt"))])
gene_map = pd.read_csv(data("reference", "orgHsegdb-gene-map.txt"), sep="\t")
gene_map = gene_map[["ENSEMBL", "SYMBOL", "ALIAS"]].drop_duplicates()
bed = pd.read_csv(BED, header=None, sep="\t")
gex_info = bed[[3, 4]].rename(columns={3: "gene", 4: "ensembl_id"})
rescued_index = {}
for g in sig.index:
    if g not in genes:
        # symbol first, then alias
        idx = gene_map["SYMBOL"] == g
        if sum(idx) == 0:
            idx = gene_map["ALIAS"] == g
        if sum(idx) > 0:
            rescued_index[g] = gene_map.loc[idx, "ENSEMBL"].values[0]
gene_alias = {}
for key in rescued_index.keys():
    idx = gex_info["ensembl_id"] == rescued_index[key]
    if sum(idx) == 0:
        continue
    gene_alias[key] = gex_info.loc[idx, "gene"].values[0]
sig.index = [gene_alias[g] if g in gene_alias else g for g in sig.index]
sig.loc[sig.index.isin(list(genes))].to_csv(EXT / "myeloma_nmf_Boiarsky_etal_NatCommun2022.csv")

# ---- GEP70 and GEP5 risk scores -------------------------------------------------------------------------------------------
gep = pd.read_csv(RAW / "GEP_scores" / "GEP70_risk_score_Shaughnessy_etal_Blood2007.csv", index_col=0)
gep70 = gep.loc[~gep["gencode_id"].isna(), "gencode_id"].values
gep5 = [g.rstrip() for g in open(RAW / "GEP_scores" / "GEP5_risk_score_Heuck_etal_Leukemia2014.txt")]
gep70_genes = []
for g in gep70:
    idx = gex_info["ensembl_id"] == g
    if sum(idx) == 0:
        continue
    gep70_genes.append(gex_info.loc[idx, "gene"].values[0])
pd.DataFrame({"gene": gep70_genes + gep5,
              "context": ["GEP70" for i in range(len(gep70_genes))] + ["GEP5" for i in range(len(gep5))]}).to_csv(
    EXT / "GEP_risk_score_Heuck-Shaughnessy_etal_Leukemia2014-Blood2007.csv", index=False)

# ---- Zhan et al. subgroups for scoring: genes with a single-cell name -----------------------------------------------------
sig = pd.read_csv(EXT / "zhan_etal_Blood2006.csv", index_col=0)
sig = sig.loc[~sig["scRNA_gene_name"].isna(), ["scRNA_gene_name", "Subgroup"]].reset_index(drop=True)
sig.rename(columns={"scRNA_gene_name": "gene", "Subgroup": "context"}, inplace=True)
sig = sig.set_index("gene")
sig.to_csv(EXT / "microarray_signature_Zhan_etal_Blood2006.csv")

# ---- RAS signatures: genes with a KRAS or NRAS weight above 3 standard deviations -------------------------------------------
ras = pd.read_csv(RAW / "Lin_Blood2019" / "ras_classifier_genes.csv", index_col=0)
gene_names = gex_info.set_index("ensembl_id").to_dict()["gene"]


def get_signature_genes(df, target_col, gene_dict, std_span=3):
    X = df.loc[df[target_col] > np.std(df[target_col]) * std_span].copy()
    X["gene"] = X.index.map(gene_dict)
    X = X.loc[~X["gene"].isna()]
    res = pd.DataFrame({"gene": X["gene"].values,
                        "context": np.repeat(target_col + "_siganture", X.shape[0]).tolist(),
                        "source": np.repeat("Lin_Blood2019", X.shape[0]).tolist(),
                        "weight": X[target_col].values})
    return res.set_index("gene")


sigs = pd.concat([get_signature_genes(ras, "KRAS", gene_names, std_span=3),
                  get_signature_genes(ras, "NRAS", gene_names, std_span=3)])
sigs.to_csv(EXT / "ras_signature_Lin_etal_Blood2019.csv")
