#!/usr/bin/env python3
"""Official gene symbols and Ensembl IDs of the genes of the Zhan et al. subgroups and the CoMMpass RNA subtypes (Methods,
comparison with published signatures).

Purpose:      Each gene is matched to the gene names of the 10x reference (GRCh38-2020-A) and given its Ensembl ID; genes
              without a match are searched in the HGNC REST service (rest.genenames.org: search by name, best-scoring
              symbol kept when unique, then the symbol's Ensembl ID and official symbol; fetched = True). The Zhan and
              CoMMpass tables written by 11_published_gene_lists.py are updated in place (gene filled for unmatched rows,
              columns fetched and gencode_id added), so this script runs once after it. Results depend on the HGNC database at the time of the query. Must be run with
              PYTHONHASHSEED=0.

Inputs:       data/reference/gene_names_refdata-gex-GRCh38-2020-A.txt: ensembl, gene_name (10x reference).
              data/pathway/external/zhan_etal_Blood2006.csv and CoMMPass_12RnaSubtypes_NatGenetics2024.csv
              (11_published_gene_lists.py): gene, orig_gene_name.

Outputs:      data/pathway/external/: zhan_etal_Blood2006.csv, CoMMPass_12RnaSubtypes_NatGenetics2024.csv (updated in
              place): gene (official symbol), orig_gene_name, fetched, gencode_id.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas), requests; internet access to rest.genenames.org;
              config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import numpy as np
import pandas as pd
import requests

np.random.seed(0)
EXT = data("pathway", "external")


def hugo_symbol_to_ensembl(gene_name):
    """official HGNC symbol and Ensembl ID of a gene name, or None"""
    if gene_name.startswith("ENSG"):
        return None
    # search the name; keep the best-scoring symbol when it is unique
    search_url = "https://rest.genenames.org/search/{}"
    response = requests.get(search_url.format(gene_name))
    hugo_prefix = '<str name="symbol">'
    score_prefix = '<float name="score">'
    ensg_prefix = '<str name="ensembl_gene_id">'
    symbol = []
    score = []
    for line in response.text.rstrip().split("\n"):
        if len(line) == 0:
            continue
        if hugo_prefix in line:
            symbol.append(line.split(hugo_prefix)[1].split("<")[0])
        if score_prefix in line:
            score.append(float(line.split(score_prefix)[1].split("<")[0]))
    if len(symbol) == 0:
        return None
    symbol = np.array(symbol)
    score = np.array(score)
    if sum(score == score.max()) > 1:
        return None
    symbol_match = symbol[score.argmax()]
    # fetch the gene entry of that symbol
    fetch_url = "https://rest.genenames.org/fetch/symbol/{}"
    response = requests.get(fetch_url.format(symbol_match))
    if ensg_prefix in response.text:
        ensembl_id = response.text.split(ensg_prefix)[1].split("<")[0]
        official_symbol = response.text.split(hugo_prefix)[1].split("<")[0]
        return {"ensembl": ensembl_id, "symbol": official_symbol}
    return None


def fill_missing(df):
    """Ensembl ID of genes already named; HGNC symbol and Ensembl ID of the others"""
    for i, row in df.iterrows():
        if str(row["gene"]) != "nan":
            df.loc[i, "gencode_id"] = gencode_ids.loc[row["gene"], "ensembl"]
            continue
        fetch = hugo_symbol_to_ensembl(row["orig_gene_name"])
        if fetch is not None:
            df.loc[i, "fetched"] = True
            df.loc[i, "gencode_id"] = fetch["ensembl"]
            df.loc[i, "gene"] = fetch["symbol"]


gencode_ids = pd.read_csv(data("reference", "gene_names_refdata-gex-GRCh38-2020-A.txt"), sep="\t", index_col=1)

# ---- Zhan et al. subgroups (table of 11_published_gene_lists.py, updated in place) ----------------------------------------
zhan_genes = pd.read_csv(EXT / "zhan_etal_Blood2006.csv")
zhan_genes["fetched"] = False
zhan_genes["gencode_id"] = np.nan
fill_missing(zhan_genes)
zhan_genes.to_csv(EXT / "zhan_etal_Blood2006.csv", index=False)

# ---- CoMMpass RNA subtypes (table of 11_published_gene_lists.py, updated in place) ------------------------------------------
commpass_genes = pd.read_csv(EXT / "CoMMPass_12RnaSubtypes_NatGenetics2024.csv", index_col=0)
commpass_genes["fetched"] = False
commpass_genes["gencode_id"] = np.nan
fill_missing(commpass_genes)
commpass_genes.to_csv(EXT / "CoMMPass_12RnaSubtypes_NatGenetics2024.csv")

