#!/usr/bin/env python3
"""Ensembl gene IDs of the gene symbols of the expression matrices, for the GO enrichment (Supplementary Fig. 7A).

Purpose:      The cohort's matrices carry gene symbols only; the GO and Reactome annotation files of g:Profiler are keyed
              by Ensembl gene ID. This reads the 10x Genomics reference the data were aligned to (GRCh38-2020-A) and
              lists every gene with its ID and symbol.

Inputs:       data/reference/refdata-gex-GRCh38-2020-A/genes/genes.gtf (10x Genomics reference, public).

Outputs:      results/signatures/gene_ids_10x_GRCh38-2020-A.tsv: gene_id, symbol.

Dependencies: Python 3.8 with requirements.txt (pandas); config.py.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

rows = []
with open(data("reference", "refdata-gex-GRCh38-2020-A", "genes", "genes.gtf")) as fh:
    for line in fh:
        f = line.split("\t")
        if len(f) > 8 and f[2] == "gene":
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
            name = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
            rows.append((gid, name))
df = pd.DataFrame(rows, columns=["gene_id", "symbol"])
print("reference genes:", len(df))
df.to_csv(result_dir("signatures") / "gene_ids_10x_GRCh38-2020-A.tsv", sep="\t", index=False)
