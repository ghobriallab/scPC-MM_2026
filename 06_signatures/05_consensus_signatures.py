#!/usr/bin/env python3
"""The 16 consensus signatures from the BM and the BM and PB Hotspot runs: genes of each signature with their Hotspot
statistics, and the gene-gene local-correlation Z matrix ordered by signature (Fig. 3A,B; Methods, detection of single-cell
transcriptional signatures).

Purpose:      Each consensus signature combines one module of the BM and PB run with one (or two) modules of the BM run
              (or uses one run only), by the rule listed for it: genes in both modules ('intersection'), genes of the BM
              module(s) ('BM'), or genes of the BM and PB module only ('BM-PB_only'); MAFA is first moved from BM module
              S30 to S7 (MAF* signature). Genes assigned to two signatures: the 4 shared with circulatory capacity stay in
              both, the 20 others are removed from MHC-II immune checkpoint (1,467 gene assignments in total). The Hotspot
              C, Z, P and FDR of each gene are taken from the run listed for its signature. Z matrix: local-correlation
              Z-scores of the BM and PB run; genes in the dendrogram order of its modules, block by block: the genes of the
              signatures whose genes fall mostly in that module, then the block's genes outside all signatures. Must be run
              with PYTHONHASHSEED=0.

Inputs:       data/signature/hotspot/mm-BM-4980-full-DEs-modules.pkl (04_modules_bm.py) and
              mm-BM-PB-4980-DEs-modules.pkl (03_modules_bm_pb.py): per module context, type, top_genes.
              Hotspot results, in data/signature/hotspot/: mm-BM-4980-full-DEs/mm-BM-4980-full-DEs_module_assignment.csv
              and mm-BM-PB-4980-DEs/mm-BM-PB-4980-DEs_module_assignment.csv (index = gene; C, Z, Pval, FDR, Module);
              mm-BM-PB-4980-DEs/mm-BM-PB-4980-DEs_hotspot.pkl (Hotspot object of the BM and PB run: modules, linkage,
              local_correlation_z).

Outputs:      data/signature/hotspot/consensus-modules_notes.pkl: per signature: type, consproc, n_genes, gene_list.
              data/signature/hotspot/consensus-modules_genes.csv: index = gene; context (signature), C, Z, Pval, FDR.
              data/signature/hotspot/consensus-modules_local-correlation-z.pkl: genes x genes Z-scores (4,884 genes).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy), hotspot (to load the Hotspot object); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import pickle

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list

np.random.seed(0)
HS = data("signature", "hotspot")

# ---- module annotations and gene assignments of the two runs -------------------------------------------------------------
bm_sigs = pd.read_pickle(HS / "mm-BM-4980-full-DEs-modules.pkl")
full_sigs = pd.read_pickle(HS / "mm-BM-PB-4980-DEs-modules.pkl")
bm_genes = pd.read_csv(HS / "mm-BM-4980-full-DEs" / "mm-BM-4980-full-DEs_module_assignment.csv", index_col=0)
full_genes = pd.read_csv(HS / "mm-BM-PB-4980-DEs" / "mm-BM-PB-4980-DEs_module_assignment.csv", index_col=0)

# MAFA joins the MAF* signature module of the BM run
bm_sigs.loc["S7", "top_genes"].append("MAFA")
bm_sigs.loc["S30", "top_genes"].remove("MAFA")

# ---- consensus gene lists ---------------------------------------------------------------------------------------------------
# name: [(BM run module(s), BM and PB run module), rule]
consensus_list = {
    "MAF*_signature": [("S7", "S7"), "intersection"],
    "B_cell_activation": [("S6", "S4"), "intersection"],
    "MHC-II_immune_checkpoint": [("S6", "S4"), "BM-PB_only"],
    "HRD_signature": [("S3", "S6"), "BM"],
    "myeloma_oncogene": [("S12", "S9"), "BM"],
    "cell_cycling_G2M": [("S1", "S1"), "intersection"],
    "histone_chaperone_gene": [("S21", "S19"), "BM"],
    "MCL1-BCL2_signature": [("S5", "S5"), "intersection"],
    "vacuole_ER_signaling": [("S11,S33", "S10"), "BM"],
    "FOS-JUN_stress_response": [("S9", "S8"), "intersection"],
    "secretory_granule": [("S18", "S9"), "BM"],
    "extracellular_matrix": [("S37", "S26"), "BM"],
    "interferon_inducible": [("S20", "S18"), "BM"],
    "cell_cycling_G1S": [("S1,S4", "S2"), "intersection"],
    "MHC-I_gene_module": [("S17", "S17"), "BM"],
    "circulatory_capacity": [(None, "S16"), "BM-PB_only"],
}
consensus_sigs = {"context": [], "type": [], "consproc": [], "n_genes": [], "gene_list": []}
for key in consensus_list.keys():
    consensus_sigs["context"].append(key)
    consensus_sigs["consproc"].append(consensus_list[key][1])
    bm_id, full_id = consensus_list[key][0]
    # module of the BM and PB run only
    if bm_id is None:
        genes = list(full_sigs.loc[full_id, "top_genes"])
        consensus_sigs["type"].append(full_sigs.loc[full_id, "type"])
        consensus_sigs["n_genes"].append(len(genes))
        consensus_sigs["gene_list"].append(genes)
        continue
    bm_id = bm_id.split(",")
    full_gene_list = set(full_sigs.loc[full_id, "top_genes"])
    full_flag_type = set(full_sigs.loc[full_id, "type"].split(","))
    gene_list = set()
    flag_type = set()
    proc = consensus_list[key][1]
    for i in bm_id:
        bm_gene_list = set(bm_sigs.loc[i, "top_genes"])
        bm_flag_type = set(bm_sigs.loc[i, "type"].split(","))
        if proc == "intersection":
            gene_list = gene_list.union(full_gene_list.intersection(bm_gene_list))
            flag_type = flag_type.union(full_flag_type.intersection(bm_flag_type))
        elif proc == "BM":
            gene_list = gene_list.union(bm_gene_list)
            flag_type = flag_type.union(bm_flag_type)
        else:
            gene_list = full_gene_list.difference(bm_gene_list)
            flag_type = full_flag_type.difference(bm_flag_type)
    if len(flag_type) == 0:
        flag_type = ["not_significant"]
    consensus_sigs["type"].append(",".join(list(flag_type)))
    consensus_sigs["n_genes"].append(len(gene_list))
    consensus_sigs["gene_list"].append(sorted(gene_list))
consensus_sigs = pd.DataFrame(consensus_sigs)

consensus_gene_list = {"gene": [], "context": []}
for _, row in consensus_sigs.iterrows():
    consensus_gene_list["context"].extend([row["context"] for i in row["gene_list"]])
    consensus_gene_list["gene"].extend(row["gene_list"])
consensus_gene_list = pd.DataFrame(consensus_gene_list)

# ---- genes in two signatures: kept in circulatory capacity, otherwise removed from MHC-II immune checkpoint ----------------
freq = pd.DataFrame(consensus_gene_list.value_counts("gene"))
dups = list(freq[freq[0] > 1].index)
dup_assignments = consensus_gene_list[consensus_gene_list["gene"].isin(dups)]
ctc_assignments = dup_assignments[dup_assignments["context"] == "circulatory_capacity"]
bm_assignments = dup_assignments[~dup_assignments["gene"].isin(ctc_assignments["gene"])
                                 & (dup_assignments["context"] != "MHC-II_immune_checkpoint")]
white_list = pd.concat([ctc_assignments, bm_assignments])
black_list = dup_assignments[~dup_assignments.index.isin(white_list.index)]
black_list = black_list.set_index("gene").rename(columns={"context": "blacklist"}).join(white_list.set_index("gene"))
black_list = black_list[black_list["context"] != "circulatory_capacity"]   # these genes stay in both signatures

mhc_ii_genes = list(consensus_sigs.loc[consensus_sigs["context"] == "MHC-II_immune_checkpoint", "gene_list"])[0]
mhc_ii_genes = sorted(set(mhc_ii_genes).difference(set(black_list.index)))
consensus_sigs.loc[consensus_sigs["context"] == "MHC-II_immune_checkpoint", "n_genes"] = len(mhc_ii_genes)
gene_list_elem = list(consensus_sigs["gene_list"])
gene_list_elem[2] = sorted(set(gene_list_elem[2]).intersection(set(mhc_ii_genes)))   # MHC-II immune checkpoint
consensus_sigs["gene_list"] = gene_list_elem
consensus_sigs.set_index("context").to_pickle(HS / "consensus-modules_notes.pkl")

idx = consensus_gene_list["gene"].isin(black_list.index) & (consensus_gene_list["context"] == "MHC-II_immune_checkpoint")
consensus_gene_list = consensus_gene_list[~idx].reset_index(drop=True).set_index("gene")


# ---- Hotspot statistics of each gene, from the run that defines its signature ----------------------------------------------
def insert_values(cons, piece, sig):
    X = cons.copy()
    for col in ["C", "Z", "Pval", "FDR"]:
        # gene by gene, as a gene can belong to two signatures
        for g in list(X[X["context"] == sig].index):
            X.loc[(X.index == g) & (X["context"] == sig), col] = piece.loc[g, col]
    return X


source_run = {"MAF*_signature": full_genes, "B_cell_activation": full_genes, "MHC-II_immune_checkpoint": full_genes,
              "HRD_signature": bm_genes, "myeloma_oncogene": bm_genes, "cell_cycling_G2M": full_genes,
              "histone_chaperone_gene": bm_genes, "MCL1-BCL2_signature": full_genes, "vacuole_ER_signaling": full_genes,
              "FOS-JUN_stress_response": full_genes, "secretory_granule": bm_genes, "extracellular_matrix": bm_genes,
              "interferon_inducible": bm_genes, "cell_cycling_G1S": full_genes, "MHC-I_gene_module": bm_genes,
              "circulatory_capacity": full_genes}
for sig, piece in source_run.items():
    consensus_gene_list = insert_values(consensus_gene_list, piece, sig)
consensus_gene_list.to_csv(HS / "consensus-modules_genes.csv")

# ---- local-correlation Z matrix, genes ordered by module blocks with the signature genes first ----------------------------
with open(HS / "mm-BM-PB-4980-DEs" / "mm-BM-PB-4980-DEs_hotspot.pkl", "rb") as fi:
    full_hotspot = pickle.load(fi)
modules = pd.DataFrame(full_hotspot.modules[leaves_list(full_hotspot.linkage)])
modules["Consensus"] = modules["Module"]
modules.loc[~modules["Consensus"].index.isin(consensus_gene_list.index), "Consensus"] = -1
for sig, df in consensus_gene_list.groupby("context"):
    modules.loc[df.index, "Consensus"] = sig

# the BM and PB module holding most genes of each signature
freq = modules.value_counts(["Consensus", "Module"]).reset_index().rename(columns={0: "count"})
freq = freq[freq["Consensus"] != -1]
freq = freq.sort_values("Consensus")
module_nums = freq.groupby("Consensus").apply(lambda x: x.loc[x["count"] == x["count"].max(), "Module"].values[0]).to_dict()
module_names = {v: [] for v in set(module_nums.values())}
for key in module_nums:
    module_names[module_nums[key]].append(key)

block_num = modules.loc[modules["Module"].diff() != 0, "Module"]
block_gene = modules.loc[modules["Module"].diff() != 0, "Module"].index
gene_order = []
for i in range(len(block_num)):
    beg = list(modules.index).index(block_gene[i])
    if i < len(block_num) - 1:
        end = list(modules.index).index(block_gene[i + 1])
    else:
        end = len(modules.index)
    block_gene_set = list(modules.index[beg:end])
    hits = module_names.get(block_num[i], [])
    if len(hits) > 0:
        for hit in hits:
            gene_order.extend(modules.loc[modules["Consensus"] == hit].index)
    idx = modules.index.isin(block_gene_set) & (modules["Consensus"] == -1)
    if sum(idx) > 0:
        gene_order.extend(modules.loc[idx].index)

Z = full_hotspot.local_correlation_z.loc[gene_order, gene_order]
Z.to_pickle(HS / "consensus-modules_local-correlation-z.pkl")
