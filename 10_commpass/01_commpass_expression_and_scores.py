#!/usr/bin/env python3
"""CoMMpass IA22 bulk RNA-seq of CD138-selected samples: gene-level tables, baseline BM metadata and survival, BM-PB pairs,
and signature, CNV-arm and partner-gene scores of the baseline BM samples (inputs of Figs. 3D-F, 5D; Supplementary Figs.
9B,C, 10, 15A,B).

Purpose:      1. Gene counts: sum of the salmon counts of each gene's transcripts (transcript map of the 10x reference
              GRCh38-2020-A). 2. Pairs of BM and PB samples of the same visit. 3. Gene names of the reference in place of
              Ensembl IDs (IDs without a name dropped). 4. Baseline BM samples (visit 1) with PER_PATIENT,
              PER_PATIENT_VISIT (visit of the sample) and STAND_ALONE_SURVIVAL data. 5. Per baseline BM sample, in TPM,
              DESeq2 variance-stabilized (VST) and log1p TPM units: mean expression and AUCell score (pySCENIC aucell,
              normalize=True) of the 16 signatures, SenMayo, the 11-gene proliferation index (CoMMpassPI) and the six
              stage-wise DE gene sets; for VST and log1p TPM also the CNV-arm scores (mean expression of the genes of 1p,
              1q, 13q, 14q and 17p minus that of the genes of chromosomes 2, 4, 10, 12 and 20) and the expression of the
              nine translocation partner genes. numpy seed 0. Run with PYTHONHASHSEED=0.

Inputs:       data/commpass/transcripts/tumor/MMRF_CoMMpass_IA22_salmon_transcriptUnstranded_counts.tsv: transcripts x
              samples. data/commpass/MMRF_CoMMpass_IA22_salmon_geneUnstranded_tpm.tsv: Ensembl genes x samples.
              data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT.tsv: IMWG_RISK_CLASS, PROLIF_INDEX, D_PT_iss, DEMOG_PATIENTAGE.
              data/commpass/MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv: PUBLIC_ID, SPECTRUM_SEQ, D_IM_DNA_INDEX,
              KEATSNFKB_INDEX, KEATSPROLIF_INDEX, PI_PBL_INDEX, UAMS70_INDEX, UAMS70_CALL,
              D_IM_FLOWCYT_PCT_ANEUPLOID_POPUL, D_IM_aneupgp.
              data/commpass/MMRF_CoMMpass_IA22_STAND_ALONE_SURVIVAL.tsv: censpfs1, ttcpfs1, censos, ttcos.
              data/commpass/MMRF_CoMMpass_IA22_BM_1_DESeq2_VST.tsv: VST expression, genes x baseline BM samples.
              data/reference/refdata-gex-GRCh38-2020-A/genes/: gene_to_transcript_id_map.tsv (gene_id, transcript_id),
              genes.bed (no header; column 1 chromosome, 4 gene name, 7 chromosome arm).
              data/reference/gene_names_refdata-gex-GRCh38-2020-A.txt: ensembl (index), gene_name.
              data/signature/hotspot/consensus-modules_genes.csv: gene (index), context (signature).
              data/signature/gene_list/cross_tvn_de/*.txt: the six DE gene sets.
              data/pathway/external/senmayo_Saul_etal_NatCommun_2022.txt: SenMayo genes.

Outputs:      data/commpass/MMRF_CoMMpass_IA22_salmon_geneUnstranded_aggCounts.tsv and .scRNA_compatible.tsv (gene
              names); data/commpass/raw_data/MMRF_CoMMpass_IA22_salmon_geneUnstranded_tpm.scRNA_compatible.tsv;
              data/commpass/MMRF_CoMMpass_IA22_BM-PB_pair_id.csv: public_id, sample_id, BM, PB.
              data/commpass/MMRF_CoMMpass_IA22_BM_1_metadata.tsv, _counts.tsv, _tpm.tsv: baseline BM samples.
              data/commpass/MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv (TPM), _VST_sig_score.tsv and
              _LogExpr_sig_score.tsv: metadata and scores (mean_expr:, auc:, CNV-arm and partner-gene columns).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); pySCENIC (aucell) and ctxcore; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import warnings

import numpy as np
import pandas as pd
from ctxcore.genesig import GeneSignature
from pyscenic.aucell import aucell

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(0)

CP = data("commpass")
REF = data("reference", "refdata-gex-GRCh38-2020-A", "genes")
GENE_LISTS = data("signature", "gene_list", "cross_tvn_de")


def load_gene_list(input_file):
    return [x.rstrip() for x in open(input_file)]


def compute_signature(table, gene_module, gene_lists):
    """per sample (column of table): mean expression and AUCell score of each signature and gene set"""
    gene_sigs = []
    score = pd.DataFrame(index=table.columns)

    # first the 16 signatures
    for name, members in gene_module.groupby("context"):
        genes = list(members.index)
        gene_sigs.append(GeneSignature(name, genes))
        score["mean_expr:{}".format(name)] = table.loc[genes].mean()

    # then the other gene sets
    for name in gene_lists:
        genes = gene_lists[name]
        gene_sigs.append(GeneSignature(name, genes))
        score["mean_expr:{}".format(name)] = table.loc[genes].mean()

    # AUCell
    auc_score = aucell(table.T, gene_sigs, normalize=True)
    auc_score = auc_score.rename(columns={c: "auc:{}".format(c) for c in auc_score.columns})

    score.index = [x.split("_1_BM_")[0] for x in score.index]
    auc_score.index = [x.split("_1_BM_")[0] for x in auc_score.index]
    return score.join(auc_score)


def compute_cnvscore(table, cnv_genes, control_genes):
    """per sample: mean expression of the genes of each arm minus that of the genes of the control chromosomes"""
    _X = table.copy()
    _control_genes = list(set(table.index).intersection(set(control_genes)))
    control_mean = _X.loc[_control_genes].mean()

    cnv_score = pd.DataFrame(index=_X.columns)
    for cnv_arm in cnv_genes.keys():
        genes = list(set(table.index).intersection(set(cnv_genes[cnv_arm])))
        cnv_mean = _X.loc[genes].mean()
        cnv_score["mean_expr:{}_full_log_fc".format(cnv_arm)] = cnv_mean - control_mean

    cnv_score.index = [x.split("_1_BM_")[0] for x in cnv_score.index]
    return cnv_score


def extract_cytogenes(table):
    """per sample: expression of the nine translocation partner genes"""
    cytogenes = ["CCND1", "CCND2", "CCND3", "MAF", "MAFA", "MAFB", "MYC", "NSD2", "FGFR3"]
    _X = table.loc[cytogenes].T
    _X.index = [x.split("_1_BM_")[0] for x in _X.index]
    return _X


# ---- 1. gene-level counts: sum of the transcript counts of each gene -----------------------------------------------------
transcripts = pd.read_csv(CP / "transcripts" / "tumor" / "MMRF_CoMMpass_IA22_salmon_transcriptUnstranded_counts.tsv",
                          index_col=0, sep="\t")
gene_to_transcript = pd.read_csv(REF / "gene_to_transcript_id_map.tsv", sep="\t")
gene_counts = pd.DataFrame(index=gene_to_transcript["gene_id"].unique(), columns=transcripts.columns)
for gene_id, df in gene_to_transcript.groupby("gene_id"):
    target_ids = list(set(df["transcript_id"].values).intersection(set(transcripts.index)))
    gene_counts.loc[gene_id, transcripts.columns] = transcripts.loc[target_ids].sum(axis=0)
gene_counts.to_csv(CP / "MMRF_CoMMpass_IA22_salmon_geneUnstranded_aggCounts.tsv", sep="\t")

# ---- 2. gene TPM; BM and PB samples of the same visit --------------------------------------------------------------------
expr = pd.read_csv(CP / "MMRF_CoMMpass_IA22_salmon_geneUnstranded_tpm.tsv", index_col=0, sep="\t")
bm_samples = [x for x in expr.columns if "_BM_" in x]
pb_samples = [x for x in expr.columns if "_PB_" in x]
paired_bm = [x for x in bm_samples if x.replace("_BM_", "_PB_") in pb_samples]
paired_pb = [x for x in pb_samples if x.replace("_PB_", "_BM_") in paired_bm]
paired_meta = pd.DataFrame({"public_id": ["_".join(x.split("_")[:2]) for x in paired_bm],
                            "sample_id": ["_".join(x.split("_")[:3]) for x in paired_bm],
                            "BM": paired_bm, "PB": paired_pb})
paired_meta.to_csv(CP / "MMRF_CoMMpass_IA22_BM-PB_pair_id.csv", index=False)

# ---- 3. gene names of the single-cell reference in place of Ensembl IDs ----------------------------------------------------
gene_names = pd.read_csv(data("reference", "gene_names_refdata-gex-GRCh38-2020-A.txt"), sep="\t", index_col=0)
gene_idx = expr.index.intersection(gene_names.index)
expr = expr.loc[gene_idx]
expr["gene"] = list(gene_names.loc[expr.index, "gene_name"])
expr.set_index("gene", drop=True, inplace=True)
(CP / "raw_data").mkdir(exist_ok=True)
expr.to_csv(CP / "raw_data" / "MMRF_CoMMpass_IA22_salmon_geneUnstranded_tpm.scRNA_compatible.tsv", sep="\t")

gene_counts["gene"] = list(gene_names.loc[gene_counts.index, "gene_name"])
gene_counts.set_index("gene", drop=True, inplace=True)
gene_counts.to_csv(CP / "MMRF_CoMMpass_IA22_salmon_geneUnstranded_aggCounts.scRNA_compatible.tsv", sep="\t")

# ---- 4. baseline BM samples with clinical data and survival ---------------------------------------------------------------
bm_to_check = [x for x in expr.columns if "_1_BM_" in x]
public_id = ["_".join(x.split("_")[:2]) for x in bm_to_check]

df = pd.read_csv(CP / "MMRF_CoMMpass_IA22_PER_PATIENT.tsv", index_col=0, sep="\t")
columns = ["IMWG_RISK_CLASS", "PROLIF_INDEX", "D_PT_iss", "DEMOG_PATIENTAGE"]
idx = df.index.intersection(public_id)
meta = df.loc[idx, columns].copy()

# visit of each baseline BM sample
df = pd.read_csv(CP / "MMRF_CoMMpass_IA22_PER_PATIENT_VISIT.tsv", sep="\t")
spectrum_seq = ["_".join(x.split("_")[:3]) for x in bm_to_check]
df = df[df["SPECTRUM_SEQ"].isin(spectrum_seq)]
df = df.set_index("PUBLIC_ID")
columns = ["SPECTRUM_SEQ", "D_IM_DNA_INDEX", "KEATSNFKB_INDEX", "KEATSPROLIF_INDEX",
           "PI_PBL_INDEX", "UAMS70_INDEX", "UAMS70_CALL", "D_IM_FLOWCYT_PCT_ANEUPLOID_POPUL", "D_IM_aneupgp"]
meta = df[columns].join(meta)

# progression-free and overall survival
df = pd.read_csv(CP / "MMRF_CoMMpass_IA22_STAND_ALONE_SURVIVAL.tsv", sep="\t", index_col=0)
columns = ["censpfs1", "ttcpfs1", "censos", "ttcos"]
df = df.loc[meta.index, columns]
meta = meta.join(df)
meta.to_csv(CP / "MMRF_CoMMpass_IA22_BM_1_metadata.tsv", sep="\t")

# counts and TPM of the baseline BM samples, in the order of the metadata
gene_counts = gene_counts[[x + "_BM_CD138pos" for x in meta["SPECTRUM_SEQ"]]]
gene_counts.to_csv(CP / "MMRF_CoMMpass_IA22_BM_1_counts.tsv", sep="\t")
expr = expr[[x + "_BM_CD138pos" for x in meta["SPECTRUM_SEQ"]]]
expr.to_csv(CP / "MMRF_CoMMpass_IA22_BM_1_tpm.tsv", sep="\t")

# ---- 5. scores of the baseline BM samples in three units -----------------------------------------------------------------
vst = pd.read_csv(CP / "MMRF_CoMMpass_IA22_BM_1_DESeq2_VST.tsv", sep="\t", index_col=0)
expr = pd.read_csv(CP / "MMRF_CoMMpass_IA22_BM_1_tpm.tsv", sep="\t", index_col=0)
log_expr = np.log1p(expr)                                                               # log1p TPM

gene_module = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
gene_lists = {
    "SenMayo": load_gene_list(data("pathway", "external", "senmayo_Saul_etal_NatCommun_2022.txt")),
    "CoMMpassPI": ["TYMS", "TK1", "CCNB1", "MKI67", "GINS1", "CKS1B", "TOP2A", "UBE2C", "ZWINT", "TRIP13", "KIF11"],
    "MGUS_UP": load_gene_list(GENE_LISTS / "early_upreg_111genes.txt"),
    "SMM_UP": load_gene_list(GENE_LISTS / "smm-on_upreg_116genes.txt"),
    "MM_UP": load_gene_list(GENE_LISTS / "late_upreg_74genes.txt"),
    "MGUS_DN": load_gene_list(GENE_LISTS / "early_downreg_36genes.txt"),
    "SMM_DN": load_gene_list(GENE_LISTS / "smm-on_downreg_66genes.txt"),
    "MM_DN": load_gene_list(GENE_LISTS / "late_downreg_74genes.txt")
}

# genes of the control chromosomes and of the five driver arms
gene_bed = pd.read_csv(REF / "genes.bed", sep="\t", header=None)
control_chroms = ["chr2", "chr4", "chr10", "chr12", "chr20"]
control_genes = list(gene_bed.loc[gene_bed[0].isin(control_chroms), 3])
cnv_arms = ["chr1p", "chr1q", "chr13q", "chr14q", "chr17p"]
cnv_genes = dict()
for cnv_arm in cnv_arms:
    cnv_genes[cnv_arm.replace("chr", "")] = list(gene_bed.loc[gene_bed[6] == cnv_arm, 3])

meta = pd.read_csv(CP / "MMRF_CoMMpass_IA22_BM_1_metadata.tsv", sep="\t", index_col=0)

# TPM: signatures and gene sets
expr_sig = compute_signature(expr, gene_module, gene_lists)
meta.join(expr_sig).to_csv(CP / "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv", sep="\t")

# VST: signatures and gene sets, CNV arms, partner genes
vst_sig = compute_signature(vst, gene_module, gene_lists)
vst_cnv = compute_cnvscore(vst, cnv_genes, control_genes)
vst_cyto = extract_cytogenes(vst)
vst_final = vst_sig.join(vst_cnv).join(vst_cyto)
meta.join(vst_final).to_csv(CP / "MMRF_CoMMpass_IA22_BM_1_VST_sig_score.tsv", sep="\t")

# log1p TPM: signatures and gene sets, CNV arms, partner genes
log_expr_sig = compute_signature(log_expr, gene_module, gene_lists)
log_expr_cnv = compute_cnvscore(log_expr, cnv_genes, control_genes)
log_expr_cyto = extract_cytogenes(log_expr)
log_expr_final = log_expr_sig.join(log_expr_cnv).join(log_expr_cyto)
meta.join(log_expr_final).to_csv(CP / "MMRF_CoMMpass_IA22_BM_1_LogExpr_sig_score.tsv", sep="\t")
