#!/usr/bin/env python3
"""Annotation of the Hotspot gene modules of the BM and PB run: flags for modules driven by one gene or one individual,
Kruskal-Wallis and Dunn tests across disease stage, tissue, cell type and cytogenetic group, and module names (Methods,
detection of single-cell transcriptional signatures).

Purpose:      Module activity (Hotspot module score per cell) is averaged per individual x disease stage x tissue x cell
              type (malignant or normal). A module is flagged 'single_gene' when its top gene carries >= 50% of the summed
              Hotspot Z-scores of its genes, and 'single_participant' when the z-scored unit means show a gap >= 6 between
              the two highest units; other artifacts are set by review (one-individual, batch and noise modules). For the
              other modules: Kruskal-Wallis tests across the stage x cell-type groups of BM units (11), PB units (11) and
              both (22) and, for BM malignant units, across eight cytogenetic groups; Benjamini-Hochberg over modules.
              Dunn's post hoc tests (BH within a module) for modules with cytogenetic FDR <= 0.05 and with all-group FDR
              <= 0.01. Modules are then named from their genes and test results (sig_status 'passed' unless
              'not_significant'). Disease stage: SMM risk group from SMM_stage_lenient; HD renamed Donor. Must be run with
              PYTHONHASHSEED=0.

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv: index = gex_id; disease_status, SMM_stage_lenient.
              data/annotation/plasma-cell-annotation-final-240327.csv: one row per cell; annotation, tumor_id, redcap_id.
              data/metadata/cytogenetics-final-240327.csv: index = <redcap_id>:tumor:<k>; cytogenetics, hrd_status.
              data/h5/sig/BM-PB_cells.h5ad (01_downsample_cells.py): obs gex_id, redcap_id, sample_type.
              Hotspot results on the BM and PB input of 02_hotspot_inputs.py, in data/signature/hotspot/mm-BM-PB-4980-DEs/:
              mm-BM-PB-4980-DEs_cell_module_score.csv (cells x modules) and mm-BM-PB-4980-DEs_module_assignment.csv
              (index = gene; C, Z, Pval, FDR, Module).

Outputs:      data/signature/hotspot/mm-BM-PB-4980-DEs-modules.pkl: one row per module: context, type, outlier, zdiff,
              n_genes, top_genes, test P and FDR values, sig_status.
              data/signature/hotspot/mm-BM-PB-4980-DEs_cytogenetics-posthoc.csv, mm-BM-PB-4980-DEs_group-posthoc.csv:
              significant Dunn comparisons (comparison, signature, qval).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels, scanpy), natsort, scikit-posthocs;
              config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import natsort
import numpy as np
import pandas as pd
import scanpy as sc
import scikit_posthocs as sp
import scipy.stats
import statsmodels.stats.multitest as multi

np.random.seed(0)
RUN = data("signature", "hotspot", "mm-BM-PB-4980-DEs")
HS = data("signature", "hotspot")
STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
CYTO = ["t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)", "t(12;14)", "t(14;16)", "t(14;20)", "HRD", "Other"]

# ---- PCs with disease stage, cell type and cytogenetic group --------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
lenient = ~meta["SMM_stage_lenient"].isna()
meta.loc[lenient, "disease_status"] = meta.loc[lenient, "SMM_stage_lenient"]
annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
cytogenetics = pd.read_csv(data("metadata", "cytogenetics-final-240327.csv"), index_col=0)
adata = sc.read(data("h5", "sig", "BM-PB_cells.h5ad"))
adata.obs = adata.obs.join(annot[sorted(set(annot.columns).difference(set(adata.obs.columns)))])
adata.obs = adata.obs.join(cytogenetics[["cytogenetics", "hrd_status"]], on="tumor_id")
adata.obs["cytogenetics"] = pd.Categorical(adata.obs["cytogenetics"], categories=CYTO)
adata.obs["disease_status"] = list(meta.loc[adata.obs["gex_id"], "disease_status"])
adata.obs.loc[adata.obs["disease_status"] == "HD", "disease_status"] = "Donor"
adata.obs["disease_status"] = pd.Categorical(adata.obs["disease_status"], categories=STAGES)

# ---- Hotspot module scores and module assignment ----------------------------------------------------------------------------
module_scores = pd.read_csv(RUN / "mm-BM-PB-4980-DEs_cell_module_score.csv", index_col=0)
module_scores.columns = "S" + module_scores.columns            # module id 'S' + integer
signatures = list(module_scores.columns)
adata.obs = adata.obs.join(module_scores)
results = pd.read_csv(RUN / "mm-BM-PB-4980-DEs_module_assignment.csv", index_col=0)
results = results[~results["Module"].isna() & (results["Module"] > 0)]
results["Module"] = ["S" + str(i) for i in results["Module"]]

# ---- groups: disease stage x tissue x cell type, and units: individual x disease stage x tissue x cell type -----------------
cell_categs = ["Donor:BM:normal", "MGUS:BM:normal", "LRSMM:BM:normal", "IRSMM:BM:normal", "HRSMM:BM:normal", "MM:BM:normal",
               "MGUS:BM:tumor", "LRSMM:BM:tumor", "IRSMM:BM:tumor", "HRSMM:BM:tumor", "MM:BM:tumor",
               "Donor:PB:normal", "MGUS:PB:normal", "LRSMM:PB:normal", "IRSMM:PB:normal", "HRSMM:PB:normal", "MM:PB:normal",
               "MGUS:PB:tumor", "LRSMM:PB:tumor", "IRSMM:PB:tumor", "HRSMM:PB:tumor", "MM:PB:tumor"]
adata.obs["cell_categ"] = list(np.array(adata.obs["disease_status"]) + ":" + np.array(adata.obs["sample_type"]) + ":"
                               + np.array(adata.obs["annotation"]))
adata.obs["cell_categ"] = pd.Categorical(adata.obs["cell_categ"], categories=cell_categs, ordered=True)

adata.obs["sample_categ"] = list(np.array(adata.obs["disease_status"]) + ":" + np.array(adata.obs["sample_type"]) + ":"
                                 + np.array(adata.obs["redcap_id"]))
sample_group = sorted(adata.obs["sample_categ"].unique())
ordered = []
for tissue in ["BM", "PB"]:
    for stage in STAGES:
        ordered += [x for x in sample_group if x.startswith(stage + ":" + tissue)]
sample_group = ordered
adata.obs["sample_categ"] = pd.Categorical(adata.obs["sample_categ"], categories=sample_group, ordered=True)

adata.obs["sample_pcg"] = list(np.array(adata.obs["disease_status"]) + ":" + np.array(adata.obs["sample_type"]) + ":"
                               + np.array(adata.obs["redcap_id"]) + ":" + np.array(adata.obs["annotation"]))
sample_pcg_bm = ([sg + ":normal" for sg in sample_group if ":BM:" in sg]
                 + [sg + ":tumor" for sg in sample_group if (sg.split(":")[0] != "Donor") and (":BM:" in sg)])
sample_pcg_pb = ([sg + ":normal" for sg in sample_group if ":PB:" in sg]
                 + [sg + ":tumor" for sg in sample_group if (sg.split(":")[0] != "Donor") and (":PB:" in sg)])
adata.obs["sample_pcg"] = pd.Categorical(adata.obs["sample_pcg"], categories=sample_pcg_bm + sample_pcg_pb, ordered=True)


# ---- module table: gene and individual dominance flags ------------------------------------------------------------------
def summarize_signature(sig_mean, hotspot):
    signatures = natsort.natsorted(hotspot["Module"].unique())
    x = pd.DataFrame(index=signatures, columns=["context", "type", "outlier", "zdiff", "n_genes", "top_genes"])
    for sig in signatures:
        flag = []
        context = []
        outlier = []
        # single gene driving the module (share of the summed Z-scores)
        sig_genes = hotspot.loc[hotspot["Module"] == sig, "Z"].sort_values()
        wgap = (sig_genes / sum(sig_genes))[-1]
        if wgap >= 0.5:
            flag.append("single_gene")
            context.append(sig_genes.index[-1])
            outlier.append(wgap)
        # single individual driving the module (gap between the two highest z-scored unit means; empirical cutoff)
        sig_zscores = scipy.stats.zscore(sig_mean[sig]).sort_values()
        zgap = sig_zscores.diff()[-1]
        if zgap >= 6:
            flag.append("single_participant")
            context.append(sig_zscores.index[-1])
            outlier.append(zgap)
        g = list(hotspot[hotspot["Module"] == sig].index)
        x.loc[sig, "context"] = ",".join(context) if len(context) > 0 else np.nan
        x.loc[sig, "type"] = ",".join(flag) if len(flag) > 0 else np.nan
        x.loc[sig, "outlier"] = outlier if len(outlier) > 0 else np.nan
        x.loc[sig, "zdiff"] = zgap
        x.loc[sig, "n_genes"] = len(g)
        x.loc[sig, "top_genes"] = list(sig_genes.index[sig_genes.index.isin(g)][::-1])
    return x


categ_sig_mean = adata.obs[signatures + ["sample_pcg"]].groupby("sample_pcg").mean()
categ_sig_mean = categ_sig_mean[list(categ_sig_mean.apply(lambda x: sum(x[signatures].isna()) != len(signatures), axis=1))]
sig_table = summarize_signature(categ_sig_mean, results)

# artifacts set by review: modules driven by one or two individuals, ambient or batch signals, noise
artifact = {
    "S11": ["IRSMM:BM:single_patient:tumor", "sparse_metabolism_signature"],
    "S15": ["IRSMM:[BM|PB]:single_patient:tumor", "single_participant"],
    "S21": ["LRSMM:[BM|PB]:single_patient:tumor", "single_participant"],
    "S25": ["MM:[BM|PB]:two_patients:tumor", "two_participant"],
    "S28": ["other_lymphoid_myeloid_genes", "ambient_contamination"],
    "S31": ["non_coding_genes", "noise_in_single_cluster"],
    "S33": ["GEX5_Donor_BM_high", "batch_effect"],
    "S34": ["GEX5_Donor_BM_high", "batch_effect"],
    "S35": ["low_level_signature", "potential_noise"],
}
for key in artifact:
    sig_table.loc[key, "context"] = artifact[key][0]
    sig_table.loc[key, "type"] = artifact[key][1]
sigs_to_explore = list(sig_table[sig_table["type"].isna()].index)

# ---- unit means with cytogenetic group --------------------------------------------------------------------------------------
pcg_sig_mean = categ_sig_mean.copy()
pcg_sig_mean["disease_status"] = [x.split(":")[0] for x in pcg_sig_mean.index]
pcg_sig_mean["sample_type"] = [x.split(":")[1] for x in pcg_sig_mean.index]
pcg_sig_mean["redcap_id"] = [x.split(":")[2] for x in pcg_sig_mean.index]
pcg_sig_mean["annotation"] = [x.split(":")[3] for x in pcg_sig_mean.index]
pcg_sig_mean["cytogenetics"] = np.nan
for i, row in pcg_sig_mean.iterrows():
    key = ":".join(i.split(":")[2:]) + ":0"
    if key not in cytogenetics.index:
        continue
    pcg_sig_mean.loc[i, "cytogenetics"] = cytogenetics.loc[key, "cytogenetics"]
pcg_sig_mean["cytogenetics"] = pd.Categorical(pcg_sig_mean["cytogenetics"], categories=adata.obs["cytogenetics"].cat.categories)
pcg_sig_mean["group"] = pcg_sig_mean["disease_status"] + ":" + pcg_sig_mean["sample_type"] + ":" + pcg_sig_mean["annotation"]
pcg_sig_mean = pcg_sig_mean[~pcg_sig_mean["group"].isin(["Donor:BM:tumor", "Donor:PB:tumor"])]
pcg_sig_mean["group"] = pd.Categorical(pcg_sig_mean["group"], categories=list(adata.obs["cell_categ"].cat.categories))

# ---- Kruskal-Wallis tests across groups (BM, PB, both) and cytogenetic groups (BM malignant units) ------------------------
sig_table["pval"] = np.nan
sig_table["bm_pval"] = np.nan
sig_table["pb_pval"] = np.nan
sig_table["cyto_pval"] = np.nan
for sig in sigs_to_explore:
    is_valid = ~pcg_sig_mean[sig].isna()
    is_normal = pcg_sig_mean["annotation"] == "normal"
    is_tumor = pcg_sig_mean["annotation"] == "tumor"
    groups = {}
    for tissue in ["BM", "PB"]:
        in_tissue = pcg_sig_mean["sample_type"] == tissue
        values = [pcg_sig_mean.loc[(pcg_sig_mean["disease_status"] == "Donor") & is_normal & is_valid & in_tissue, sig]]
        for stage in ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]:
            is_stage = pcg_sig_mean["disease_status"] == stage
            values.append(pcg_sig_mean.loc[is_stage & is_normal & is_valid & in_tissue, sig])
            values.append(pcg_sig_mean.loc[is_stage & is_tumor & is_valid & in_tissue, sig])
        groups[tissue] = values
    sig_table.loc[sig, "bm_pval"] = scipy.stats.kruskal(*groups["BM"])[1]
    sig_table.loc[sig, "pb_pval"] = scipy.stats.kruskal(*groups["PB"])[1]
    sig_table.loc[sig, "pval"] = scipy.stats.kruskal(*(groups["BM"] + groups["PB"]))[1]
    # cytogenetic groups, BM units (the group 'Other' is left out)
    is_marrow = pcg_sig_mean["sample_type"] == "BM"
    cyto = [pcg_sig_mean.loc[(pcg_sig_mean["cytogenetics"] == c) & is_marrow, sig]
            for c in ["HRD", "t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)", "t(12;14)", "t(14;16)", "t(14;20)"]]
    sig_table.loc[sig, "cyto_pval"] = scipy.stats.kruskal(*cyto)[1]

for col in ["", "bm_", "pb_", "cyto_"]:
    sig_table[col + "fdr"] = np.nan
    idx = ~sig_table[col + "pval"].isna()
    sig_table.loc[idx, col + "fdr"] = multi.fdrcorrection(sig_table.loc[idx, col + "pval"], alpha=0.05, method="indep",
                                                          is_sorted=False)[1]

# ---- Dunn's post hoc tests: cytogenetic groups (FDR <= 0.05) and stage x tissue x cell-type groups (FDR <= 0.01) ----------
cyto_fdr_cutoff = 0.05
cyto_sig = pcg_sig_mean[~pcg_sig_mean["cytogenetics"].isna()].copy()
cyto_sig_for_dunn = cyto_sig.loc[cyto_sig["cytogenetics"].isin(["HRD", "t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)",
                                                                 "t(12;14)", "t(14;16)", "t(14;20)"])].copy()
cyto_sig_for_dunn["cytogenetics"] = list(cyto_sig_for_dunn["cytogenetics"])
cyto_pw_res = []
for sig in sig_table[sig_table["cyto_fdr"] <= cyto_fdr_cutoff].index:
    pw = sp.posthoc_dunn(cyto_sig_for_dunn[[sig, "cytogenetics"]], val_col=sig, group_col="cytogenetics", p_adjust="fdr_bh")
    pw = pd.melt(pw.reset_index().rename(columns={"index": "cyto1"}), id_vars="cyto1", var_name="cyto2", value_name="qval")
    pw = pw[pw["cyto1"] != pw["cyto2"]]
    pw["comparison"] = pw.apply(lambda x: ":".join(sorted([x["cyto1"], x["cyto2"]])), axis=1)
    pw["signature"] = sig
    cyto_pw_res.append(pw.loc[pw["qval"] <= cyto_fdr_cutoff, ["comparison", "signature", "qval"]].drop_duplicates())
pd.concat(cyto_pw_res).to_csv(HS / "mm-BM-PB-4980-DEs_cytogenetics-posthoc.csv", index=False)

fdr_cutoff = 0.01
group_pw_res = []
for sig in sig_table[sig_table["fdr"] <= fdr_cutoff].index:
    pw = sp.posthoc_dunn(pcg_sig_mean[[sig, "group"]], val_col=sig, group_col="group", p_adjust="fdr_bh")
    pw = pd.melt(pw.reset_index().rename(columns={"index": "group1"}), id_vars="group1", var_name="group2", value_name="qval")
    pw = pw[pw["group1"] != pw["group2"]]
    pw["comparison"] = pw.apply(lambda x: ":".join(sorted([x["group1"], x["group2"]])), axis=1)
    pw["signature"] = sig
    group_pw_res.append(pw.loc[pw["qval"] <= fdr_cutoff, ["comparison", "signature", "qval"]].drop_duplicates())
pd.concat(group_pw_res).to_csv(HS / "mm-BM-PB-4980-DEs_group-posthoc.csv", index=False)

# ---- module names, set from the genes and tests (name, pattern, marker genes) ---------------------------------------------
sig_annotation = {
    "S1": ["cell_cycling_G2M", "PB_up,normal_up", ["MKI67", "TOP2A"]],
    "S2": ["cell_cycling_G1S", "PB_up,normal_up", ["RRM2", "TYMS"]],
    "S4": ["MHC-II_B_cell_genes", "PB_up,normal_up", ["HLA-DPB1", "HLA-DPA1", "CD27", "CD70", "CD19", "CD79A", "PTPRC"]],
    "S5": ["MCL1-BCL2_signature", "not_significant", ["MCL1", "BCL2", "PELI1"]],
    "S6": ["HRD_signature", "cytogenetics,tumor_up,progression", ["RPLP1", "UBA52", "MYC", "CD47", "CD200", "BTLA"]],
    "S7": ["MAF*_signature", "cytogenetics,tumor_up", ["MAFA", "MAF", "MAFB", "ITGB7", "DEPTOR", "CD82"]],
    "S8": ["FOS-JUN_genes", "BM_up", ["JUN", "JUNB", "FOS", "CXCR4"]],
    "S9": ["myeloma_oncogene", "tumor_up", ["MLLT3", "NCAM1", "IGF1", "SMAD1", "SMAD3", "SMAD7"]],
    "S10": ["myeloma_oncogene", "tumor_up", ["SDC1", "CTSL", "CTSS", "CTSW", "C2", "C4A", "C4B", "CD9", "CD63", "QPCT", "LY96"]],
    "S16": ["circulatory_capacity", "CTC_up", ["EZR", "EMP3", "VIM", "TAGLN2", "ANXA2"]],
    "S17": ["MHC-I_genes", "tumor_up", ["B2M", "HLA-A", "HLA-C", "HLA-F", "GPRC5D", "SOCS3"]],
    "S18": ["interferon_inducible", "progression", ["IFI6", "ISG15", "IFITM1", "IFI44L", "IFI44", "IFI27", "IFIT3"]],
    "S19": ["chromatin_chaperone_genes", "progression",
            ["HIST1H2AC", "HIST1H2BC", "HIST1H1C", "HSPA1B", "HSPA1A", "DNAJB1", "CCL3", "CCL4"]],
    "S22": ["TNF_genes", "BM_up,tumor_up", ["CD276", "LTA", "TNFSF12", "TNFSF13"]],
    "S23": ["cell_motility", "BM_up,tumor_up", ["RNF207", "TRIM3"]],
    "S26": ["extra_cellular_matrix", "BM_up,normal_up", ["COL6A2", "COL11A2", "AEBP1", "EFEMP2", "TNFRSF18", "TNFRSF4"]],
    "S27": ["signaling_pathway_genes", "BM_up", ["MAP3K21", "PTPRS", "IQSEC2"]],
    "S29": ["retroelement_derived", "BM_up,tumor_up", ["RTL5", "RTL6", "ZCCHC14", "ARRDC4"]],
}
sig_table["sig_status"] = np.nan
for sig in sig_annotation:
    sig_table.loc[sig, "context"] = sig_annotation[sig][0]
    sig_table.loc[sig, "type"] = sig_annotation[sig][1]
    if sig_annotation[sig][1] != "not_significant":
        sig_table.loc[sig, "sig_status"] = "passed"
sig_table.to_pickle(HS / "mm-BM-PB-4980-DEs-modules.pkl")
