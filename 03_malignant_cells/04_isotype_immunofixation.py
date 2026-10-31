#!/usr/bin/env python3
"""Isotype and light chain of the malignant BM PCs of each patient from scBCR-seq, and their comparison with the
clinical serum immunofixation (isotype table and immunofixation concordance table).

Purpose:      Units = tissue x individual x disease stage (SMM risk group from SMM_stage_lenient) x PC annotation
              (malignant "tumor" or "normal", from the per-cell annotation table); each BCR cell takes the disease stage
              of its GEX library. In each unit: number and fraction of PCs per isotype (isotype_status: IgG, IgA, IgD,
              IgM, IgM/IgD, IgE, None = no heavy chain call) and per chain status (isotype + light chain). For the
              malignant BM units, the isotype fractions, the light chain fractions (IgK, IgL: sums over the chain
              statuses with that light chain) and the clinical immunofixation result (IFX; BM libraries of the cohort
              table with a result, matched on tissue, individual and disease stage) form the immunofixation
              concordance table. In each unit, the clonal heavy chain (among IgG, IgA, None, IgD, IgM, IgM/IgD) and
              light chain are those with the largest fraction; the isotype table gives, per patient, the immunofixation
              result, the clonal calls and the mean fractions over the patient's units. Must be run with
              PYTHONHASHSEED=0.

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv (one row per GEX library, first column gex_id):
              redcap_id, disease_status, SMM_stage_lenient, sample_type, IFX.
              data/h5/agg/plasma-cell-GEX5-umap-250207.h5ad (obs redcap_id: individuals with PCs; read in backed mode).
              data/annotation/plasma-cell-annotation-final-240327.csv (one row per PC): G2M_score, S_score, phase,
              sample_type, catissue_id, redcap_id, clonotype_id, redcap_clonotype_id, redcap_clonotype_rank,
              annotation, tumor_id.
              data/metadata/cytogenetics-final-240327.csv (first column tumor_id): cytogenetics, hrd_status.
              data/vdj/cohort-BCR-filtered-223pts.pkl (written by 03_cohort_bcr_tables.py): isotype_status,
              chain_status_main.

Outputs:      data/vdj/ifx_concordance_111824.csv: one row per malignant BM unit (BM:<individual>:<stage>:tumor): IFX,
              fractions of the unit's PCs by isotype (IgG, IgA, None, IgD, IgM, IgM/IgD) and light chain (IgK, IgL),
              disease_status, cytogenetics.
              data/metadata/isotype-final-250409.csv: one row per patient (redcap_id): IFX, sc_clonal_heavy_chain,
              sc_clonal_light_chain and the mean fractions IgG, IgA, None, IgD, IgM, IgM/IgD, IgK, IgL.

Dependencies: Python 3.8 with dandelion 0.3.3 (class of the pickled BCR object; pandas 2.0.3, numpy 1.24.4) and scanpy;
              config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import pickle
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import dandelion  # noqa: F401  (class of the pickled BCR object)

warnings.simplefilter(action="ignore", category=FutureWarning)

STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
CYTOGENETICS = ["t(4;14)", "t(6;14)", "t(8;14)", "t(11;14)", "t(12;14)", "t(14;16)", "t(14;20)", "HRD", "Other"]
ISOTYPES = ["IgG", "IgA", "IgD", "IgM", "IgM/IgD", "IgE", "None"]
CHAIN_STATUS = ["IgG+IgK", "IgG+IgL", "IgA+IgK", "IgA+IgL", "IgD+IgK", "IgD+IgL", "IgM+IgK", "IgM+IgL",
                "IgM/IgD+IgK", "IgM/IgD+IgL", "IgE+IgK", "IgE+IgL", "None+IgK", "None+IgL"]

# ---- 1. Cohort metadata, PC annotation and cytogenetic groups ----------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), index_col=0)
adata = sc.read(data("h5", "agg", "plasma-cell-GEX5-umap-250207.h5ad"), backed="r")
# individuals with PCs in the atlas
meta = meta.loc[meta["redcap_id"].isin(adata.obs["redcap_id"].unique())]
# SMM risk group as disease stage
smm = ~meta["SMM_stage_lenient"].isna()
meta.loc[smm, "disease_status"] = meta.loc[smm, "SMM_stage_lenient"]

annot = pd.read_csv(data("annotation", "plasma-cell-annotation-final-240327.csv"), index_col=0)
cytogenetics = pd.read_csv(data("metadata", "cytogenetics-final-240327.csv"), index_col=0)

# ---- 2. BCR cells with their PC annotation, disease stage and cytogenetic group ----------------
with open(data("vdj", "cohort-BCR-filtered-223pts.pkl"), "rb") as f:
    vdj = pickle.load(f)

annot_cols_to_add = ["G2M_score", "S_score", "phase", "sample_type", "catissue_id", "redcap_id",
                     "clonotype_id", "redcap_clonotype_id", "redcap_clonotype_rank",
                     "annotation", "tumor_id"]
vdj.metadata = vdj.metadata.join(annot[annot_cols_to_add])
vdj.metadata["gex_id"] = [x.split("-1-")[1] for x in vdj.metadata.index]

# disease stage of the GEX library of each cell
vdj.metadata["disease_status"] = list(meta.loc[list(vdj.metadata["gex_id"]), "disease_status"])
vdj.metadata.loc[vdj.metadata["disease_status"] == "HD", "disease_status"] = "Donor"
vdj.metadata["disease_status"] = pd.Categorical(vdj.metadata["disease_status"], categories=STAGES)

# cytogenetic group of the malignant clone of each cell
vdj.metadata = vdj.metadata.join(cytogenetics[["cytogenetics", "hrd_status"]], on="tumor_id")
vdj.metadata["cytogenetics"] = pd.Categorical(vdj.metadata["cytogenetics"], categories=CYTOGENETICS)

# ---- 3. PCs per isotype and per chain status in each unit --------------------------------------
# unit = tissue:individual:disease stage:PC annotation
vdj.metadata["id"] = vdj.metadata.apply(
    lambda x: ":".join([x["sample_type"], x["redcap_id"], x["disease_status"], x["annotation"]]), axis=1)


def count_cells(obs, key, pat_key):
    """Number and fraction of the cells of each value of key in each unit."""
    tab = pd.crosstab(obs[key], obs[pat_key])
    tab.columns = tab.columns.astype(str)
    freq = tab.reset_index().melt(id_vars=key, value_name="count")

    total_cells = pd.DataFrame(tab.sum(), columns=["n"])
    freq = freq.join(total_cells, on=pat_key, how="left")
    freq["frac"] = freq["count"] / freq["n"]
    return freq


def reorg_meta(df, cytogenetics, col="id"):
    """Unit fields as columns, cytogenetic group of the first malignant clone (tumor:0), fractions per sample and
    individual."""
    x = df.copy()
    for i, row in x.iterrows():
        sample_type, redcap_id, disease_status, annotation = row[col].split(":")
        x.loc[i, "indiv"] = ":".join([sample_type, redcap_id])
        x.loc[i, "sample"] = ":".join([sample_type, redcap_id, disease_status])
        x.loc[i, "sample_type"] = sample_type
        x.loc[i, "redcap_id"] = redcap_id
        x.loc[i, "disease_status"] = disease_status
        x.loc[i, "annotation"] = annotation

        key = ":".join([redcap_id, annotation, "0"])

        if key in cytogenetics.index:
            x.loc[i, "cytogenetics"] = cytogenetics.loc[key, "cytogenetics"]

    for i, grp in x.groupby("sample"):
        x.loc[x["sample"] == i, "sample_n"] = grp["count"].sum()
    x["sample_frac"] = x["count"] / x["sample_n"]

    for i, grp in x.groupby("indiv"):
        x.loc[x["indiv"] == i, "indiv_n"] = grp["count"].sum()
    x["indiv_frac"] = x["count"] / x["indiv_n"]

    x["disease_status"] = pd.Categorical(x["disease_status"], categories=STAGES)
    x["cytogenetics"] = pd.Categorical(x["cytogenetics"], categories=CYTOGENETICS)
    return x


isotype_status = count_cells(vdj.metadata, "isotype_status", "id")
chain_status = count_cells(vdj.metadata, "chain_status_main", "id")

isotype_status = reorg_meta(isotype_status, cytogenetics, "id")
chain_status = reorg_meta(chain_status, cytogenetics, "id")

isotype_status["isotype_status"] = pd.Categorical(isotype_status["isotype_status"], categories=ISOTYPES)
isotype_status["annotation"] = pd.Categorical(isotype_status["annotation"], categories=["normal", "tumor"])
chain_status["chain_status_main"] = pd.Categorical(chain_status["chain_status_main"], categories=CHAIN_STATUS)
chain_status["annotation"] = pd.Categorical(chain_status["annotation"], categories=["normal", "tumor"])


# ---- 4. Isotype and light chain fractions of the malignant BM units ----------------------------
def prepare_data(df, df2, annotation="tumor"):
    """Isotype fractions (one row per BM unit of one PC annotation) and the light chain fractions of these units."""
    idx = (df["annotation"] == annotation) & (df["sample_type"] == "BM")

    unit_annot = df.loc[idx, ["id", "disease_status", "cytogenetics"]].drop_duplicates().set_index("id", drop=True)

    mat = pd.DataFrame(index=df.loc[idx, "id"].unique(), columns=df["isotype_status"].cat.categories)
    for i, row in df[idx].iterrows():
        mat.loc[row["id"], row["isotype_status"]] = row["frac"]
        unit = df2["id"] == row["id"]
        unit_annot.loc[row["id"], "IgK"] = df2.loc[unit & (df2["chain_status_main"].str.endswith("IgK")), "frac"].sum()
        unit_annot.loc[row["id"], "IgL"] = df2.loc[unit & (df2["chain_status_main"].str.endswith("IgL")), "frac"].sum()

    return unit_annot, mat


unit_annot, mat = prepare_data(isotype_status, chain_status, "tumor")
mat.loc[unit_annot.index, "IgK"] = list(unit_annot["IgK"])
mat.loc[unit_annot.index, "IgL"] = list(unit_annot["IgL"])

for i, row in mat.iterrows():
    mat.loc[i, "disease_status"] = list(isotype_status.loc[isotype_status["id"] == i, "disease_status"])[0]
    mat.loc[i, "sample"] = list(isotype_status.loc[isotype_status["id"] == i, "sample"])[0]
    mat.loc[i, "cytogenetics"] = list(isotype_status.loc[isotype_status["id"] == i, "cytogenetics"])[0]
# individual of each unit (second field of the unit name)
mat["redcap_id"] = [x.split(":")[1] for x in mat.index]

# ---- 5. Immunofixation concordance table -------------------------------------------------------
# clinical immunofixation of the BM libraries, matched to the units on tissue, individual and disease stage
meta = meta[~meta["IFX"].isna() & meta["redcap_id"].isin(mat["redcap_id"].unique()) & (meta["sample_type"] == "BM")]
meta.index = meta.apply(lambda x: ":".join([x["sample_type"], x["redcap_id"], x["disease_status"]]), axis=1)
meta = meta[["redcap_id", "disease_status", "IFX"]].drop_duplicates()

ig_order = ["IgG", "IgA", "None", "IgD", "IgE", "IgM", "IgM/IgD"]
manual = mat[ig_order + ["IgK", "IgL", "disease_status", "cytogenetics", "redcap_id", "sample"]].join(
    meta[["IFX"]], on="sample").copy()
manual[["IFX", "IgG", "IgA", "None", "IgD", "IgM", "IgM/IgD", "IgK", "IgL", "disease_status", "cytogenetics"]].to_csv(
    data("vdj", "ifx_concordance_111824.csv"))

# ---- 6. Isotype table: clonal heavy and light chain of each patient ----------------------------
# the table as written (fractions read back as floats)
manual = pd.read_csv(data("vdj", "ifx_concordance_111824.csv"), index_col=0)

light_chain = ["IgK", "IgL"]
heavy_chain = ["IgG", "IgA", "None", "IgD", "IgM", "IgM/IgD"]
for idx, row in manual.iterrows():
    clonal_hc = heavy_chain[np.argmax(row[heavy_chain])]
    clonal_lc = light_chain[np.argmax(row[light_chain])]
    manual.loc[idx, "sc_clonal_heavy_chain"] = clonal_hc
    manual.loc[idx, "sc_clonal_light_chain"] = clonal_lc

manual["redcap_id"] = [x.split(":")[1] for x in manual.index]

redcap_based_clones = (manual[["redcap_id", "IFX", "sc_clonal_heavy_chain", "sc_clonal_light_chain"]]
                       .drop_duplicates().set_index("redcap_id"))
redcap_based_freq = manual[["redcap_id"] + heavy_chain + light_chain].groupby("redcap_id").mean()
bcr_res = redcap_based_clones.join(redcap_based_freq)
bcr_res.to_csv(data("metadata", "isotype-final-250409.csv"))
print("malignant BM units: %d, patients: %d" % (manual.shape[0], bcr_res.shape[0]))
