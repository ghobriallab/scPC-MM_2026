#!/usr/bin/env python3
"""Cohort BCR tables: combines the dandelion objects of all individuals and filters cells and gene calls (Methods,
scBCR-seq data preprocessing and BCR filtering).

Purpose:      Concatenates the per-individual dandelion objects and saves the combined object (as a pickle: the HDF5
              writer of dandelion overflows at this size) with its contig and cell tables. Then removes cells with
              several heavy chains or a heavy chain only (locus_status "Orphan Extra VDJ" or "Orphan IGH"), cells with
              no isotype call and no constant region call on either chain, and cells whose heavy chain constant region
              call is a light chain gene (IGK, IGL) or whose light chain call is a heavy chain gene (IGH); contigs
              follow their cells. Main loci: repeated loci collapsed (locus_VDJ_main, locus_VJ_main); a cell with both
              IGK and IGL light chains gets the locus supported by more of its main V, J and C calls (IGK|IGL if tied).
              chain_status_main = isotype_status + light chain (IgK or IgL). Main junctions (nucleotide and amino
              acid, both chains): the junction of the contig whose genotyped V call is the main V call. Finally removes
              cells with a pseudogene ("OR" in the main V or J call of either chain), and saves the filtered object and
              tables.

Inputs:       data/amp1q/bcr/firstpass_h5ddl/*_vdj.h5ddl (one dandelion object per individual, written by
              02_dandelion_clonotypes.py): cell table columns locus_status, isotype, isotype_status, locus_VDJ,
              locus_VJ, v_call_VDJ_main, j_call_VDJ_main, c_call_VDJ_main, v_call_VJ_main, j_call_VJ_main,
              c_call_VJ_main, v_call_genotyped_VDJ, v_call_genotyped_VJ, junction_VDJ, junction_VJ, junction_aa_VDJ,
              junction_aa_VJ; contig table column cell_id.

Outputs:      data/amp1q/bcr/cohort_bcr_223pts.pkl, cohort_bcr_vdj_contig_data.csv, cohort_bcr_vdj_cell_data.csv: all
              cells, before filtering.
              data/vdj/cohort-BCR-filtered-223pts.pkl, cohort-BCR-filtered-contigs.csv, cohort-BCR-filtered-cells.csv:
              filtered cells and their contigs, with the columns locus_VDJ_main, locus_VJ_main, chain_status_main,
              junction_VDJ_main, junction_VJ_main, junction_aa_VDJ_main and junction_aa_VJ_main.

Dependencies: Python 3.8 with dandelion 0.3.3 and its dependencies (pandas 2.0.3, numpy 1.24.4, scipy 1.10.1);
              config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import glob
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import dandelion as ddl

BCR = data("amp1q", "bcr")
VDJ = data("vdj")
VDJ.mkdir(parents=True, exist_ok=True)

# ---- 1. Combine the dandelion objects of all individuals ---------------------------------------
paths = sorted(glob.glob(os.path.join(str(BCR / "firstpass_h5ddl"), "*_vdj.h5ddl")))
vdj = ddl.concat([ddl.read_h5ddl(path) for path in paths])

# the HDF5 writer of dandelion overflows at this size: pickle and CSV tables instead
with open(BCR / "cohort_bcr_223pts.pkl", "wb") as f:
    pickle.dump(vdj, f)
vdj.data.to_csv(BCR / "cohort_bcr_vdj_contig_data.csv")
vdj.metadata.to_csv(BCR / "cohort_bcr_vdj_cell_data.csv")

# ---- 2. Filter cells ---------------------------------------------------------------------------
# cells with several heavy chains or a heavy chain only
vdj.metadata = vdj.metadata[(vdj.metadata["locus_status"] != "Orphan Extra VDJ") &
                            (vdj.metadata["locus_status"] != "Orphan IGH")]
# cells with no isotype call and no constant region call on either chain
vdj.metadata = vdj.metadata[(vdj.metadata["isotype"] != "None") | (vdj.metadata["c_call_VDJ_main"] != "None") |
                            (vdj.metadata["c_call_VJ_main"] != "None")]
# cells whose constant region calls do not match the chain type
vdj.metadata = vdj.metadata[~(vdj.metadata["c_call_VDJ_main"].str.match("IGL") |
                              vdj.metadata["c_call_VDJ_main"].str.match("IGK") |
                              vdj.metadata["c_call_VJ_main"].str.match("IGH"))]
# contigs of the remaining cells
vdj.data = vdj.data[vdj.data["cell_id"].isin(vdj.metadata.index)]


# ---- 3. Main loci and chain status -------------------------------------------------------------
def remove_dups(x):
    """Repeated loci collapsed, e.g. IGK|IGK -> IGK (sorted, '|'-separated)."""
    return "|".join(sorted(set(x.split("|"))))


def count_VJ(genes):
    """Presence (0 or 1) of an IGK gene and of an IGL gene among the gene calls (','-separated)."""
    g = sorted(set([gene[:3] for gene in genes.split(",")]))
    return g.count("IGK"), g.count("IGL")


def pick_VJ_main(x):
    """Light chain locus of a cell: the locus supported by more of its main V, J and C calls."""
    if len(x["locus_VJ_main"].split("|")) == 1:
        return x["locus_VJ_main"]

    v = count_VJ(x["v_call_VJ_main"])
    j = count_VJ(x["j_call_VJ_main"])
    c = count_VJ(x["c_call_VJ_main"])

    IGK = v[0] + j[0] + c[0]
    IGL = v[1] + j[1] + c[1]

    if IGK > IGL:
        return "IGK"
    elif IGL > IGK:
        return "IGL"
    else:
        return "IGK|IGL"


vdj.metadata["locus_VDJ_main"] = list(vdj.metadata["locus_VDJ"].apply(remove_dups))
vdj.metadata["locus_VJ_main"] = list(vdj.metadata["locus_VJ"].apply(remove_dups))
vdj.metadata["locus_VJ_main"] = list(vdj.metadata.apply(pick_VJ_main, axis=1))
light = vdj.metadata["locus_VJ_main"].apply(lambda x: {"IGK": "IgK", "IGL": "IgL"}.get(x))
vdj.metadata["chain_status_main"] = np.array(vdj.metadata["isotype_status"]) + "+" + np.array(light)


# ---- 4. Main junction sequences ----------------------------------------------------------------
def curate_sequence(x, chain_type="VJ", seq_type="aa"):
    """Junction of the contig whose genotyped V call is the main V call of the cell."""
    if seq_type == "aa":
        seq = x["junction_aa_{}".format(chain_type)]
    elif seq_type == "nt":
        seq = x["junction_{}".format(chain_type)]
    else:
        raise Exception("unknown sequence type!")
    seq = seq.split("|")

    # position of the main V call among the genotyped V calls
    raw_call = x["v_call_genotyped_{}".format(chain_type)].split("|")
    main_index = raw_call.index(x["v_call_{}_main".format(chain_type)])

    return seq[main_index]


# nucleotide sequences
vdj.metadata["junction_VJ_main"] = vdj.metadata.apply(curate_sequence, chain_type="VJ", seq_type="nt", axis=1)
vdj.metadata["junction_VDJ_main"] = vdj.metadata.apply(curate_sequence, chain_type="VDJ", seq_type="nt", axis=1)
# amino acid sequences
vdj.metadata["junction_aa_VJ_main"] = vdj.metadata.apply(curate_sequence, chain_type="VJ", seq_type="aa", axis=1)
vdj.metadata["junction_aa_VDJ_main"] = vdj.metadata.apply(curate_sequence, chain_type="VDJ", seq_type="aa", axis=1)

# ---- 5. Remove cells with pseudogene calls -----------------------------------------------------
is_pseudo_hv = vdj.metadata["v_call_VDJ_main"].apply(lambda x: "OR" in x)
is_pseudo_hj = vdj.metadata["j_call_VDJ_main"].apply(lambda x: "OR" in x)
is_pseudo_lv = vdj.metadata["v_call_VJ_main"].apply(lambda x: "OR" in x)
is_pseudo_lj = vdj.metadata["j_call_VJ_main"].apply(lambda x: "OR" in x)
vdj.metadata = vdj.metadata[~(is_pseudo_hv | is_pseudo_hj)]
vdj.metadata = vdj.metadata[~(is_pseudo_lv | is_pseudo_lj)]
vdj.data = vdj.data[vdj.data["cell_id"].isin(vdj.metadata.index)]

# ---- 6. Save the filtered cohort tables --------------------------------------------------------
with open(VDJ / "cohort-BCR-filtered-223pts.pkl", "wb") as f:
    pickle.dump(vdj, f)
vdj.data.to_csv(VDJ / "cohort-BCR-filtered-contigs.csv")
vdj.metadata.to_csv(VDJ / "cohort-BCR-filtered-cells.csv")
print("cells: %d, contigs: %d" % (vdj.metadata.shape[0], vdj.data.shape[0]))
