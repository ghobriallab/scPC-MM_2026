#!/usr/bin/env python3
"""Genome-wide CNV maps of the CNV clones on 5-kb genome bins and on cytobands (Methods, Copy number variant
detection).

Purpose:      Bins: 5-kb bins of chromosomes 1-22 (GRCh38 lengths), mapped to cytobands with bedtools intersect (overlap
              length and fraction of the bin). Segment map: the bins overlapped by each amp, del, bamp or bdel segment
              (bedtools intersect). For every clone of the clone phylogeny table (rows stage:tissue:clone_id, sorted by
              tumor and depth), each bin and each cytoband covered by one of the clone's CNV segments (GT) is +1 (amp,
              bamp) or -1 (del, bdel), 0 elsewhere; a second pair of matrices keeps only the segments the clone gained
              relative to its parent (subGT). Cytoband matrices: autosomal bands in genome order. Must be run with
              PYTHONHASHSEED=0.

Inputs:       data/cnv/cnv_segments_152_tumors.tsv (10_clone_driver_arms.py): chr, start, end, seg_id, cnv_state,
              tumor_name, cytoband.
              data/reference/refdata-gex-GRCh38-2020-A/genes/cytoBand.txt (chromosome, start, end, band, stain).
              data/reference/chrom_length.txt: chromosome lengths (GRCh38).
              data/cnv/clone-phylogeny-250321.pkl (11_clone_phylogeny.py): clone_id, tumor_name, disease_status,
              sample_type, depth, GT, subGT.

Outputs:      data/cnv/genomic_bins/: genome_5k_bins.bed (chr, start, end), consensus_segs.bed (chr, start, end, seg_id,
              cnv_state, tumor_name), cytoband.bed (chr, start, end, band), genome_5k_bins_cytoband.tsv (bin, band,
              overlap length, fraction of the bin), seg_genome_map.pkl (segment -> bins), all_cnv_genome_5k_bins.pkl and
              unique_cnv_genome_5k_bins.pkl (clones x bins), all_cnv_cytoband_bins.pkl and unique_cnv_cytoband_bins.pkl
              (clones x cytobands).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); bedtools; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import pickle
import shlex
import subprocess

import numpy as np
import pandas as pd

np.random.seed(0)

BINS = data("cnv", "genomic_bins")
BINS.mkdir(parents=True, exist_ok=True)


def q(path):
    """A path quoted for the shell."""
    return shlex.quote(str(path))


# ---- 1. segments and cytobands ---------------------------------------------------------------------------------------
segs = pd.read_csv(data("cnv", "cnv_segments_152_tumors.tsv"), sep="\t")
segs.index = segs.apply(lambda x: x["seg_id"] + "__" + x["tumor_name"], axis=1)

cytoband = pd.read_csv(data("reference", "refdata-gex-GRCh38-2020-A", "genes", "cytoBand.txt"), sep="\t", header=None)
cytoband = cytoband[~cytoband[3].isna()]
cytoband.rename(columns={0: "chr", 1: "start", 2: "end", 3: "band", 4: "."}, inplace=True)
cytoband["chr"] = [c.replace("chr", "") for c in cytoband["chr"]]
cytoband["band"] = np.array(cytoband["chr"].astype(str)) + cytoband["band"]

# ---- 2. 5-kb genome bins of the autosomes ----------------------------------------------------------------------------
bin_length = 5000
genome_bins = {"chr": [], "start": [], "end": []}

for line in open(data("reference", "chrom_length.txt")):
    chrom, length = line.rstrip().split("\t")

    # remove X, Y and M
    if chrom in ["chrX", "chrY", "chrM"]:
        continue

    chrom = int(chrom.replace("chr", ""))
    length = int(length)

    for pos in range(0, length, bin_length):
        genome_bins["chr"].append(chrom)
        genome_bins["start"].append(pos)
        genome_bins["end"].append(min(pos + bin_length, length))
genome_bins = pd.DataFrame(genome_bins)
genome_bins.to_csv(BINS / "genome_5k_bins.bed", header=False, index=False, sep="\t")

segs_bed = segs.loc[segs["cnv_state"].isin(["amp", "del", "bamp", "bdel"]),
                    ["chr", "start", "end", "seg_id", "cnv_state", "tumor_name"]]
segs_bed.to_csv(BINS / "consensus_segs.bed", header=False, index=False, sep="\t")

cytoband[["chr", "start", "end", "band"]].to_csv(BINS / "cytoband.bed", header=False, index=False, sep="\t")

# cytoband of each bin, with the overlap length and its fraction of the bin
cmd = "bedtools intersect -a %s -b %s -wo" % (q(BINS / "genome_5k_bins.bed"), q(BINS / "cytoband.bed"))

fo = open(BINS / "genome_5k_bins_cytoband.tsv", "w")
for line in subprocess.check_output(cmd, shell=True).decode("utf-8").rstrip("\n").split("\n"):
    line = line.split("\t")
    chrom = line[0]
    start = line[1]
    end = line[2]
    band = line[6]
    match = line[7]
    weight = str(float(line[7]) / bin_length)
    fo.write("\t".join(["{}:{}-{}".format(chrom, start, end), band, match, weight]) + "\n")
fo.close()

# bins overlapped by each CNV segment
cmd = "bedtools intersect -a %s -b %s -wo" % (q(BINS / "consensus_segs.bed"), q(BINS / "genome_5k_bins.bed"))

seg_genome_map = {}
for line in subprocess.check_output(cmd, shell=True).decode("utf-8").rstrip("\n").split("\n"):
    line = line.split("\t")
    seg_id = line[3] + "__" + line[5]
    cnv_state = line[4]
    pos = "{}:{}-{}".format(line[6], line[7], line[8])

    seg_genome_map.setdefault(seg_id, [])
    seg_genome_map[seg_id].append(pos)

fo = open(BINS / "seg_genome_map.pkl", "wb")
pickle.dump(seg_genome_map, fo)
fo.close()

# ---- 3. genome-bin matrices of the clones ----------------------------------------------------------------------------
genome_bins = pd.read_csv(BINS / "genome_5k_bins_cytoband.tsv", sep="\t", header=None)
genome_bins.rename(columns={0: "bin", 1: "band", 2: "length", 3: "weight"}, inplace=True)

clones = pd.read_pickle(data("cnv", "clone-phylogeny-250321.pkl"))
clones["tumor_id"] = clones.apply(lambda x: x["disease_status"] + ":" + x["sample_type"] + ":" + x["clone_id"], axis=1)
clones = clones.sort_values(["tumor_name", "depth"]).reset_index(drop=True)

genome_mat = pd.DataFrame(index=clones["tumor_id"].values, columns=genome_bins["bin"].values).fillna(0)
subgen_mat = pd.DataFrame(index=clones["tumor_id"].values, columns=genome_bins["bin"].values).fillna(0)

unit = {"amp": 1, "bamp": 1, "del": -1, "bdel": -1}

for _, row in clones[~clones["GT"].isna()].iterrows():
    subGT = [gt.split("_")[0] for gt in row["subGT"].split(",")]

    for gt in row["GT"].split(","):
        gt = gt.split("_")[0]
        seg_id = gt + "__" + row["tumor_name"]

        cnv_state = segs.loc[seg_id, "cnv_state"]
        if not cnv_state in unit:
            continue

        # segment gained by this clone
        fill_subGT = False
        if gt in subGT:
            fill_subGT = True

        # all CNVs, and CNVs gained by the clone
        pos = seg_genome_map.get(seg_id, [])
        if len(pos) == 0:
            continue

        genome_mat.loc[row["tumor_id"], pos] = unit[cnv_state]
        if fill_subGT:
            subgen_mat.loc[row["tumor_id"], pos] = unit[cnv_state]

genome_mat.to_pickle(BINS / "all_cnv_genome_5k_bins.pkl")
subgen_mat.to_pickle(BINS / "unique_cnv_genome_5k_bins.pkl")

# ---- 4. cytoband matrices of the clones ------------------------------------------------------------------------------
cytoband = cytoband[~cytoband["chr"].isin(["Y", "X", "M"])]
cytoband["chr"] = cytoband["chr"].astype(int)
cytoband = cytoband.sort_values(["chr", "start", "end"])

cytoband_mat = pd.DataFrame(index=clones["tumor_id"].values, columns=cytoband["band"].values).fillna(0)
subcytob_mat = pd.DataFrame(index=clones["tumor_id"].values, columns=cytoband["band"].values).fillna(0)

for _, row in clones[~clones["GT"].isna()].iterrows():
    subGT = [gt.split("_")[0] for gt in row["subGT"].split(",")]

    for gt in row["GT"].split(","):
        gt = gt.split("_")[0]
        seg_id = gt + "__" + row["tumor_name"]

        cnv_state = segs.loc[seg_id, "cnv_state"]
        if not cnv_state in unit:
            continue

        cnv_cytobands = segs.loc[seg_id, "cytoband"].split(",")

        # segment gained by this clone
        fill_subGT = False
        if gt in subGT:
            fill_subGT = True

        cytoband_mat.loc[row["tumor_id"], cnv_cytobands] = unit[cnv_state]
        if fill_subGT:
            subcytob_mat.loc[row["tumor_id"], cnv_cytobands] = unit[cnv_state]

cytoband_mat.to_pickle(BINS / "all_cnv_cytoband_bins.pkl")
subcytob_mat.to_pickle(BINS / "unique_cnv_cytoband_bins.pkl")
