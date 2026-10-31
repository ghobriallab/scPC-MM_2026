#!/usr/bin/env python3
"""Consensus CNV segments of the tumors with CNVs, with the genes, cytobands and chromosome arms they cover (Methods,
Copy number variant detection; segment table read by every later CNV step).

Purpose:      Reads the Numbat consensus segments (iteration 2) of every tumor and keeps tumors whose record ID (from
              the tumor name) has cells in the plasma cell object, and segments longer than 1 bp. Genes: segments
              intersected with the gene annotation (bedtools intersect -wo); a gene is assigned to a segment when
              >= 30% of its length overlaps it. Cytobands and arms: segments intersected with the cytobands and with the
              chromosome arms (p and q arm limits taken from the cytoband table); a segment lists every cytoband it
              overlaps and every arm it covers over > 75% of the arm length. Also writes the genes of all amp, del, bamp
              and bdel segments and each segment's length as a fraction of its chromosome. Must be run with
              PYTHONHASHSEED=0.

Inputs:       data/amp1q/cnv/SubcloneDetection/<tumor_name>/segs_consensus_2.tsv (Numbat): sample, CHROM, seg_start,
              seg_end, seg_cons, cnv_state_post.
              data/h5/agg/plasma-cell-GEX5-counts-250207.h5ad: obs redcap_id.
              data/reference/refdata-gex-GRCh38-2020-A/genes/genes.bed (gene coordinates and names) and cytoBand.txt
              (chromosome, start, end, band, stain).
              data/reference/chrom_length.txt: chromosome lengths (GRCh38).

Outputs:      data/cnv/cnv_segments_152_tumors.tsv: one row per segment: chr, start, end, seg_id, seg_length,
              cnv_state, clone_id, tumor_name, redcap_id, cytoband, arm, n_genes, genes, seg_chr_ratio.
              data/cnv/intersected.bed and data/cnv/cytoband_intersected.bed: bedtools overlaps.
              data/reference/refdata-gex-GRCh38-2020-A/genes/chrom_arms.bed: chromosome arms (chr, start, end, arm).
              data/cnv/cnv_genes.txt: genes of the amp, del, bamp and bdel segments, one per line.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scanpy); bedtools, bash, grep, awk, cut, ls; config.py.
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

np.random.seed(0)

NUMBAT = data("amp1q", "cnv", "SubcloneDetection")
GENES = data("reference", "refdata-gex-GRCh38-2020-A", "genes")
SEGS = data("cnv", "cnv_segments_152_tumors.tsv")
GENE_HITS = data("cnv", "intersected.bed")
BAND_HITS = data("cnv", "cytoband_intersected.bed")
ARMS = GENES / "chrom_arms.bed"

# segment chromosomes have no 'chr' prefix: add it before intersecting with the reference files
ADD_CHR = r'''awk -F "\t" '{$1="chr"$1; print $0}' OFS="\t"'''
# cytoband names prefixed with their chromosome (e.g. 1p36.33); rows without a band name dropped
BAND_NAMES = r'''awk -F "\t" '{if($4!=""){split($1,a,"chr"); $4=a[2]$4; print $0}}' OFS="\t"'''


def q(path):
    """A path quoted for the shell."""
    return shlex.quote(str(path))


def bash(cmd):
    subprocess.run(["bash", "-c", cmd], check=True)


# ---- 1. consensus segments of every tumor ----------------------------------------------------------------------------
seg_files = subprocess.check_output("ls -1 %s/*/segs_consensus_2.tsv" % q(NUMBAT), shell=True)
seg_files = seg_files.decode("utf-8").rstrip("\n").split("\n")

a = sc.read(str(data("h5", "agg", "plasma-cell-GEX5-counts-250207.h5ad")), backed="r")

segs = []
columns = ["sample", "CHROM", "seg_start", "seg_end", "seg_cons", "cnv_state_post"]
for seg_file in seg_files:
    tumor_name = os.path.dirname(seg_file).split("/")[-1]

    # tumor names: record ID with '-' for '/', then '_Tumor'; keep tumors with cells in the plasma cell object
    redcap_id = tumor_name.split("_Tumor")[0].replace("-", "/")
    if sum(redcap_id == a.obs["redcap_id"]) == 0:
        continue

    seg = pd.read_csv(seg_file, sep="\t")
    seg = seg.loc[~seg["sample"].isna(), columns]
    seg["tumor_name"] = tumor_name
    seg["redcap_id"] = redcap_id
    segs.append(seg.copy())

segs = pd.concat(segs)
segs.rename(columns={"sample": "clone_id", "CHROM": "chr", "seg_start": "start", "seg_end": "end",
                     "seg_cons": "seg_id", "cnv_state_post": "cnv_state"},
            inplace=True)
segs["seg_length"] = segs["end"] - segs["start"] + 1
segs = segs[["chr", "start", "end", "seg_id", "seg_length", "cnv_state", "clone_id", "tumor_name",
             "redcap_id"]].reset_index(drop=True)

# drop 1-bp segments
segs = segs[segs["seg_length"] > 1]
segs.to_csv(SEGS, sep="\t", index=False)

# ---- 2. genes covered by each segment --------------------------------------------------------------------------------
bash("grep -v ^chr %s | bedtools sort | %s | bedtools intersect -a stdin -b %s -wo > %s"
     % (q(SEGS), ADD_CHR, q(GENES / "genes.bed"), q(GENE_HITS)))

cnv_dict = dict()
for line in open(GENE_HITS):
    line = line.rstrip("\n").split("\t")
    # segment
    seg_id = line[3]
    cnv_state = line[5]
    tumor_name = line[7]

    # overlapping gene
    match_len = int(line[16])
    gene_len = int(line[11]) - int(line[10])
    gene_name = line[12]

    # the gene belongs to the segment when >= 30% of its length overlaps it
    if match_len / gene_len >= 0.3:
        cnv_dict.setdefault((tumor_name, cnv_state, seg_id), [])
        cnv_dict[(tumor_name, cnv_state, seg_id)].append(gene_name)

for key in cnv_dict.keys():
    idx = (segs["tumor_name"] == key[0]) & (segs["cnv_state"] == key[1]) & (segs["seg_id"] == key[2])
    segs.loc[idx, "n_genes"] = len(cnv_dict[key])
    segs.loc[idx, "genes"] = ",".join(cnv_dict[key])

segs.to_csv(SEGS, sep="\t", index=False)

# ---- 3. chromosome arms from the cytobands ---------------------------------------------------------------------------
cytoband = pd.read_csv(GENES / "cytoBand.txt", sep="\t", header=None)
cytoband = cytoband.rename(columns={0: "chr", 1: "start", 2: "end", 3: "band", 4: "type"})
cytoband = cytoband[~cytoband["band"].isna()]

arm_loci = {"chr": [], "start": [], "end": [], "arm": []}
for chrom, df in cytoband.groupby("chr"):
    # p arm
    if sum(df["band"].str.startswith("p")) > 0:
        arm_loci["chr"].append(chrom)
        arm_loci["start"].append(df.loc[df["band"].str.startswith("p"), "start"].min())
        arm_loci["end"].append(df.loc[df["band"].str.startswith("p"), "end"].max())
        arm_loci["arm"].append(chrom.replace("chr", "") + "p_arm")

    # q arm
    if sum(df["band"].str.startswith("p")) > 0:
        arm_loci["chr"].append(chrom)
        arm_loci["start"].append(df.loc[df["band"].str.startswith("q"), "start"].min())
        arm_loci["end"].append(df.loc[df["band"].str.startswith("q"), "end"].max())
        arm_loci["arm"].append(chrom.replace("chr", "") + "q_arm")
arm_loci = pd.DataFrame(arm_loci)
arm_loci.to_csv(ARMS, sep="\t", index=False, header=False)

# ---- 4. cytobands and arms covered by each segment -------------------------------------------------------------------
bash("grep -v ^chr %s | cut -f1-8 | %s | bedtools sort | "
     "bedtools intersect -a stdin -b <(cut -f1-4 %s | %s | cat - %s | bedtools sort) -wo > %s"
     % (q(SEGS), ADD_CHR, q(GENES / "cytoBand.txt"), BAND_NAMES, q(ARMS), q(BAND_HITS)))

segs = pd.read_csv(SEGS, sep="\t")
segs.index = segs.apply(lambda x: "__".join(map(str, [x["chr"], x["start"], x["end"], x["seg_id"], x["cnv_state"],
                                                       x["tumor_name"]])), axis=1)

overlaps = pd.read_csv(BAND_HITS, sep="\t", header=None)
overlaps[0] = overlaps[0].str.replace("chr", "")
overlaps["id"] = overlaps.apply(lambda x: "__".join(map(str, [x[0], x[1], x[2], x[3], x[5], x[7]])), axis=1)
overlaps = overlaps[[8, 9, 10, 11, 12, "id"]].rename(columns={8: "chr", 9: "start", 10: "end", 11: "cytoband",
                                                              12: "overlap_len"})

# arms covered over > 75% of their length
cutoff = 0.75
arm_events = {"id": [], "arm": []}
for id_name, df in overlaps[overlaps["cytoband"].str.endswith("_arm")].groupby("id"):
    passed = list(df.loc[df["overlap_len"] / (df["end"] - df["start"]) > cutoff, "cytoband"])
    if len(passed) == 0:
        continue
    cytobands = ",".join(sorted(passed))
    arm_events["id"].append(id_name)
    arm_events["arm"].append(cytobands)
arm_events = pd.DataFrame(arm_events)

# every cytoband a segment overlaps
focal_events = {"id": [], "cytoband": []}
idx = ~overlaps["cytoband"].str.endswith("_arm")
for id_name, df in overlaps[idx].groupby("id"):
    cytobands = list(df["cytoband"].unique())
    if len(cytobands) == 0:
        continue
    focal_events["id"].append(id_name)
    focal_events["cytoband"].append(",".join(cytobands))
focal_events = pd.DataFrame(focal_events)

events = focal_events.join(arm_events.set_index("id"), on="id")
segs.loc[list(events["id"]), "cytoband"] = list(events["cytoband"])
segs.loc[list(events["id"]), "arm"] = list(events["arm"])

columns = ["chr", "start", "end", "seg_id", "seg_length", "cnv_state",
           "clone_id", "tumor_name", "redcap_id", "cytoband", "arm", "n_genes", "genes"]
segs = segs[columns]
segs = segs.reset_index(drop=True)
segs.to_csv(SEGS, sep="\t", index=False)

# ---- 5. genes of the amp, del, bamp and bdel segments ----------------------------------------------------------------
segs = pd.read_csv(SEGS, sep="\t")
segs = segs.loc[~segs["genes"].isna() & segs["cnv_state"].isin(["amp", "del", "bamp", "bdel"])]

genes = set()
for gene_str in list(segs["genes"].unique()):
    genes = genes.union(set(gene_str.split(",")))
genes = sorted(genes)
print("genes in CNV segments:", len(genes))

fo = open(data("cnv", "cnv_genes.txt"), "w")
for gene in genes:
    fo.write(gene + "\n")
fo.close()

# ---- 6. segment length as a fraction of the chromosome ---------------------------------------------------------------
segs = pd.read_csv(SEGS, sep="\t")

chrom_lengths = {}
for line in open(data("reference", "chrom_length.txt")):
    line = line.rstrip().split("\t")
    chrom_lengths[line[0].replace("chr", "")] = int(line[1])

segs["seg_chr_ratio"] = segs.apply(lambda x: (x["end"] - x["start"] + 1) / chrom_lengths[str(x["chr"])], axis=1)
segs.to_csv(SEGS, sep="\t", index=False)
