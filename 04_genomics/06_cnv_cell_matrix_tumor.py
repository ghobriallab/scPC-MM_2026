#!/usr/bin/env python3
"""Cell-by-cytoband CNV table of one tumor from its Numbat cell posteriors (input of the CNV matrix of all cells,
08_cnv_cytoband_matrix.py; Methods, Copy number variant detection).

Purpose:      For the tumor named on the command line, reads the Numbat posteriors of iteration 2 per cell and segment
              (joint_post_2.tsv) and keeps amp, del, bamp and bdel calls (and, when the column cnv_state_expand exists,
              only rows whose expanded state is part of the segment label). One row per cell with such a call; for every
              cytoband and arm of each amp/del/bamp/bdel segment of the tumor in the segment table, the column
              cnv_prob:<band> holds the posterior p_cnv and the column <band> the CNV state when p_cnv >= 0.9
              (--cnv-cutoff), empty otherwise (probabilities default to 0). Columns: every band and arm of an
              amp/del/bamp/bdel segment of any tumor, arms first, each group in natural order; tumor_id =
              <record ID>:tumor:<0, or 1 for tumors named ..._Tumor2>. The script re-runs itself with
              PYTHONHASHSEED=0. Run it once for every tumor_name of data/cnv/cnv_segments_152_tumors.tsv
              (07_run_cnv_cell_matrix.sh).

Inputs:       data/cnv/cnv_segments_152_tumors.tsv (05_cnv_segments.py): seg_id, cnv_state, cytoband, tumor_name.
              data/amp1q/cnv/SubcloneDetection/<tumor_name>/joint_post_2.tsv (Numbat): cell, seg, seg_label,
              cnv_state, cnv_state_expand, p_cnv.

Outputs:      data/amp1q/cnv/cnv_status/<tumor_name>.csv (--out-path): one row per cell.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas) and natsort; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import argparse

hashseed = os.getenv("PYTHONHASHSEED")
if not hashseed:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import natsort
import numpy as np
import pandas as pd

np.random.seed(0)


def main(args):
    cnv_of_interest = ["amp", "del", "bamp", "bdel"]

    segs = pd.read_csv(data("cnv", "cnv_segments_152_tumors.tsv"), sep="\t")
    segs = segs.loc[segs["cnv_state"].isin(cnv_of_interest)]

    cnv_states = []
    for chunk in segs["cytoband"].unique():
        cnv_states.extend(chunk.split(","))
    cnv_states = list(set(cnv_states))

    # column order: arms, then cytobands
    cnv_states = natsort.natsorted([cnv for cnv in cnv_states if cnv.endswith("_arm")]) + \
        natsort.natsorted([cnv for cnv in cnv_states if not cnv.endswith("_arm")])
    cnv_columns = cnv_states + ["cnv_prob:" + cnv for cnv in cnv_states]

    tumor_name = args.tumor_name
    numbat_dir = data("amp1q", "cnv", "SubcloneDetection", tumor_name)
    cells = pd.read_csv(numbat_dir / "joint_post_2.tsv", sep="\t", index_col=0)
    cells = cells[cells["cnv_state"].isin(cnv_of_interest)]
    # outliers: keep rows whose expanded state is part of the segment label
    if "cnv_state_expand" in cells.columns:
        cells = cells[cells.apply(lambda x: x["cnv_state_expand"] in x["seg_label"], axis=1)]

    # one row per cell with a CNV call
    cell_ids = cells.index.unique()
    mat = pd.DataFrame(np.zeros((len(cell_ids), len(cnv_columns))), index=cell_ids, columns=cnv_columns)
    mat[cnv_states] = np.nan

    # tumor ID: <record ID>:tumor:<0 or 1>
    tumor_id = tumor_name.replace("-", "/").split("_Tumor")[0] + ":tumor:"
    tumor_id += "1" if tumor_name.endswith("2") else "0"
    mat["tumor_id"] = tumor_id

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, tumor_name + ".csv")

    # no CNVs
    if cells.shape[0] == 0:
        mat.to_csv(out_file)
        return

    # probabilities and calls per cytoband of each segment of the tumor
    for _, row in segs[segs["tumor_name"] == tumor_name].iterrows():
        df = cells[(cells["seg"] == row["seg_id"]) & (cells["cnv_state"] == row["cnv_state"])]
        for cytoband in row["cytoband"].split(","):
            mat.loc[df.index, "cnv_prob:" + cytoband] = list(df["p_cnv"])
            if sum(df["p_cnv"] >= args.cnv_cutoff) > 0:
                df = df[df["p_cnv"] >= args.cnv_cutoff]
                mat.loc[df.index, cytoband] = list(df["cnv_state"])
    mat.to_csv(out_file)


if __name__ == "__main__":
    description = "Create single-cell p_cnv matrix"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-path", help="output directory of matrix file",
                        default=str(data("amp1q", "cnv", "cnv_status")))
    parser.add_argument("--cnv-cutoff", help="CNV probability cutoff (default=0.9)", type=float, default=0.9)
    parser.add_argument("tumor_name", help="tumor folder in data/amp1q/cnv/SubcloneDetection with joint_post_2.tsv")

    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
