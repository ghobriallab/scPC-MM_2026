#!/usr/bin/env python3
"""CNV state and probability of every cell with a Numbat CNV call, per cytoband and chromosome arm, for all tumors
(Methods, Copy number variant detection).

Purpose:      Concatenates the per-tumor tables written by 06_cnv_cell_matrix_tumor.py (all files of
              data/amp1q/cnv/cnv_status, in the order listed by ls) into one cell-by-band matrix. Must be run with
              PYTHONHASHSEED=0.

Inputs:       data/amp1q/cnv/cnv_status/<tumor_name>.csv (06_cnv_cell_matrix_tumor.py): cell, <band> (CNV state),
              cnv_prob:<band> (posterior probability), tumor_id.

Outputs:      data/cnv/plasma-cell-cnv-cytoband-matrix-250109.pkl: one row per cell (index cell name), the same columns.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); ls; config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

import shlex
import subprocess

import numpy as np
import pandas as pd

np.random.seed(0)

# per-tumor tables
outs = subprocess.check_output("ls %s/*.csv" % shlex.quote(str(data("amp1q", "cnv", "cnv_status"))),
                               shell=True).decode("utf-8")
outs = outs.rstrip("\n").split("\n")

mat = []
for in_file in outs:
    mat.append(pd.read_csv(in_file, index_col=0))
mat = pd.concat(mat)

mat.to_pickle(data("cnv", "plasma-cell-cnv-cytoband-matrix-250109.pkl"))
