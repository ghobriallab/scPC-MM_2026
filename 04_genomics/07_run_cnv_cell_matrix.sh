#!/usr/bin/env bash
# ============================================================================
# Purpose:      Runs 06_cnv_cell_matrix_tumor.py once for every tumor with CNV segments, that is every distinct
#               tumor_name of data/cnv/cnv_segments_152_tumors.tsv (153 tumors of 152 patients).
# Inputs:       data/cnv/cnv_segments_152_tumors.tsv (05_cnv_segments.py): column tumor_name; located through config.py.
# Outputs:      data/amp1q/cnv/cnv_status/<tumor_name>.csv, one per tumor (written by 06_cnv_cell_matrix_tumor.py).
# Dependencies: bash; Python 3.8 with requirements.txt (pandas); config.py.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# tumor names of the segment table (no spaces in the names)
tumors=$(python3 -c '
import sys
sys.path.insert(0, "..")
from config import data
import pandas as pd
segs = pd.read_csv(data("cnv", "cnv_segments_152_tumors.tsv"), sep="\t")
print("\n".join(segs["tumor_name"].unique()))
')

for tumor_name in $tumors; do
    python3 06_cnv_cell_matrix_tumor.py "$tumor_name"
done
