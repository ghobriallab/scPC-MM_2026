#!/usr/bin/env python3
"""Association of the WGS regression residuals with disease stage (Supplementary Fig. 15C,D).

Purpose:      The MST score of the 84 patients with WGS was regressed on the WGS MM-like score (least squares) and on the
              WGS alterations (LASSO) in 01_extract_tables.py. This step tests whether the residuals of each model, the part
              of the MST score that the genomic data do not explain, still vary with disease stage: Spearman's correlation
              of the residuals with disease stage coded 0 (MGUS), 1 (LRSMM), 2 (IRSMM), 3 (HRSMM) and 4 (MM), two-sided.

Inputs:       results/figure_inputs/wgs_samples.tsv (01_extract_tables.py, part wgs): patient_id, disease_status (IMWG
              2/20/20 risk groups), lm_resid, lasso_resid.

Outputs:      results/figure_inputs/wgs_residual_stage.tsv (model, residual, n, rho, p), wgs_residuals_by_stage_log.txt.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy); config.py.
"""
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import result, result_dir

OUT = str(result_dir("figure_inputs"))
STAGE_CODE = {"MGUS": 0, "LRSMM": 1, "IRSMM": 2, "HRSMM": 3, "MM": 4}

w = pd.read_csv(result("figure_inputs", "wgs_samples.tsv"), sep="\t")
assert len(w) == 84 and w["disease_status"].isin(list(STAGE_CODE)).all()
stage = w["disease_status"].map(STAGE_CODE)

rows = []
for model, col in [("linear regression on the WGS MM-like score", "lm_resid"), ("LASSO on the WGS alterations", "lasso_resid")]:
    rho, p = spearmanr(stage, w[col])
    rows.append(dict(model=model, residual=col, n=len(w), rho=rho, p=p))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "wgs_residual_stage.tsv"), sep="\t", index=False)

with open(os.path.join(OUT, "wgs_residuals_by_stage_log.txt"), "w") as log:
    log.write("patients by stage: %s\n" % w["disease_status"].value_counts().reindex(list(STAGE_CODE)).to_dict())
    for r in rows:
        line = "%s: Spearman rho with disease stage = %.4f, P = %.3g (n = %d)" % (r["model"], r["rho"], r["p"], r["n"])
        print(line)
        log.write(line + "\n")
