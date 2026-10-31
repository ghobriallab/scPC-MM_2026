#!/usr/bin/env python3
"""Genes per signature, genes shared between signatures and Spearman correlations between the activities of the 16 signatures across the BM
tumors (Supplementary Fig. 7D; Results; Methods, detection of single-cell transcriptional signatures).

Purpose:      1. Genes per signature and genes assigned to more than one signature in the consensus signature table. 2. Spearman correlation of
              the 16 signature activities (mean log-normalized expression of the signature genes, averaged over the
              cells of a unit) across malignant BM PCs, one unit per patient and disease stage with >= 30 cells (SMM risk
              groups of the IMWG 2/20/20 model, risk.py; units that share a group are pooled), with Benjamini-Hochberg correction over the 120 pairs. 3. Because the activity of every signature
              rises with sequencing depth, partial Spearman correlations given the unit's median UMIs per cell (ranks of
              both signatures regressed on the rank of the median UMIs; Pearson correlation of the residuals).

Inputs:       data/paper/signatures/consensus-modules_genes.csv: index = gene, context = signature.
              data/paper/signatures/per-sample-mean-expr-score.pkl: mean_expr:<signature>, patient_id, sample_type,
              annotation (tumor/normal), disease_status.
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: redcap_id, disease_status, sample_type, annotation,
              total_counts (cells and median UMIs per unit).
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id -> patient_id.

Outputs:      results/signatures/signature_sizes.tsv: genes per signature.
              results/signatures/signature_shared_genes.tsv: genes assigned to two signatures.
              results/signatures/signature_correlation_pairs.tsv: each pair: rho, P, q, partial rho, partial P and q.
              results/signatures/signature_correlations_malignant.tsv: Spearman rho, 16 x 16, across the BM tumors.
              results/signatures/signature_partial_correlations_malignant.tsv: partial rho given median UMIs, 16 x 16.
              results/signatures/signature_correlations_log.txt: counts and the strongest pairs.

Dependencies: Python 3.8 with requirements.txt (pandas, numpy, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import pool_units, stages

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = result_dir("signatures")
FH = open(os.path.join(OUT, "signature_correlations_log.txt"), "w")
MIN_CELLS = 30
NAMES = {"B_cell_activation": "B cell activation", "FOS-JUN_stress_response": "FOS-JUN stress response",
         "HRD_signature": "HRD signature", "MAF*_signature": "MAF signature", "MCL1-BCL2_signature": "MCL1-BCL2 signature",
         "MHC-II_immune_checkpoint": "MHC-II immune checkpoint", "MHC-I_gene_module": "MHC-I gene module",
         "cell_cycling_G1S": "Cell cycling G1/S", "cell_cycling_G2M": "Cell cycling G2/M",
         "circulatory_capacity": "Circulatory capacity", "extracellular_matrix": "Extracellular matrix",
         "histone_chaperone_gene": "Histone/chaperone", "interferon_inducible": "Interferon inducible",
         "myeloma_oncogene": "Myeloma oncogene", "secretory_granule": "Secretory granule",
         "vacuole_ER_signaling": "Vacuole/ER signaling"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


def partial_resid(x, z):
    """residual of rank(x) after linear regression on rank(z)"""
    rx, rz = stats.rankdata(x), stats.rankdata(z)
    b = np.polyfit(rz, rx, 1)
    return rx - np.polyval(b, rz)


# ---- 1. genes shared between signatures ---------------------------------------------------------------------------------
sig = pd.read_csv(data("paper", "signatures", "consensus-modules_genes.csv"), index_col=0)
sig["signature"] = sig["context"].map(NAMES)
shared = sig[sig.index.duplicated(keep=False)].reset_index().sort_values(["gene", "signature"])
shared[["gene", "signature"]].to_csv(os.path.join(OUT, "signature_shared_genes.tsv"), sep="\t", index=False)
log("signature gene assignments %d, unique genes %d, genes in two signatures %d:" % (len(sig), sig.index.nunique(),
                                                                                  shared["gene"].nunique()))
for g, d in shared.groupby("gene"):
    log("  %s: %s" % (g, " and ".join(d["signature"])))
sizes = sig.reset_index().groupby("signature")["gene"].nunique().sort_values()
sizes.rename("genes").to_csv(os.path.join(OUT, "signature_sizes.tsv"), sep="\t")
log("genes per signature: %d-%d (%s)" % (sizes.min(), sizes.max(), ", ".join("%s %d" % kv for kv in sizes.items())))

# ---- 2. units and cell counts -------------------------------------------------------------------------------------------
m = pd.read_pickle(data("paper", "signatures", "per-sample-mean-expr-score.pkl"))
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells[["redcap_id", "disease_status", "sample_type", "annotation", "total_counts"]]
cells[["redcap_id", "disease_status", "sample_type", "annotation"]] = cells[["redcap_id", "disease_status", "sample_type", "annotation"]].astype(str)
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"), dtype=str)
cells["patient_id"] = cells["redcap_id"].map(meta[["redcap_id", "patient_id"]].dropna().drop_duplicates()
                                              .set_index("redcap_id")["patient_id"])
n = cells.groupby(["patient_id", "disease_status", "sample_type", "annotation"]).agg(
    n_cells=("total_counts", "size"), umi=("total_counts", "median")).reset_index()
m = m.merge(n, on=["patient_id", "disease_status", "sample_type", "annotation"], how="left")
# IMWG risk groups; units that then share patient, stage, tissue and PC status are pooled (cell-count weighted means), and
# the median UMIs are those of the pooled cells
KEYS = ["patient_id", "disease_status", "sample_type", "annotation"]
m["disease_status"] = stages(m["patient_id"], m["sample_type"], m["disease_status"])
m = pool_units(m[m["n_cells"].notna()], KEYS, "n_cells", [c for c in m.columns if c.startswith("mean_expr:")])
cells["disease_status"] = stages(cells["patient_id"], cells["sample_type"], cells["disease_status"])
umi = cells.groupby(KEYS)["total_counts"].median()
m["umi"] = [umi[tuple(k)] for k in m[KEYS].values]
COLS = ["mean_expr:" + k for k in NAMES]
SETS = {"malignant BM PCs": m[(m["sample_type"] == "BM") & (m["annotation"] == "tumor") & (m["n_cells"] >= MIN_CELLS)]}

# ---- 3. correlations ----------------------------------------------------------------------------------------------------
rows = []
for name, d in SETS.items():
    log("\n%s: %d units from %d individuals" % (name, len(d), d["patient_id"].nunique()))
    rho = d[COLS].rename(columns=lambda c: NAMES[c.split(":", 1)[1]]).corr(method="spearman")
    rho.to_csv(os.path.join(OUT, "signature_correlations_%s.tsv" % name.split()[0]), sep="\t")
    sub = []
    for a, b in itertools.combinations(rho.columns, 2):
        r, pv = stats.spearmanr(d["mean_expr:" + [k for k, v in NAMES.items() if v == a][0]],
                                d["mean_expr:" + [k for k, v in NAMES.items() if v == b][0]])
        ra, rb = (partial_resid(d["mean_expr:" + [k for k, v in NAMES.items() if v == x][0]], d["umi"]) for x in (a, b))
        pr, pp = stats.pearsonr(ra, rb)
        sub.append({"set": name, "signature_1": a, "signature_2": b, "rho": r, "P": pv, "partial_rho_umi": pr,
                    "partial_P": pp, "units": len(d)})
    sub = pd.DataFrame(sub)
    sub["q"] = multipletests(sub["P"], method="fdr_bh")[1]
    sub["partial_q"] = multipletests(sub["partial_P"], method="fdr_bh")[1]
    um = {c: stats.spearmanr(d[c], d["umi"])[0] for c in COLS}
    log("Spearman rho of each signature with the median UMIs per cell: median %.2f (range %.2f to %.2f)" % (
        np.median(list(um.values())), min(um.values()), max(um.values())))
    log("partial correlations given UMIs: pairs with q < 0.05: %d of 120 (positive %d, negative %d); median |rho| %.2f" % (
        (sub["partial_q"] < 0.05).sum(), ((sub["partial_q"] < 0.05) & (sub["partial_rho_umi"] > 0)).sum(),
        ((sub["partial_q"] < 0.05) & (sub["partial_rho_umi"] < 0)).sum(), sub["partial_rho_umi"].abs().median()))
    pm = rho.copy()
    for _, r in sub.iterrows():
        pm.loc[r["signature_1"], r["signature_2"]] = pm.loc[r["signature_2"], r["signature_1"]] = r["partial_rho_umi"]
    pm.to_csv(os.path.join(OUT, "signature_partial_correlations_%s.tsv" % name.split()[0]), sep="\t")
    rows.append(sub)
    s = sub.reindex(sub["rho"].abs().sort_values(ascending=False).index)
    log("pairs with q < 0.05: %d of 120 (positive %d, negative %d); |rho| >= 0.5: %d" % (
        (sub["q"] < 0.05).sum(), ((sub["q"] < 0.05) & (sub["rho"] > 0)).sum(), ((sub["q"] < 0.05) & (sub["rho"] < 0)).sum(),
        (sub["rho"].abs() >= 0.5).sum()))
    log("median |rho| %.2f; strongest pairs:" % sub["rho"].abs().median())
    for _, r in s.head(8).iterrows():
        log("  %-26s %-26s rho %.2f (q = %.2g); partial %.2f" % (r["signature_1"], r["signature_2"], r["rho"], r["q"],
                                                                 r["partial_rho_umi"]))
    sp_ = sub.reindex(sub["partial_rho_umi"].abs().sort_values(ascending=False).index)
    log("strongest partial correlations:")
    for _, r in sp_.head(6).iterrows():
        log("  %-26s %-26s partial %.2f (q = %.2g)" % (r["signature_1"], r["signature_2"], r["partial_rho_umi"], r["partial_q"]))
    # three pairs of related signatures
    for a, b in [("Cell cycling G1/S", "Cell cycling G2/M"), ("B cell activation", "MHC-II immune checkpoint"),
                 ("FOS-JUN stress response", "MCL1-BCL2 signature")]:
        r = sub[((sub["signature_1"] == a) & (sub["signature_2"] == b)) | ((sub["signature_1"] == b) & (sub["signature_2"] == a))].iloc[0]
        log("  pair %s and %s: rho %.2f (q = %.2g), rank %d of 120 by |rho|; partial %.2f (q = %.2g), rank %d" % (
            a, b, r["rho"], r["q"], 1 + list(s.index).index(r.name), r["partial_rho_umi"], r["partial_q"],
            1 + list(sp_.index).index(r.name)))
pd.concat(rows).to_csv(os.path.join(OUT, "signature_correlation_pairs.tsv"), sep="\t", index=False)
FH.close()
