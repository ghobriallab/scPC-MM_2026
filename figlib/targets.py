"""The sixteen therapeutic targets and the per-tumor differential expression shared by Fig. 2 and
Supplementary Fig. 6.

Purpose:      Per-tumor tables are genes x BM tumors named <patient>_BM_<stage>_tumor_<k> (two-sided Wilcoxon tests of
              malignant versus normal PCs of the same sample). Immunoglobulin genes and the minor tumors (tumor_1) are
              removed, and a target counts as differentially expressed in a tumor when q <= 0.01 and |log2 fold
              change| >= 1.

Inputs:       BM_scRNA_tumor_DE_FDR.csv and BM_scRNA_tumor_DE_logFoldChange.csv (results/differential_expression, IMWG
              2/20/20 risk groups; 05_differential_expression/15_tumor_de_imwg_tables.py).

Outputs:      de_tables(), significant_lfc(), per_cell() and the target lists, imported by the figure scripts.

Dependencies: numpy, pandas; paths.py.
"""
import numpy as np
import pandas as pd

import paths

TARGETS = ["TNFRSF17", "GPRC5D", "SDC1", "CD38", "SLAMF7", "FCRL5", "CD46", "CD47", "CD48", "NCAM1", "ITGB7", "CD74", "LY9",
           "TNFRSF13B", "CD19", "BCL2"]
TARGETS_SORTED = ["FCRL5", "SLAMF7", "CD48", "TNFRSF17", "GPRC5D", "CD47", "NCAM1", "SDC1", "CD46", "CD74", "BCL2", "ITGB7",
                  "TNFRSF13B", "LY9", "CD38", "CD19"]               # Fig. 2A row order (row clustering, checked in 02_Fig2.py)
Q_CUT, LFC_CUT = 0.01, 1.0


def de_tables():
    rd = lambda n: pd.read_csv(paths.result("differential_expression", n), index_col=0)
    fdr, lfc = rd("BM_scRNA_tumor_DE_FDR.csv"), rd("BM_scRNA_tumor_DE_logFoldChange.csv")
    keep_g = ~fdr.index.str.match("^IG[HKL]")
    keep_c = ~fdr.columns.str.endswith("tumor_1")
    fdr, lfc = fdr.loc[keep_g, fdr.columns[keep_c]], lfc.loc[keep_g, lfc.columns[keep_c]]
    fdr.index.name = lfc.index.name = None
    return fdr, lfc


def significant_lfc(fdr, lfc, genes):
    """genes x tumors: the log2 fold change where the gene is differentially expressed in the tumor, 0 elsewhere."""
    de = (fdr.loc[genes] <= Q_CUT) & (lfc.loc[genes].abs() >= LFC_CUT)
    return lfc.loc[genes].where(de, 0.0)


def tumor_key(col):
    """<patient>_BM_<stage>_tumor_<k> -> (tissue, stage, tumor id as in the cell tables: <patient>:tumor:<k>)."""
    info = col.split("_")
    return info[1], info[2], ":".join([info[0], info[-2], info[-1]])


def per_cell(tissue, stage, tumor_id, genes):
    """Per-cell value: the tumor's significant log2 fold change (0 if not differentially expressed), and
    NaN (drawn white) for cells of untested units: normal PCs, PB samples, minor tumors. Arguments are arrays of cell
    labels; returns (cells x genes array, table of the tumors used)."""
    fdr, lfc = de_tables()
    sig = significant_lfc(fdr, lfc, genes)
    key = pd.Series(np.arange(len(tumor_id))).groupby(
        [np.asarray(tissue), np.asarray(stage), np.asarray(tumor_id)]).apply(lambda s: s.values)
    out = np.full((len(tumor_id), len(genes)), np.nan, dtype=np.float32)
    rows = []
    for col in sig.columns:
        k = tumor_key(col)
        assert k in key.index, "tumor without cells: %s" % (k,)
        idx = key[k]
        out[idx] = sig[col].values.astype(np.float32)
        rows.append(dict(tumor=k[2], tissue=k[0], stage=k[1], cells=len(idx), **sig[col].to_dict()))
    return out, pd.DataFrame(rows)
