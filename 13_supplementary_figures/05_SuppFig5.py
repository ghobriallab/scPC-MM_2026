#!/usr/bin/env python3
"""Supplementary Figure 5: sixteen therapeutic targets on the UMAP of Fig. 1C.

Purpose:      A, log-normalized expression of each target in every PC (viridis, highest values drawn last). B, per cell,
              the log2 fold change of its tumor (malignant versus normal PCs of the same BM sample) when the target is
              differentially expressed in that tumor (q <= 0.01 and |log2 fold change| >= 1), 0 otherwise; white for the
              normal PCs, PB samples and minor tumors that were not tested (coolwarm, -3 to 3). Immunoglobulin genes and
              minor tumors are removed.

Inputs:       results/figure_inputs: pcs_targets.npz, pcs_cells.npz; results/differential_expression:
              BM_scRNA_tumor_DE_FDR.csv, BM_scRNA_tumor_DE_logFoldChange.csv (IMWG 2/20/20 risk groups, risk.py).

Outputs:      figures/SuppFig5.svg, SuppFig5.pdf; figures/source_data/SuppFig5_B_tumor_log2fc.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

import paths
import stage_tests as ST
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

TARGETS = ["TNFRSF17", "GPRC5D", "SDC1", "CD38", "SLAMF7", "FCRL5", "CD46", "CD47", "CD48", "NCAM1", "ITGB7", "CD74", "LY9",
           "TNFRSF13B", "CD19", "BCL2"]
LAYOUT = FigureLayout("SuppFig5", [Slot("expression", "A", 0, 0, 183, 84), Slot("log2fc", "B", 0, 86, 183, 84)],
                      max_height=247)
COL_W, ROW_H, UM = 30.5, 26.8, 21.5


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig5_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center"):
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs, size=size, ha=ha, va=va, transform=sf.transSubfigure, gid=gid)


def cells():
    z = np.load(paths.cache("pcs_cells.npz"))
    cat = lambda k: z[k + "__cats"].astype(str)[z[k + "__codes"]]
    d = pd.DataFrame({"tissue": cat("sample_type"), "stage": cat("disease_status"),
                      "annotation": cat("annotation"), "tumor_id": cat("tumor_id")})
    d["stage"] = ST.imwg_stages(cat("patient_id"), d["tissue"], d["stage"])          # IMWG 2/20/20 risk groups
    return z["umap"], d


def tumor_log2fc(meta):
    """Per-cell value of panel B for each target."""
    rd = lambda n: pd.read_csv(paths.result("differential_expression", n), index_col=0)
    fdr, lfc = rd("BM_scRNA_tumor_DE_FDR.csv"), rd("BM_scRNA_tumor_DE_logFoldChange.csv")
    keep = ~fdr.columns.str.endswith("tumor_1")                         # minor tumors were not tested
    fdr, lfc = fdr.loc[TARGETS, keep], lfc.loc[TARGETS, keep]
    key = (meta["tissue"] + "|" + meta["stage"] + "|" + meta["tumor_id"]).values
    pos = pd.Series(np.arange(len(key))).groupby(key).apply(lambda s: s.values)
    out = np.full((len(key), len(TARGETS)), np.nan, dtype=np.float32)
    rows = []
    for col in fdr.columns:
        info = col.split("_")
        k = "%s|%s|%s" % (info[1], info[2], ":".join([info[0], info[-2], info[-1]]))
        idx = pos.get(k)
        assert idx is not None, "tumor without cells: %s" % k
        for j, g in enumerate(TARGETS):
            de = fdr.loc[g, col] <= 0.01 and abs(lfc.loc[g, col]) >= 1
            out[idx, j] = lfc.loc[g, col] if de else 0.0
        rows.append(dict(tumor=":".join([info[0], info[-2], info[-1]]), tissue=info[1], stage=info[2], cells=len(idx),
                         **{g: (lfc.loc[g, col] if (fdr.loc[g, col] <= 0.01 and abs(lfc.loc[g, col]) >= 1) else 0.0)
                            for g in TARGETS}))
    return out, pd.DataFrame(rows)


def grid(sf, xy, values, cmap, norm, key, colorbars):
    for j, g in enumerate(TARGETS):
        r, c = divmod(j, 6)
        x0, y0 = 1.5 + c * COL_W, 5.5 + r * ROW_H
        ax = axes_mm(sf, x0, y0 + 2.6, UM, UM)
        v = values[:, j]
        ok = ~np.isnan(v)
        order = np.r_[np.where(~ok)[0], np.where(ok)[0][np.argsort(v[ok], kind="stable")]]
        col = np.ones((len(v), 4))                                         # no value: white
        col[ok] = plots.matplotlib_cm(cmap)(norm(v[ok]) if norm is not None else Normalize(v.min(), v.max())(v[ok]))
        plots.umap(ax, xy, col, "%s-%s" % (key, g), s=0.012, order=order, labels=False)
        rich_text(ax, 0.5, 1, [(g, "")], size=BASE, ha="center", va="bottom", transform=plots.pts(ax, 0, 1.0, ax.transAxes),
                  gid="pn-%s-title%d" % (key, j))
        if colorbars:
            import matplotlib.cm as mcm
            cax = axes_mm(sf, x0 + UM + 0.6, y0 + 2.6 + 4.0, 1.0, UM - 8.0)
            cb = plots.colorbar(sf, mcm.ScalarMappable(norm=Normalize(float(v.min()), float(v.max())), cmap=cmap), cax,
                                ticks=plots.range_ticks(float(v.min()), float(v.max())))
            cb.update_ticks()
            plots.plain_ticks(cb.ax.yaxis)


def main():
    import matplotlib.cm as mcm
    xy, meta = cells()
    z = np.load(paths.cache("pcs_targets.npz"), allow_pickle=True)
    assert [str(g) for g in z["genes"]] == TARGETS
    lfc_cells, tumors = tumor_log2fc(meta)
    with style():
        fig, pn = new_figure(LAYOUT)
        sa, sb = pn["expression"], pn["log2fc"]
        at_mm(sa, 4.5, 1.7, [("Log-normalized expression", "")], "pn-a-title")
        grid(sa, xy, z["expr"], "viridis", None, "sf5a", colorbars=True)
        at_mm(sb, 4.5, 1.7, [("Log", ""), ("2", "_"), (" fold change in each BM tumor (malignant vs normal PCs) where the "
                                                     "target is differentially expressed", "")], "pn-b-title")
        norm = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
        grid(sb, xy, lfc_cells, "coolwarm", norm, "sf5b", colorbars=False)
        cax = axes_mm(sb, 4 * COL_W + 8.0, 5.5 + 2 * ROW_H + 12.0, 40.0, 1.3)
        cb = plots.colorbar(sb, mcm.ScalarMappable(norm=norm, cmap="coolwarm"), cax, orientation="horizontal",
                            ticks=[-3, -1.5, 0, 1.5, 3])
        plots.plain_ticks(cb.ax.xaxis)
        rich_text(cax, 0.5, 0, [("Log", ""), ("2", "_"), (" fold change (0, not differentially expressed)", "")],
                  size=SMALL, ha="center", va="top", transform=plots.pts(cax, 0, -8.0, cax.transAxes), gid="pn-b-cblab")
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig5", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")
    source("B_tumor_log2fc", tumors)


if __name__ == "__main__":
    main()
