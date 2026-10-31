#!/usr/bin/env python3
"""Supplementary Figure 6: genes stably up- or downregulated in malignant PCs across disease stages.

Purpose:      Heatmap: per BM tumor, the significant log2 fold change (malignant versus normal PCs; figlib/targets.py)
              of the six genes up and the six genes down in at least half of the tumors of every stage; columns as in
              Fig. 2A; rows by Ward clustering (asserted). UMAPs: per malignant PC, the significant log2 fold change of
              its tumor on the UMAP of Fig. 2C.

Inputs:       data/paper (degs/scrna per-tumor tables); results/figure_inputs/tumors_cells.npz; drawing helpers of
              12_main_figures/02_Fig2.py.

Outputs:      figures/SuppFig6.svg, SuppFig6.pdf; figures/source_data/SuppFig6_heatmap.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys
import importlib

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.patches import FancyArrowPatch

import palettes as P
import paths
import plots

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "12_main_figures"))
fig2 = importlib.import_module("02_Fig2")        # Fig. 2A matrix, tracks and tumor-cell UMAP
import targets as T
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("SuppFig6", [Slot("stable_genes", "", 0, 0, 183, 128)], max_height=247)
PT_MM = 25.4 / 72.0
UP = ["IGF1", "MLLT3", "QPCT", "TJP1", "CDKN1A", "BMI1"]
DOWN = ["CD27", "HLA-DPA1", "CD81", "CD79A", "TMSB4X", "CD99"]
ROW_ORDER = ["TJP1", "BMI1", "QPCT", "MLLT3", "IGF1", "CDKN1A", "HLA-DPA1", "CD81", "CD99", "TMSB4X", "CD27", "CD79A"]


def heatmap(sf):
    M_t, meta = fig2.tumor_matrix()                      # Fig. 2A columns (groups, Ward within each group)
    fdr, lfc = T.de_tables()
    Y = T.significant_lfc(fdr, lfc, UP + DOWN)[list(meta.index)]
    rows = list(Y.index[sch.leaves_list(sch.linkage(Y.values, method="ward", metric="euclidean"))])
    assert rows == ROW_ORDER, rows
    Y = Y.loc[rows]
    x0, x1 = 16.0, 172.0
    y0, y1 = 18.5, 56.0
    ah = axes_mm(sf, x0, y0, x1 - x0, y1 - y0)
    fig2.mesh(ah, Y.values, "sf6")
    ah.set_yticks(np.arange(len(Y)) + 0.5)
    ah.set_yticklabels(Y.index, fontsize=SMALL)
    ah.set_xticks([])
    ah.tick_params(axis="y", length=0, pad=1.0)
    for sp in ah.spines.values():
        sp.set_visible(False)
    k = rows.index("TMSB4X")                              # arrow at the TMSB4X row
    tr = plots.blended_transform_factory(ah.transAxes, ah.transData)
    a = FancyArrowPatch((0, k + 0.5), (0, k + 0.5), transform=tr, arrowstyle="-|>,head_length=2.2,head_width=1.1",
                        mutation_scale=1.0, lw=0.6, color="#333333", shrinkA=0, shrinkB=0, clip_on=False)
    a.set_positions((-0.093, k + 0.5), (-0.070, k + 0.5))
    a.set_gid("arrow-tmsb4x")
    ah.add_patch(a)
    tracks = [("Disease stage", [P.TUMOR[s] for s in meta["stage"]]),
              ("HRD", [P.CYTO["HRD"] if h else "#FFFFFF" for h in meta["hrd"]]),
              ("Translocation", [P.CYTO[t] if t else "#FFFFFF" for t in meta["translocation"]])]
    for i, (lab, colors) in enumerate(tracks):
        y = 9.4 + i * 2.75
        at_ = axes_mm(sf, x0, y, x1 - x0, 2.3)
        fig2.track(at_, colors, "s%d" % i)
        fig2.at_mm(sf, x0 - 1.0, y + 1.15, lab, "pn-tr%d" % i, size=SMALL, ha="right")
    cax = axes_mm(sf, x1 + 1.8, y0 + 8.0, 1.3, y1 - y0 - 16.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=fig2.NORM, cmap="coolwarm"), cax, ticks=[-3, 0, 3])
    plots.plain_ticks(cb.ax.yaxis)
    rich_text(cax, 0.0, 1, [("Log", ""), ("2", "_"), (" FC", "")], size=SMALL, ha="left", va="bottom",
              transform=plots.pts(cax, 0, 3.8, cax.transAxes), gid="pn-cblab")
    for r, (title, items) in enumerate((("Disease stage", [(s, P.TUMOR[s]) for s in fig2.STAGES5]),
                                         ("Cytogenetic group", [(g, P.CYTO[g]) for g in fig2.TR + ["HRD"]]))):
        y = 2.0 + r * 3.3
        fig2.at_mm(sf, x0, y, title, "pn-kt%d" % r, size=SMALL)
        kx = x0 + rich_width_pt(sf, [(title, "")], SMALL) * PT_MM + 2.5
        plots.key(sf, kx / sf.w_mm, 1 - (y - 1.0) / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.3,
                  marker="s", ncol=len(items), row_pt=2.0 / PT_MM, gid="pn-k%d" % r)
    out = Y.T.copy()
    out.insert(0, "stage", meta["stage"].values)
    out.insert(0, "tumor", meta["tumor"].values)
    out.reset_index(drop=True).to_csv(paths.SOURCE_DATA + "/SuppFig6_heatmap.tsv", sep="\t", index=False)
    return rows


def umaps(sf, genes):
    xy, c = fig2.tumor_cells()
    vals, _ = T.per_cell(c["sample_type"], c["disease_status"], c["tumor_id"], genes)
    for j, g in enumerate(genes):
        r, k = divmod(j, 6)
        x, y = 1.0 + k * 29.0, 62.0 + r * 32.5
        ax = axes_mm(sf, x, y + 2.6, 26.0, 26.0)
        v = vals[:, j]
        ok = ~np.isnan(v)
        order = np.r_[np.where(~ok)[0], np.where(ok)[0][np.argsort(v[ok], kind="stable")]]
        col = np.ones((len(v), 4))
        col[ok] = plots.matplotlib_cm("coolwarm")(fig2.NORM(v[ok]))
        plots.umap(ax, xy, col, "sf6-" + g, s=0.012, order=order, labels=False)
        rich_text(ax, 0.5, 1, [(g, "")], size=BASE, ha="center", va="bottom", transform=plots.pts(ax, 0, 1.0, ax.transAxes),
                  gid="pn-u%d" % j)
    fig2.at_mm(sf, 1.0, 62.0 + 32.5 + 2.6 + 26.0 + 2.2, [("UMAPs: each malignant PC colored by the log", ""), ("2", "_"),
                                                (" FC of its tumor (scale as in the heatmap); white, not tested", "")],
               "pn-unote", size=SMALL)


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        sf = pn["stable_genes"]
        rows = heatmap(sf)
        umaps(sf, rows)
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig6", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
