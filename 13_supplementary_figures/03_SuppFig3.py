#!/usr/bin/env python3
"""Supplementary Figure 3: processing batches and the PC UMAP, and paired BM and PB samples.

Purpose:      A, origin of the 30 nearest neighbors of each PC in the graph of the Fig. 1C UMAP, mean over individuals.
              B, per-patient UMAPs of three same-visit BM and PB pairs (at most 800 cells per tissue and PC status
              drawn, seeded).

Inputs:       results/preprocessing: batch_neighbor_composition.tsv, batch_patient_umaps.tsv.gz.

Outputs:      figures/SuppFig3.svg, SuppFig3.pdf; figures/source_data/SuppFig3_A.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import numpy as np
import pandas as pd

import palettes as P
import paths
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rng, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("SuppFig3", [Slot("neighbors", "A", 0, 0, 44, 60), Slot("paired_umaps", "B", 46, 0, 137, 60)],
                      max_height=247)
PT_MM = 25.4 / 72.0
COMP = [("same_sample", "Same sample", "#4D4D4D"), ("same_individual_other_sample", "Same individual, other sample", "#A6A6A6"),
        ("other_individual_same_day", "Other individuals, same processing day", "#E69F00"),
        ("other_individual_other_day", "Other individuals, other days", "#B5DDF5")]
GROUPS = [("BM", "malignant"), ("PB", "malignant"), ("BM", "normal"), ("PB", "normal")]


def panel_neighbors(sf, C):
    order = ["normal", "malignant"]
    m = C.groupby("cell_type")[[c[0] for c in COMP]].mean().loc[order]
    ax = axes_mm(sf, 12.0, 4.0, 24.0, 30.0)
    bottom = np.zeros(len(order))
    for k, lab, col in COMP:
        bars = ax.bar(range(len(order)), m[k].values, bottom=bottom, color=col, width=0.65, edgecolor="#333333",
                      linewidth=0.4, zorder=2)
        for b in bars:
            b.set_gid("bar")
        bottom += m[k].values
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Normal\nPCs", "Malignant\nPCs"], fontsize=SMALL)
    plots.n_below(ax, [int((C["cell_type"] == o).sum()) for o in order])
    ax.set_ylabel("Nearest neighbors (%)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plots.key(sf, 1.0 / sf.w_mm, 1 - 44.0 / sf.h_mm, [(lab, col) for _, lab, col in COMP][::-1], transform=sf.transSubfigure,
              size=SMALL, ms=2.3, marker="s", row_pt=2.1 / PT_MM, gid="pn-kA")
    out = m.reset_index().rename(columns={"cell_type": "PC status"})
    out.insert(1, "individuals", [int((C["cell_type"] == o).sum()) for o in order])
    out.to_csv(paths.SOURCE_DATA + "/SuppFig3_A.tsv", sep="\t", index=False)


def panel_umaps(sf, U):
    r = rng("sf3b")
    for i, (p, q) in enumerate(U.groupby("patient", sort=False)):
        ax = axes_mm(sf, 3.0 + i * 45.0, 4.0, 40.0, 40.0)
        for tissue, ct in GROUPS:
            d = q[(q["tissue"] == tissue) & (q["cell_type"] == ct)]
            d = d.iloc[np.sort(r.choice(len(d), min(len(d), 800), replace=False))] if len(d) else d
            sc = ax.scatter(d["umap1"], d["umap2"], s=0.6, c=P.PC_STATUS["%s %s" % (tissue, ct)], linewidths=0,
                            rasterized=True, zorder=1)
            sc.set_gid("raster-sf3b-%d-%s%s" % (i, tissue, ct))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("%s, %s" % (p, q["stage"].iloc[0]), fontsize=BASE, pad=2.0)
        ax.set_xlabel("UMAP1", labelpad=1.0)
        if i == 0:
            ax.set_ylabel("UMAP2", labelpad=1.0)
    items = [("%s %s PCs" % (t, c), P.PC_STATUS["%s %s" % (t, c)]) for t, c in GROUPS]
    plots.key(sf, 3.0 / sf.w_mm, 1 - 50.5 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.4, ncol=4,
              col_pt=24.0 / PT_MM, row_pt=2.1 / PT_MM, gid="pn-kB")


def main():
    C = pd.read_csv(paths.result("preprocessing", "batch_neighbor_composition.tsv"), sep="\t")
    U = pd.read_csv(paths.result("preprocessing", "batch_patient_umaps.tsv.gz"), sep="\t")
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_neighbors(pn["neighbors"], C)
        panel_umaps(pn["paired_umaps"], U)
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig3", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
