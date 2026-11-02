#!/usr/bin/env python3
"""Supplementary Figures 3 and 4: summary of the sixteen transcriptional signatures (eight signatures per figure, panels A-H).

Purpose:      One row per signature: (1) UMAP of single-cell signature activity on the embedding of Fig. 1C (viridis,
              highest values drawn last, cells without a score in light gray); (2) the 15 genes with the highest Hotspot
              Z-score; (3) mean activity per patient by disease stage, tissue and PC status, with two-sided Wilcoxon
              rank-sum tests and Benjamini-Hochberg q (figlib/stage_tests.py); (4) mean activity of the malignant PCs of
              each BM or PB sample by cytogenetic group.

Inputs:       results/figure_inputs: pcs_cells.npz, pcs_signatures.npz, hotspot_genes.tsv,
              per-sample-mean-expr-score.csv.gz; results/stage_and_progression/stage_comparisons.tsv.

Outputs:      figures/SuppFig3.svg, SuppFig3.pdf, SuppFig4.svg, SuppFig4.pdf; figures/source_data/SuppFig3_4_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import os

import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import names
import palettes as P
import paths
import plots
import stage_tests as ST
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, SMALL, BASE, LW
from layout import FigureLayout, Slot, new_figure, axes_mm, save

ROW_H = 29.0
LETTERS = "ABCDEFGH"
PAGES = [names.SIGNATURE_ORDER[:8], names.SIGNATURE_ORDER[8:]]
LAYOUTS = [FigureLayout("SuppFig%d" % (k + 3), [Slot(sig, LETTERS[i], 0, i * ROW_H, 183, ROW_H)
                                                for i, sig in enumerate(page)], max_height=247)
           for k, page in enumerate(PAGES)]
INK = "#333333"
NAN_GRAY = "#D3D3D3"


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig3_4_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center"):
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs, size=size, ha=ha, va=va, transform=sf.transSubfigure, gid=gid)


def load_cells():
    c = np.load(paths.cache("pcs_cells.npz"))
    s = np.load(paths.cache("pcs_signatures.npz"), allow_pickle=True)
    return c["umap"], s["scores"], [str(x) for x in s["names"]]


def signature_row(sf, sig, xy, values, units, genes):
    key = "sf8-" + sig.replace("*", "").replace("/", "")
    at_mm(sf, 4.5, 1.6, [(names.SIGNATURES[sig], "")], "pn-title")
    # (1) UMAP of single-cell activity
    au = axes_mm(sf, 4.5, 4.5, 19.0, 19.0)
    ok = ~np.isnan(values)
    norm = Normalize(np.nanmin(values), np.nanmax(values))
    order = np.r_[np.where(~ok)[0], np.where(ok)[0][np.argsort(values[ok], kind="stable")]]
    col = np.empty((len(values), 4))
    col[~ok] = (0.827, 0.827, 0.827, 1.0)
    col[ok] = plots.matplotlib_cm("viridis")(norm(values[ok]))
    plots.umap(au, xy, col, key + "-umap", s=0.02, order=order, labels=False)
    au.set_xlabel("UMAP1", fontsize=SMALL, labelpad=0.8)
    au.set_ylabel("UMAP2", fontsize=SMALL, labelpad=0.8)
    cax = axes_mm(sf, 24.3, 7.5, 1.1, 13.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap="viridis"), cax)
    cb.locator = plots.nice_locator(2)
    cb.update_ticks()
    plots.plain_ticks(cb.ax.yaxis)
    rich_text(cax, 0.0, 1, [("Activity", "")], size=SMALL, ha="left", va="bottom",
              transform=plots.pts(cax, 0, 3.8, cax.transAxes), gid="pn-cbt")
    # (2) the 15 genes with the highest Hotspot Z-score
    top = genes[genes["context"] == sig].sort_values("Z", ascending=False).head(15)
    ab = axes_mm(sf, 39.0, 5.0, 31.0, 10.5)
    xb = np.arange(len(top))
    bars = ab.bar(xb, top["Z"].values, width=0.72, color=P.SIGNATURE[sig], edgecolor=INK, linewidth=0.3, zorder=2)
    for b in bars:
        b.set_gid("bar")
    ab.set_xlim(-0.6, len(top) - 0.4)
    ab.set_xticks(xb)
    ab.set_xticklabels(top["gene"], rotation=90, fontsize=SMALL)
    ab.tick_params(axis="x", length=0, pad=1.0)
    ab.tick_params(axis="y", labelsize=SMALL)
    ab.yaxis.set_major_locator(plots.nice_locator(2))
    ab.set_ylabel("Hotspot Z", fontsize=SMALL, labelpad=1.0)
    for sp in ("top", "right"):
        ab.spines[sp].set_visible(False)
    # (3) mean activity per patient by disease stage, tissue and PC status
    colname = "mean_expr:" + sig
    t = ST.tests(units, colname)
    ST.check_against_saved("signature " + sig, t)
    specs = ST.brackets(units, colname, t, 0.05, "signature")
    ax = axes_mm(sf, 82.0, 5.5, 60.0, 13.0)
    lo, hi = units[colname].min(), units[colname].max()
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    plots.stage_box(ax, units, colname, specs, key + "-stage", ylabel="Mean activity", point=1.0)
    ax.yaxis.set_major_locator(plots.nice_locator(2))
    plots.plain_ticks(ax.yaxis)
    ax.tick_params(axis="y", labelsize=SMALL)
    ax.yaxis.label.set_size(SMALL)
    # (4) mean activity of malignant PCs by cytogenetic group (BM and PB samples)
    cy = units[units["annotation"].eq("tumor") & units["cytogenetics"].notna()]
    cy = cy.assign(value=cy[colname].astype(float), group=cy["cytogenetics"])
    ac = axes_mm(sf, 152.0, 5.5, 30.5, 13.0)
    plots.box_strip(ac, cy, P.CYTO_ORDER, P.CYTO, key + "-cyto", point=1.0, rotate=90, label_size=SMALL, n_labels="row",
                    width=0.62, jitter=0.18, edge=0.25)
    clo, chi = cy["value"].min(), cy["value"].max()
    ac.set_ylim(clo - 0.06 * (chi - clo), chi + 0.08 * (chi - clo))          # points clear of the frame
    ac.yaxis.set_major_locator(plots.nice_locator(2))
    plots.plain_ticks(ac.yaxis)
    ac.tick_params(axis="y", labelsize=SMALL)
    ac.set_ylabel("Mean activity", fontsize=SMALL, labelpad=1.0)
    rich_text(ac, 0.5, 1, [("Cytogenetic group", "")], size=SMALL, ha="center", va="bottom",
              transform=plots.pts(ac, 0, 1.2, ac.transAxes), gid="pn-cytohdr")
    return top, specs, cy


def main():
    xy, scores, snames = load_cells()
    genes = pd.read_csv(paths.cache("hotspot_genes.tsv"), sep="\t")
    units = ST.signature_units()
    tops, brs = [], []
    with style():
        for lay, page in zip(LAYOUTS, PAGES):
            fig, pn = new_figure(lay)
            for sig in page:
                top, specs, cy = signature_row(pn[sig], sig, xy, scores[:, snames.index("mean_expr:" + sig)], units, genes)
                tops.append(top.assign(signature=names.SIGNATURES[sig])[["signature", "gene", "Z"]])
                brs += [dict(signature=names.SIGNATURES[sig], block=b["block"], kind=b["kind"], q=b["q"],
                             comparisons=b["n"]) for b in specs]
            issues = figure_checks(fig, lay)
            svg, pdf = save(fig, lay)
            fi, ntext = file_checks(svg, pdf, lay)
            print(report(lay.name, issues + fi, "SVG text elements: %d" % ntext))
            print("\n".join(issues + fi) or "ALL CHECKS PASS")
    cols = ["mean_expr:" + s for s in names.SIGNATURE_ORDER]
    source("activity_per_patient", units[["patient_id", "disease_status", "sample_type", "annotation", "cytogenetics"] + cols]
           .rename(columns={c: names.SIGNATURES[c.split(":")[1]] for c in cols}))
    source("top_genes", pd.concat(tops))
    source("brackets", pd.DataFrame(brs))


if __name__ == "__main__":
    main()
