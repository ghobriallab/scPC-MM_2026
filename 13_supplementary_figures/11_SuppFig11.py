#!/usr/bin/env python3
"""Supplementary Figure 11: additional features of the malignancy and MST classifiers.

Purpose:      A, SenMayo senescence score of every PC on the UMAP of Fig. 1C, and its mean per individual, disease stage,
              tissue and PC status, by stage (BM normal | BM malignant | PB normal | PB malignant) and, for malignant PCs,
              by cytogenetic group (the most frequent call among the unit's cells); two-sided Wilcoxon rank-sum tests
              between stages within a block and malignant versus normal PCs per stage, Benjamini-Hochberg, each bracket
              with the largest q it summarizes. B, 1q and 13q on the malignant-cell UMAP of Fig. 2C, colored by the
              Numbat call of each cell (no call = neutral) and by the CNV-like score (mean log2 fold change of the arm's
              genes against control chromosomes 2, 4, 10, 12 and 20, quantile-transformed to a normal distribution), and
              the density of that score by call.

Inputs:       results/figure_inputs: pcs_cells.npz, pcs_senescence.npz, tumors_cells.npz, tumors_cnv.npz,
              patient_id_mean_probability.csv.gz.

Outputs:      figures/SuppFig11.svg, SuppFig11.pdf; figures/source_data/SuppFig11_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.stats import gaussian_kde

import palettes as P
import paths
import plots
import stage_tests as ST
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("SuppFig11", [
    Slot("senescence", "A", 0, 0, 183, 54),
    Slot("cnv_like_scores", "B", 0, 56, 183, 92),
], max_height=247)
COL = "mean_expr:SenMayo"
CNV = {"Amplification": "#D55E00", "Neutral": "#BDBDBD", "Deletion": "#0072B2"}


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig11_%s.tsv" % key, sep="\t", index=False)


def cat(z, k):
    return z[k + "__cats"].astype(str)[z[k + "__codes"]]


# ---- A ---------------------------------------------------------------------------------------------------------------
def senescence_units():
    z = np.load(paths.cache("pcs_cells.npz"))
    c = pd.DataFrame({k: cat(z, k) for k in ("patient_id", "sample_type", "annotation", "disease_status", "cytogenetics")})
    c[COL] = np.load(paths.cache("pcs_senescence.npz"))["senmayo"].astype(float)
    c["disease_status"] = ST.imwg_stages(c["patient_id"], c["sample_type"], c["disease_status"])     # IMWG risk groups
    keys = ["disease_status", "sample_type", "annotation", "patient_id"]
    u = c.groupby(keys, observed=True)[COL].agg(["mean", "size"]).reset_index().rename(columns={"mean": COL, "size": "n"})
    top = c[c["annotation"] == "tumor"].groupby(keys, observed=True)["cytogenetics"].agg(lambda s: s.value_counts().index[0])
    u = u.merge(top.rename("cyto").reset_index(), on=keys, how="left")
    u["cytogenetics"] = u["cyto"].map(ST.GENE)
    u["group"] = u["disease_status"] + ":" + u["sample_type"] + ":" + u["annotation"]
    u = u[u["group"].isin(ST.GROUPS)].drop(columns="cyto")
    return z["umap"], c[COL].values, u


def panel_senescence(sf):
    xy, v, u = senescence_units()
    ref = ST.score_units()                                     # units of the per-sample tables (risk groups from the clinical records)
    assert int(u["n"].sum()) == 1649290 and set(zip(u["group"], u["patient_id"])) == set(zip(ref["group"], ref["patient_id"]))
    ax = axes_mm(sf, 5.0, 7.0, 30.0, 30.0)
    norm = Normalize(float(np.min(v)), float(np.max(v)))
    plots.umap(ax, xy, plots.matplotlib_cm("viridis")(norm(v)), "sf11a", s=0.03, order=np.argsort(v, kind="stable"))
    ax.set_title("Senescence score (SenMayo)", fontsize=BASE, pad=2.0)
    cax = axes_mm(sf, 36.0, 14.0, 1.4, 17.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap="viridis"), cax, ticks=[0, 0.1, 0.2, 0.3])
    plots.plain_ticks(cb.ax.yaxis)
    t = ST.tests(u, COL)
    specs = ST.brackets(u, COL, t, 0.01, "score", score_rule=True)
    a2 = axes_mm(sf, 52.0, 8.0, 88.0, 30.0)
    lo, hi = u[COL].min(), u[COL].max()
    a2.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    plots.stage_box(a2, u, COL, specs, "sf11a-stage", ylabel="Mean senescence score", point=1.1)
    a2.yaxis.set_major_locator(plots.nice_locator(3))
    plots.plain_ticks(a2.yaxis)
    a2.tick_params(axis="y", labelsize=SMALL)
    cy = u[u["annotation"].eq("tumor") & u["cytogenetics"].notna()]
    assert len(cy) == 253
    a3 = axes_mm(sf, 151.0, 8.0, 31.0, 30.0)
    plots.box_strip(a3, cy.assign(value=cy[COL], group=cy["cytogenetics"]), P.CYTO_ORDER, P.CYTO, "sf11a-cyto", point=1.1,
                    rotate=90, label_size=SMALL, n_labels="row", width=0.62, jitter=0.18, edge=0.25)
    a3.set_ylim(*a2.get_ylim())
    a3.yaxis.set_major_locator(plots.nice_locator(3))
    plots.plain_ticks(a3.yaxis)
    a3.tick_params(axis="y", labelsize=SMALL)
    a3.set_ylabel("Mean senescence score", fontsize=SMALL, labelpad=1.0)
    for s in ("top", "right"):
        a3.spines[s].set_visible(False)
    rich_text(a3, 0.5, 1, [("Malignant PCs by cytogenetic group", "")], size=SMALL, ha="center", va="bottom",
              transform=plots.pts(a3, 0, 1.2, a3.transAxes), gid="pn-cytohdr")
    rows = []
    for key, df in t.items():
        rows += [dict(test=key, comparison=r["comparison"], p=r["p"], q=r["fdr"]) for _, r in df.iterrows()]
    source("A_units", u[["patient_id", "disease_status", "sample_type", "annotation", "cytogenetics", "n", COL]]
           .rename(columns={COL: "mean_senescence_score", "n": "cells"}))
    source("A_tests", pd.DataFrame(rows))
    source("A_brackets", pd.DataFrame(specs))


# ---- B ---------------------------------------------------------------------------------------------------------------
def density(ax, x, calls, key):
    """seaborn kdeplot(hue=call): Gaussian KDE per call (Scott bandwidth) scaled by the call's share of the cells; grid
    extended by 3 bandwidths (cut = 3)."""
    lo, hi = np.inf, -np.inf
    curves = {}
    for g in CNV:
        v = x[calls == g]
        if len(v) < 2:
            continue
        k = gaussian_kde(v)
        bw = float(np.sqrt(k.covariance[0, 0]))
        grid = np.linspace(v.min() - 3 * bw, v.max() + 3 * bw, 400)
        curves[g] = (grid, k(grid) * len(v) / len(x))
        lo, hi = min(lo, grid[0]), max(hi, grid[-1])
    for g, (grid, y) in curves.items():
        ax.plot(grid, y, color=CNV[g], lw=0.8, zorder=3)[0].set_gid("kde-%s-%s" % (key, g))
    ax.axvline(0, color="black", lw=0.5, ls=(0, (3, 2)), zorder=1).set_gid("ref-zero")
    return curves


def panel_cnv(sf):
    zc = np.load(paths.cache("tumors_cells.npz"))
    zn = np.load(paths.cache("tumors_cnv.npz"))
    xy = zc["umap"]
    assert len(xy) == len(zn["1q_qt"]) == 1232567
    counts = []
    for r, arm in enumerate(("1q", "13q")):
        y0 = 6.0 + 44.0 * r
        calls = cat(zn, arm + "_call")
        qt = zn[arm + "_qt"].astype(float)
        at_mm(sf, 4.0, y0 - 3.0, "%s arm" % arm, "pn-arm%d" % r, size=BASE)
        a1 = axes_mm(sf, 4.0, y0 + 2.0, 32.0, 32.0)
        col = np.array([CNV[c] for c in calls])
        plots.umap(a1, xy, col, "sf11b-call-" + arm, s=0.02)
        a1.set_title("Numbat CNV call", fontsize=BASE, pad=1.5)
        a2 = axes_mm(sf, 46.0, y0 + 2.0, 32.0, 32.0)
        norm = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
        plots.umap(a2, xy, plots.matplotlib_cm("coolwarm")(norm(np.clip(qt, -2, 2))), "sf11b-qt-" + arm, s=0.02,
                   order=np.argsort(qt, kind="stable"))
        a2.set_title("CNV-like score", fontsize=BASE, pad=1.5)
        cax = axes_mm(sf, 79.5, y0 + 10.0, 1.4, 17.0)
        cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap="coolwarm"), cax, ticks=[-2, -1, 0, 1, 2])
        plots.plain_ticks(cb.ax.yaxis)
        a3 = axes_mm(sf, 102.0, y0 + 2.0, 44.0, 28.0)
        curves = density(a3, qt, calls, arm)
        a3.set_xlim(-5.5, 5.5)
        a3.set_ylim(0, max(c[1].max() for c in curves.values()) * 1.08)
        a3.yaxis.set_major_locator(plots.nice_locator(4))
        plots.plain_ticks(a3.yaxis)
        plots.plain_ticks(a3.xaxis)
        a3.tick_params(labelsize=SMALL)
        a3.set_ylabel("Density")
        if r == 1:
            rich_text(a3, 0.5, 0, [("CNV-like score (quantile-normalized log", ""), ("2", "_"), (" fold change)", "")],
                      size=BASE, ha="center", va="top", transform=plots.pts(a3, 0, -9.0, a3.transAxes), gid="pn-xlab")
        a3.set_title("CNV-like score by Numbat call", fontsize=BASE, pad=1.5)
        for s in ("top", "right"):
            a3.spines[s].set_visible(False)
        n = pd.Series(calls).value_counts().reindex(list(CNV)).fillna(0).astype(int)
        counts.append(n.rename(arm))
        items = [([(g, ""), (" (", ""), ("n", "i"), (" = %s cells)" % format(int(n[g]), ","), "")], CNV[g]) for g in CNV
                 if n[g] > 0]                                    # no 1q deletion is called
        plots.key(sf, 17.0 / sf.w_mm, 1 - (y0 - 4.1) / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.4,
                  ncol=len(items), row_pt=2.2 * 72 / 25.4, gid="pn-key%d" % r)
    source("B_cells_per_call", pd.concat(counts, axis=1).reset_index().rename(columns={"index": "call"}))


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_senescence(pn["senescence"])
        panel_cnv(pn["cnv_like_scores"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig11", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
