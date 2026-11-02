#!/usr/bin/env python3
"""Extended Data Figure 2: malignant PCs without detectable CNV.

Purpose:      A, percentage of the 182 tumors without any CNV called by Numbat, by cytogenetic group; Fisher's exact
              test, CCND1 versus other translocations. B, percentage of malignant PCs without a CNV call in each of the
              152 tumors with CNVs; Kruskal-Wallis test across groups, Wilcoxon rank-sum test CCND1 versus other
              translocations. C, all PCs on the UMAP of Fig. 1C by CNV status (every cell drawn, the groups of interest
              last). Every P value is read from the saved tests.

Inputs:       results/genomics: cnv_negative_tumors.tsv, cnv_negative_cells_per_tumor.tsv, cnv_negative_tests.tsv,
              cnv_status_umap.csv.gz.

Outputs:      figures/EDFig2.svg, EDFig2.pdf; figures/source_data/EDFig2_*.tsv.

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
from figstyle import style, rich_text, rich_width_pt, p_runs, join_runs, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("EDFig2", [
    Slot("tumors_without_cnv", "A", 0, 0, 44, 66),
    Slot("cells_without_cnv", "B", 46, 0, 50, 66),
    Slot("umap_cnv_status", "C", 98, 0, 85, 66),
], max_height=247)
PT_MM = 25.4 / 72.0
GROUPS = ["t(11;14)", "Other IgH translocation", "HRD", "No translocation or HRD"]       # table keys
LABELS = {"t(11;14)": "CCND1", "Other IgH translocation": "Other translocation", "HRD": "HRD",
          "No translocation or HRD": "No translocation\nor HRD detected"}
COLORS = {"t(11;14)": P.CYTO["CCND1"], "Other IgH translocation": "#88CCEE", "HRD": P.CYTO["HRD"],
          "No translocation or HRD": P.CYTO["Other"]}
UMAP = [("Normal PC", "Normal PCs", "#EBEBEB", 0.02), ("Malignant, not analyzed", "Malignant, not analyzed by Numbat",
                                                         "#8C8C8C", 0.02),
        ("Malignant, CNV detected", "Malignant, CNV detected", "#56B4E9", 0.02),
        ("Malignant, no CNV detected (tumor with CNVs)", "Malignant, no CNV call in a tumor with CNVs", "#D55E00",
         0.05),
        ("Malignant, tumor without CNV", "Malignant, tumor without CNV", "#CC79A7", 0.05)]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/EDFig2_%s.tsv" % key, sep="\t", index=False)


def pval(tests, test, text):
    hit = tests[(tests["test"] == test) & tests["detail"].str.contains(text, regex=False)]
    assert len(hit) == 1, text
    return float(hit["P"].iloc[0])


def labels_below(ax, ns):
    """Rotated group labels under a row of counts (n)."""
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([LABELS[g] for g in GROUPS], rotation=90, fontsize=SMALL, ha="center", va="top")
    plots.n_row(ax, ns)


def panel_tumors(sf, tum, tests):
    k = tum.groupby("group")["cnv_negative"].sum().reindex(GROUPS)
    n = tum.groupby("group").size().reindex(GROUPS)
    assert int(n.sum()) == 182
    pct = 100 * k / n
    ax = axes_mm(sf, 11.0, 6.0, 30.0, 30.0)
    bars = ax.bar(range(len(GROUPS)), pct.values, color=[COLORS[g] for g in GROUPS], edgecolor="#333333", lw=0.5,
                  width=0.65, zorder=2)
    for b in bars:
        b.set_gid("bar")
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.set_ylim(0, 50)
    for i, g in enumerate(GROUPS):
        rich_text(ax, i, pct[g], [("%d/%d" % (k[g], n[g]), "")], size=SMALL, ha="center", va="bottom",
                  transform=plots.pts(ax, 0, 1.0), gid="ax-frac%d" % i)
    p = pval(tests, "Fisher two-sided", "t(11;14) %d/%d vs other IgH" % (k["t(11;14)"], n["t(11;14)"]))
    tops = {i: float(pct[g]) + 2.2 * 50 / (30 / PT_MM) * 3 for i, g in enumerate(GROUPS)}
    plots.brackets(ax, [dict(i=0, j=1, p=p, color="black")], tops, ymin=0, key="sf2a")      # groups, not stages
    ax.set_ylabel("Tumors without\ndetectable CNV (%)")
    labels_below(ax, list(n.values))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    source("A", pd.DataFrame({"group": [LABELS[g].replace("\n", " ") for g in GROUPS], "tumors_without_cnv": k.values,
                              "tumors": n.values}))


def panel_cells(sf, per, tests):
    d = per.dropna(subset=["pct_cnv_negative_cells"]).copy()
    assert len(d) == 152
    d["value"] = d["pct_cnv_negative_cells"]
    ax = axes_mm(sf, 11.0, 6.0, 36.0, 30.0)
    ax.set_ylim(-3, 103)
    tops = plots.box_strip(ax, d, GROUPS, COLORS, "sf2b", point=1.4, width=0.6, n_labels=None, edge=0.25)
    pw = pval(tests, "Wilcoxon rank-sum two-sided", "CNV-negative cells: t(11;14) median")
    plots.brackets(ax, [dict(i=0, j=1, p=pw, color="black")], tops, ymin=-3, key="sf2b")
    pk = pval(tests, "Kruskal-Wallis", "4 cytogenetic groups")
    rich_text(ax, 0.5, 1.0, [("Kruskal-Wallis ", "")] + p_runs(pk), size=SMALL, ha="center", va="bottom",
              transform=plots.pts(ax, 0, 1.5, ax.transAxes), gid="pn-kw")
    ax.set_ylabel("Malignant PCs without\ndetectable CNV per tumor (%)")
    labels_below(ax, [int((d["group"] == g).sum()) for g in GROUPS])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    source("B", d[["patient_id", "group", "pct_cnv_negative_cells"]])


def panel_umap(sf):
    u = pd.read_csv(paths.result("genomics", "cnv_status_umap.csv.gz"))
    n_all = u["category"].value_counts()
    assert int(n_all.sum()) == 1649290
    ax = axes_mm(sf, 6.0, 4.0, 46.0, 46.0)
    for cat, lab, col, s in UMAP:                          # highlighted groups drawn last
        m = (u["category"] == cat).values
        plots.umap(ax, u.loc[m, ["umap1", "umap2"]].values, col, "sf2c-" + str(UMAP.index((cat, lab, col, s))), s=s)
    ax.set_title("All PCs by CNV status of malignant PCs", fontsize=BASE, pad=2.0)
    items = [("%s (%s)" % (lab, format(int(n_all[cat]), ",")), col) for cat, lab, col, s in UMAP]
    plots.key(sf, 6.0 / sf.w_mm, 1 - 54.5 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.4,
              row_pt=2.1 / PT_MM, gid="pn-kumap")


def main():
    res = lambda n: paths.result("genomics", n)
    tum = pd.read_csv(res("cnv_negative_tumors.tsv"), sep="\t")
    per = pd.read_csv(res("cnv_negative_cells_per_tumor.tsv"), sep="\t")
    tests = pd.read_csv(res("cnv_negative_tests.tsv"), sep="\t")
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_tumors(pn["tumors_without_cnv"], tum, tests)
        panel_cells(pn["cells_without_cnv"], per, tests)
        panel_umap(pn["umap_cnv_status"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("EDFig2", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
