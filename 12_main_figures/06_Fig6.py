#!/usr/bin/env python3
"""Figure 6: MGUS-like and MM-like transcriptional states within precursor tumors and across their CNV clones.

Purpose:      Out-of-sample MST scores of the malignant BM PCs of every tumor (classifier patients scored by
              cross-validation or as test-set patients, all others by the final model; 09_stage_and_progression/
              11_mm_like_cells_out_of_sample.py). A, per tumor, the 10th to 90th percentile and the median of the MST
              scores of its cells, by disease stage. B, percentage of MM-like cells (MST score above 0.5) per tumor by
              disease stage, two-sided Spearman correlation with stage and two-sided Wilcoxon rank-sum tests between
              consecutive stages (Benjamini-Hochberg q). C, percentage of MM-like cells in each Numbat CNV clone against
              the whole tumor, in tumors with at least two clones of at least 50 cells.

Inputs:       results/stage_and_progression: mm_like_oos_tumors.tsv, mm_like_oos_clones.tsv, mm_like_oos_tests.tsv.

Outputs:      figures/Fig6.svg, Fig6.pdf; figures/source_data/Fig6_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import numpy as np
import pandas as pd
from scipy import stats

import palettes as P
import paths
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, p_runs, rho_runs, join_runs, n_runs, SMALL, BASE, LW
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("Fig6", [
    Slot("tumor_distributions", "A", 0, 0, 183, 56),
    Slot("mm_like_by_stage", "B", 0, 58, 64, 60),
    Slot("cnv_clones", "C", 66, 58, 117, 60),
], max_height=170)
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/Fig6_%s.tsv" % key, sep="\t", index=False)


def close(a, b, rel=1e-6):
    assert abs(a - b) <= rel * max(1e-300, abs(b)), (a, b)


def test(T, name, detail):
    row = T[(T["test"] == name) & (T["detail"] == detail)]
    assert len(row) == 1, (name, detail)
    return row.iloc[0]


# ---- A ---------------------------------------------------------------------------------------------------------------
def panel_distributions(sf, T):
    at_mm(sf, 4.5, 1.8, "MST score of the malignant BM PCs of each tumor (out of sample)", "pn-title")
    ax = axes_mm(sf, 12.0, 7.0, 168.0, 34.0)
    T = T.sort_values(["stage", "median_mst"]).reset_index(drop=True)
    for i, r in T.iterrows():
        ax.plot([i, i], [r["q10_mst"], r["q90_mst"]], color=P.TUMOR[r["disease_status"]], lw=0.7, solid_capstyle="butt",
                zorder=2)[0].set_gid("bar-%d" % i)
    ax.scatter(T.index, T["median_mst"], s=1.6, c="black", lw=0, zorder=3)
    plots.refline(ax, 0.5)
    ax.set_xlim(-1.5, len(T) + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    plots.plain_ticks(ax.yaxis)
    ax.tick_params(axis="y", labelsize=SMALL)
    ax.set_ylabel("MST score")
    ax.set_xticks([])
    for s_ in STAGES:
        idx = T.index[T["disease_status"] == s_]
        rich_text(ax, (idx.min() + idx.max()) / 2.0, 0, [(s_, "")], size=SMALL, ha="center", va="top",
                  transform=plots.pts(ax, 0, -2.0, ax.get_xaxis_transform()), gid="pn-st" + s_)
        rich_text(ax, (idx.min() + idx.max()) / 2.0, 0, n_runs(len(idx)), size=SMALL, ha="center", va="top",
                  transform=plots.pts(ax, 0, -8.5, ax.get_xaxis_transform()), gid="pn-n" + s_)
        if s_ != "MM":
            ax.axvline(idx.max() + 0.5, color="#BDBDBD", lw=0.4, zorder=1).set_gid("bg-sep" + s_)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    at_mm(sf, 12.0, 52.5, "Bars, 10th–90th percentile of the cells of a tumor; dots, median; tumors ordered by median",
          "pn-note", size=SMALL)
    source("A", T[["patient_id", "disease_status", "source", "cells", "q10_mst", "median_mst", "q90_mst", "frac_mm_like"]])


# ---- B ---------------------------------------------------------------------------------------------------------------
def panel_by_stage(sf, T, tests):
    at_mm(sf, 4.5, 1.8, "MM-like cells per tumor", "pn-title")
    sp = test(tests, "Spearman", "fraction of MM-like cells vs disease stage (0-4)")
    r2, p2 = stats.spearmanr(T["stage"], T["frac_mm_like"])
    close(r2, sp["statistic"])
    close(p2, sp["P"])
    d = T.assign(group=T["disease_status"], value=100 * T["frac_mm_like"])
    ax = axes_mm(sf, 12.0, 8.5, 42.0, 34.0)
    tops = plots.box_strip(ax, d, STAGES, P.TUMOR, "f6b", point=1.1, width=0.62, jitter=0.18, edge=0.2, rotate=90,
                           label_size=SMALL, n_labels="row")
    comps = []
    for k, (x, y) in enumerate(zip(STAGES[:-1], STAGES[1:])):
        t = test(tests, "Wilcoxon rank-sum", "fraction of MM-like cells, %s vs %s" % (x, y))
        p = stats.mannwhitneyu(T.loc[T["disease_status"] == x, "frac_mm_like"], T.loc[T["disease_status"] == y, "frac_mm_like"],
                               alternative="two-sided").pvalue
        close(p, t["P"])
        comps.append(dict(i=k, j=k + 1, p=t["q"], sym="q", color=P.BRACKET_STAGE))
    ax.set_ylim(-3, 100)
    plots.brackets(ax, comps, tops, ymin=-3, key="f6b")
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="y", labelsize=SMALL)
    ax.set_ylabel("% MM-like cells")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    at_mm(sf, 12.0, 6.0, [("Spearman ", "")] + join_runs(rho_runs(sp["statistic"]), p_runs(sp["P"])), "pn-rho", size=SMALL)
    source("B", d[["patient_id", "disease_status", "value"]].rename(columns={"value": "pct_mm_like"}))


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_clones(sf, C, tests):
    tum = C.groupby(["patient_id", "disease_status"]).first().reset_index()
    n_t, n_sig = len(tum), int((tum["perm_q"] < 0.05).sum())
    t = test(tests, "Permutation (per tumor, BH)", "clone difference larger than chance, tumors with q < 0.05")
    assert int(t["statistic"]) == n_sig and int(t["n"]) == n_t
    at_mm(sf, 4.5, 1.8, [("Numbat CNV clones of %d tumors with ≥ 2 clones" % n_t, "")], "pn-title")
    ax = axes_mm(sf, 14.0, 8.0, 40.0, 40.0)
    for (pt, st), g in C.groupby(["patient_id", "disease_status"], sort=False):
        col = P.TUMOR[st]
        x = np.full(len(g), 100 * g["tumor_frac_mm_like"].iloc[0])
        ax.plot(x, 100 * g["frac_mm_like"], color=col, lw=0.4, zorder=2)[0].set_gid("cl-%s-%s" % (pt, st))
        ax.scatter(x, 100 * g["frac_mm_like"], s=3.5, c=col, lw=0, zorder=3)
    ax.plot([0, 100], [0, 100], color=P.REFLINE, lw=0.5, ls=(0, (3, 2)), zorder=1)[0].set_gid("ref-diag")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([0, 50, 100])
    ax.tick_params(labelsize=SMALL)
    ax.set_xlabel("% MM-like cells, whole tumor")
    ax.set_ylabel("% MM-like cells, CNV clone")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    at_mm(sf, 66.0, 10.5, "Disease stage", "pn-keyt", size=SMALL)
    plots.key(sf, 66.0 / sf.w_mm, 1 - 12.5 / sf.h_mm, [(s_, P.TUMOR[s_]) for s_ in STAGES], transform=sf.transSubfigure,
              size=SMALL, ms=2.4, gid="pn-key")
    at_mm(sf, 14.0, 57.5, "Points, clones; lines join the clones of a tumor; dashed line, clone = tumor", "pn-note",
          size=SMALL)
    source("C", C[["patient_id", "disease_status", "clone", "cells", "frac_mm_like", "tumor_frac_mm_like", "clone_range",
                   "null_median_range", "perm_p", "perm_q"]])


def main():
    T = pd.read_csv(paths.result("stage_and_progression", "mm_like_oos_tumors.tsv"), sep="\t")
    C = pd.read_csv(paths.result("stage_and_progression", "mm_like_oos_clones.tsv"), sep="\t", dtype={"clone": str})
    tests = pd.read_csv(paths.result("stage_and_progression", "mm_like_oos_tests.tsv"), sep="\t")
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_distributions(pn["tumor_distributions"], T)
        panel_by_stage(pn["mm_like_by_stage"], T, tests)
        panel_clones(pn["cnv_clones"], C, tests)
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("Fig6", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
