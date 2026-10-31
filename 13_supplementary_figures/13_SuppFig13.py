#!/usr/bin/env python3
"""Supplementary Figure 13: MGUS malignant PCs called MM, and malignancy scores in external single-cell datasets.

Purpose:      All scores are out-of-sample. A, percentage of malignant PCs called MM (MST score > 0.5) per MGUS and MM
              patient of the MGUS-MM classifier; diamonds, test-set patients. B, MST score and malignancy score of the
              MGUS-like and MM-like malignant PCs of the 12 MGUS patients with at least 20 of each (patient medians);
              two-sided Wilcoxon signed-rank P for the malignancy score (the MST score separates the groups by definition). C, malignancy score of the malignant and normal PCs of each sample, as labeled by
              the authors of three external studies; AUROC of malignant versus normal samples with its 95% CI.

Inputs:       results/classifiers: misclassified_mgus_cells.pkl.gz, misclassified_mgus_patients.tsv,
              misclassified_mgus_within_patient.tsv, external_malignancy_units.tsv,
              external_malignancy_stats.tsv.

Outputs:      figures/SuppFig13.svg, SuppFig13.pdf; figures/source_data/SuppFig13_*.tsv.

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
from figstyle import style, rich_text, p_runs, n_runs, rng, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("SuppFig13", [
    Slot("pct_called_mm", "A", 0, 0, 27, 50),
    Slot("mgus_like_vs_mm_like", "B", 29, 0, 58, 50),
    Slot("external_malignancy", "C", 0, 52, 122, 50),
], max_height=247)
STUDIES = ["Ledergor et al.", "Boiarsky et al.", "Dang et al."]
EXT = ["Donor:normal", "MGUS:normal", "SMM:normal", "MM:normal", "MGUS:tumor", "SMM:tumor", "MM:tumor"]
STAGE3 = {"Donor": "Donor", "MGUS": "MGUS", "SMM": "IRSMM", "MM": "MM"}       # SMM is not split by risk in these studies
EXT_COLORS = {g: (P.NORMAL if g.endswith("normal") else P.TUMOR)[STAGE3[g.split(":")[0]]] for g in EXT}


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig13_%s.tsv" % key, sep="\t", index=False)


def panel_pct(sf, cells):
    pat = cells.groupby(["patient", "disease_status", "split"])["mm_like"].mean().mul(100).rename("value").reset_index()
    ref = pd.read_csv(paths.result("classifiers", "misclassified_mgus_patients.tsv"), sep="\t").set_index("patient")["pct_mm_like"]
    mg = pat[pat["disease_status"] == "MGUS"].set_index("patient")["value"]
    assert len(mg) == 16 and np.allclose(mg.sort_index(), ref.sort_index())
    assert int((pat["disease_status"] == "MM").sum()) == 36
    order = ["MGUS", "MM"]
    ax = axes_mm(sf, 10.5, 4.0, 15.0, 36.0)
    plots.box_strip(ax, pat.rename(columns={"disease_status": "group"}), order, {s: P.TUMOR[s] for s in order}, "sf13a",
                    points=False, width=0.6)
    r = rng("sf13a")
    for i, s in enumerate(order):                           # dots: cross-validation patients; diamonds: test-set patients
        for split, mk, size in (("cross-validation", "o", 1.8), ("test", "D", 1.7)):
            v = pat.loc[(pat["disease_status"] == s) & (pat["split"] == split), "value"].values
            ax.scatter(i + r.uniform(-0.2, 0.2, len(v)), v, marker=mk, s=size ** 2, c=P.TUMOR[s], edgecolors="black",
                       linewidths=0.3, zorder=3)
    plots.refline(ax, 50)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Malignant PCs called MM (%)")
    for k, (lab, mk) in enumerate((("Cross-validation", "o"), ("Test set", "D"))):
        plots.key(sf, 1.0 / sf.w_mm, 1 - (44.6 + 2.1 * k) / sf.h_mm, [(lab, "#FFFFFF")], marker=mk, ms=1.9,
                  transform=sf.transSubfigure, size=SMALL, gid="pn-ksplit%d" % k)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    source("A", pat.rename(columns={"value": "pct_called_mm"}))


def panel_paired(sf, cells, W):
    mg = cells[cells["disease_status"] == "MGUS"]
    both = [p for p, q in mg.groupby("patient") if q["mm_like"].sum() >= 20 and (~q["mm_like"]).sum() >= 20]
    out = []
    for j, (col, lab) in enumerate((("mst", "MST score"), ("malignancy", "Malignancy score"))):
        paired = mg[mg["patient"].isin(both)].groupby(["patient", "mm_like"])[col].median().unstack()
        assert len(paired) == int(W.loc["malignancy", "n_patients"]) == 12
        ax = axes_mm(sf, 11.0 + 29.0 * j, 4.0, 16.0, 36.0)
        for _, rw in paired.iterrows():
            ln, = ax.plot([0, 1], [rw[False], rw[True]], color="#999999", lw=0.4, zorder=1)
            ln.set_gid("pair%d" % j)
        for x, k, s_ in ((0, False, "MGUS"), (1, True, "MM")):
            ax.scatter(np.full(len(paired), x), paired[k], s=1.8 ** 2, c=P.TUMOR[s_], edgecolors="black", linewidths=0.3,
                       zorder=3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MGUS-like\ncells", "MM-like\ncells"], fontsize=SMALL)
        plots.n_below(ax, [len(paired)] * 2)
        if col == "mst":                                   # MM-like = MST score above 0.5, by definition
            plots.refline(ax, 0.5)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1.0])
        else:
            lo = np.floor(paired.min().min() * 10) / 10
            ax.set_ylim(lo - 0.04, 1.0)
            plots.brackets(ax, [dict(i=0, j=1, p=float(W.loc["malignancy", "p"]), color=P.BRACKET_STAGE)],
                           {0: float(paired[False].max()), 1: float(paired[True].max())}, ymin=lo - 0.04, key="sf13b")
            ax.yaxis.set_major_locator(plots.nice_locator(4))
        plots.plain_ticks(ax.yaxis)
        ax.set_ylabel(lab + " (patient median)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        o = paired.rename(columns={False: "MGUS-like cells", True: "MM-like cells"}).reset_index().assign(score=lab)
        o.columns.name = None
        out.append(o)
    source("B", pd.concat(out))


def panel_external(sf):
    units = pd.read_csv(paths.result("classifiers", "external_malignancy_units.tsv"), sep="\t")
    st = pd.read_csv(paths.result("classifiers", "external_malignancy_stats.tsv"), sep="\t").set_index("study")
    jobs, axes = [], []
    for k, study in enumerate(STUDIES):
        x0 = 9.5 + 38.0 * k
        ax = axes_mm(sf, x0, 7.5, 30.5, 22.0)
        u = units[units["study"] == study].copy()
        u["group"] = u["disease_status"] + ":" + u["annotation"]
        assert set(u["group"]) <= set(EXT)
        tops = plots.box_strip(ax, u.rename(columns={"malignancy": "value"}), EXT, EXT_COLORS, "sf13c-%d" % k, point=1.7,
                               labels=[g.split(":")[0] for g in EXT], rotate=90, label_size=SMALL, n_labels="row")
        plots.refline(ax, 0.5)
        s = st.loc[study]                                 # AUROC of malignant vs normal samples: block against block
        jobs.append((ax, [dict(i=1.5, j=5.0, span_i=(0, 3), span_j=(4, 6), color=P.BRACKET_MAL, lines=[
            [("AUROC %.2f (95%% CI %.2f–%.2f)" % (s["unit_auroc"], s["unit_lo"], s["unit_hi"]), "")]])], tops, "sf13c%d" % k))
        axes.append(ax)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        plots.plain_ticks(ax.yaxis)
        ax.set_ylabel("Malignancy score" if k == 0 else "")
        ax.set_title(study, fontsize=BASE, pad=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        tr = plots.blended_transform_factory(ax.transData, ax.transAxes)
        for xc, lab in [(1.5, "Normal PCs"), (5.0, "Malignant PCs")]:
            rich_text(ax, xc, 0, [(lab, "")], size=SMALL, ha="center", va="top", transform=plots.pts(ax, 0, -26.0, tr),
                      gid="pn-grp%d%s" % (k, lab[0]))
    arts = plots.brackets_shared(jobs, -0.04)               # one y scale for the three studies
    for ax, a in zip(axes, arts):
        plots.separators(ax, [3.5], a)                      # normal PCs left, malignant PCs right
    source("C", units[["study", "sample_id", "disease_status", "annotation", "n_cells", "malignancy"]])


def main():
    cells = pd.read_pickle(paths.result("classifiers", "misclassified_mgus_cells.pkl.gz"))
    W = pd.read_csv(paths.result("classifiers", "misclassified_mgus_within_patient.tsv"), sep="\t").set_index("variable")
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_pct(pn["pct_called_mm"], cells)
        panel_paired(pn["mgus_like_vs_mm_like"], cells, W)
        panel_external(pn["external_malignancy"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig13", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
