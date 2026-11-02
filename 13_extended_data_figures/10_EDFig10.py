#!/usr/bin/env python3
"""Extended Data Figure 10: stage associations of the MST score and of the five progression-associated signatures with
and without adjustment for genetic alterations, and histone/chaperone activity and tumor ploidy.

Purpose:      A, change per disease-stage step (MGUS to MM, in SD of the outcome) from a linear regression on stage,
              without (gray) and with (purple) adjustment for the genetic alterations; 95% CI from standard errors
              clustered by patient; two-sided Wald P. B, tumor ploidy from whole-genome sequencing of the same BM sample by
              disease stage (Spearman correlation with stage). C, histone/chaperone activity against tumor ploidy, with a
              density outline per disease stage (stages with at least 10 tumors; Gaussian kernel density, region holding
              about 70% of the stage's tumors); Spearman correlation overall and within stage. D, change of
              histone/chaperone activity per stage step without and with adjustment for ploidy (tumors with WGS) or HRD
              (all tumors).

Inputs:       results/stage_and_progression: stage_effects_genetics.tsv, bm_tumor_units.tsv, histone_dna_content.tsv,
              histone_dna_content_units.tsv, histone_dna_content_tests.tsv.

Outputs:      figures/EDFig10.svg, EDFig10.pdf; figures/source_data/EDFig10_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy import stats

import palettes as P
import paths
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, p_runs, rho_runs, join_runs, n_runs, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("EDFig10", [
    Slot("stage_effects_adjusted", "A", 0, 0, 122, 52),
    Slot("ploidy_by_stage", "B", 124, 0, 59, 52),
    Slot("histone_vs_ploidy", "C", 0, 54, 95, 60),
    Slot("histone_stage_effects", "D", 97, 54, 86, 60),
], max_height=247)
MODELS = [("unadjusted", "Unadjusted", "#9E9E9E", -0.17), ("adjusted", "Adjusted for genetic alterations", "#54278F", 0.17)]
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
H = "mean_expr:histone_chaperone_gene"
GRAY, PURPLE = "#9E9E9E", "#54278F"


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/EDFig10_%s.tsv" % key, sep="\t", index=False)


def close(a, b, rel=1e-6):
    assert abs(a - b) <= rel * max(1e-300, abs(b)), (a, b)


def test(T, name, detail):
    row = T[(T["test"] == name) & (T["detail"] == detail)]
    assert len(row) == 1, (name, detail)
    return float(row["statistic"].iloc[0]), float(row["P"].iloc[0])


# ---- A ---------------------------------------------------------------------------------------------------------------
def panel_genetics(sf, eff, n, n_pt):
    outcomes = list(dict.fromkeys(eff["outcome"]))
    at_mm(sf, 4.5, 1.8, [("BM tumors with at least 30 malignant PCs (", ""), ("n", "i"),
                         (" = %d from %d patients)" % (n, n_pt), "")], "pn-title")
    ax = axes_mm(sf, 28.0, 9.0, 46.0, 30.0)
    for model, lab, col, dy in MODELS:
        d = eff[eff["model"] == model].set_index("outcome").loc[outcomes]
        y = [i + dy for i in range(len(outcomes))]
        ax.errorbar(d["coef_sd_per_stage"], y, xerr=[d["coef_sd_per_stage"] - d["ci_low"],
                                                     d["ci_high"] - d["coef_sd_per_stage"]],
                    fmt="o", ms=2.6, color=col, ecolor=col, elinewidth=0.8, capsize=0, zorder=3)
    ax.axvline(0, color="#808080", lw=0.5, ls=(0, (3, 2)), zorder=1).set_gid("ref-zero")
    ax.set_yticks(range(len(outcomes)))
    ax.set_yticklabels(outcomes)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(outcomes) - 0.5, -0.5)
    ax.set_xlim(-0.05, 0.65)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    plots.plain_ticks(ax.xaxis)
    ax.set_xlabel("Change per disease-stage step (SD)\nstage coded 0 (MGUS) to 4 (MM)", fontsize=SMALL)
    ax.tick_params(axis="x", labelsize=SMALL)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # P columns, one per model, with the model's mark in the header
    tr = blended_transform_factory(sf.transSubfigure, ax.transData)
    for k, (model, lab, col, _) in enumerate(MODELS):
        x_mm = 80.0 + 21.0 * k
        d = eff[eff["model"] == model].set_index("outcome").loc[outcomes]
        sf.add_artist(Line2D([x_mm / sf.w_mm], [1 - 6.3 / sf.h_mm], marker="o", ms=2.6, color=col, ls="none",
                             transform=sf.transSubfigure, clip_on=False))
        at_mm(sf, x_mm + 1.6, 6.3, [("P", "i"), (", unadjusted" if k == 0 else ", adjusted", "")], "pn-hdr%d" % k,
              size=SMALL)
        for i, p in enumerate(d["P"].values):
            rich_text(ax, (x_mm - 1.3) / sf.w_mm, i, p_runs(p), size=SMALL, va="center", transform=tr,
                      gid="pn-p%d%d" % (k, i))
    items = [(lab, col) for _, lab, col, _ in MODELS]
    plots.key(sf, 28.0 / sf.w_mm, 1 - 47.5 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.6, ncol=2,
              col_pt=60.0, gid="pn-key")
    source("A", eff[["outcome", "model", "n_units", "coef_sd_per_stage", "ci_low", "ci_high", "P", "retained_pct"]])


# ---- B ---------------------------------------------------------------------------------------------------------------
def panel_ploidy_by_stage(sf, m, T):
    rho, p = test(T, "Spearman", "ploidy vs disease stage (0-4)")
    r2, p2 = stats.spearmanr(m["ploidy"], m["disease_status"].map(STAGES.index))
    close(r2, rho)
    close(p2, p)
    at_mm(sf, 33.0, 1.8, "Tumor ploidy by disease stage", "pn-title", ha="center")
    ax = axes_mm(sf, 15.0, 8.0, 40.0, 26.0)
    plots.box_strip(ax, m.assign(group=m["disease_status"], value=m["ploidy"]), STAGES, P.TUMOR, "sf16b", point=1.4,
                    width=0.6, rotate=90, label_size=SMALL, n_labels="row")
    plots.refline(ax, 2.0)
    ax.set_ylim(1.8, 3.0)
    ax.set_yticks([2.0, 2.4, 2.8])
    plots.plain_ticks(ax.yaxis)
    ax.tick_params(axis="y", labelsize=SMALL)
    ax.set_ylabel("Tumor ploidy (WGS)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    rich_text(ax, 1.0, 1, join_runs([("Spearman ", "")] + rho_runs(rho), p_runs(p)), size=SMALL, ha="right", va="top",
              transform=plots.pts(ax, 0, 1.0, ax.transAxes), gid="pn-rhoB")
    source("B", m[["patient_id", "disease_status", "ploidy"]])


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_histone_vs_ploidy(sf, m, T):
    rho, p = test(T, "Spearman", "ploidy vs histone/chaperone activity")
    rho_w, p_w = test(T, "Spearman (partial, within stage)", "ploidy vs histone/chaperone activity")
    r2, p2 = stats.spearmanr(m["ploidy"], m[H])
    close(r2, rho)
    close(p2, p)
    at_mm(sf, 47.5, 1.8, "Histone/chaperone activity and tumor ploidy", "pn-title", ha="center")
    ax = axes_mm(sf, 14.0, 7.0, 60.0, 38.0)
    x0, x1, y0, y1 = 1.8, 3.0, -0.01, 0.17
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    for s_ in STAGES:                                  # density outline per stage with at least 10 tumors
        sub = m[m["disease_status"] == s_]
        if len(sub) < 10:
            continue
        kde = stats.gaussian_kde(np.vstack([sub["ploidy"], sub[H]]))
        dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        lev = np.quantile(kde(np.vstack([sub["ploidy"], sub[H]])), 0.3)     # region holding about 70% of the tumors
        cf = ax.contourf(gx, gy, dens, levels=[lev, dens.max() * 1.01], colors=[P.TUMOR[s_]], alpha=0.07, zorder=1)
        cl = ax.contour(gx, gy, dens, levels=[lev], colors=[P.TUMOR[s_]], linewidths=0.9, zorder=2)
        for art in (cf, cl):
            art.set_gid("density-" + s_)
    plots.scatter_groups(ax, m["ploidy"], m[H], m["disease_status"], P.TUMOR, "sf16c", s=2.0, order=STAGES)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([2.0, 2.4, 2.8])
    plots.plain_ticks(ax.xaxis)
    ax.yaxis.set_major_locator(plots.nice_locator(3))
    ax.tick_params(labelsize=SMALL)
    ax.set_xlabel("Tumor ploidy (WGS)")
    ax.set_ylabel("Histone/chaperone activity")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    rich_text(ax, 1.0, 1, join_runs([("All, ", "")] + rho_runs(rho), p_runs(p)), size=SMALL, ha="right", va="top",
              transform=plots.pts(ax, -1, 2.0, ax.transAxes), gid="pn-rhoC1")
    rich_text(ax, 1.0, 1, join_runs([("Within stage, ", "")] + rho_runs(rho_w), p_runs(p_w)), size=SMALL, ha="right",
              va="top", transform=plots.pts(ax, -1, -5.5, ax.transAxes), gid="pn-rhoC2")
    plots.key(sf, 14.0 / sf.w_mm, 1 - 53.5 / sf.h_mm, [(s, P.TUMOR[s]) for s in STAGES], transform=sf.transSubfigure,
              size=SMALL, ms=2.4, ncol=5, gid="pn-keyC")
    at_mm(sf, 88.0, 54.5, n_runs(len(m)) + [(" tumors", "")], "pn-nC", size=SMALL, ha="right")
    source("C", m[["patient_id", "disease_status", "ploidy", H]].rename(columns={H: "histone_chaperone_activity"}))


# ---- D ---------------------------------------------------------------------------------------------------------------
def panel_histone_stage_effects(sf, eff):
    e = eff.set_index("model")
    rows = [("%d tumors with WGS" % int(e.loc["WGS tumors, unadjusted", "n_units"]), None),
            ("Unadjusted", "WGS tumors, unadjusted"), ("Adjusted for ploidy", "WGS tumors, adjusted for ploidy"),
            ("All %d tumors" % int(e.loc["All tumors, unadjusted", "n_units"]), None),
            ("Unadjusted", "All tumors, unadjusted"), ("Adjusted for HRD", "All tumors, adjusted for HRD")]
    at_mm(sf, 4.5, 1.8, "Histone/chaperone activity per disease-stage step", "pn-title")
    ax = axes_mm(sf, 34.0, 8.0, 26.0, 4.6 * len(rows))
    tr = blended_transform_factory(sf.transSubfigure, ax.transData)
    for y, (lab, key) in enumerate(rows):
        if key is None:
            rich_text(ax, 1.5 / sf.w_mm, y, [(lab, "b")], size=SMALL, va="center", transform=tr, gid="pn-g%d" % y)
            continue
        r = e.loc[key]
        col = GRAY if lab == "Unadjusted" else PURPLE
        ax.errorbar([r["coef_sd_per_stage"]], [y], xerr=[[r["coef_sd_per_stage"] - r["ci_low"]],
                                                       [r["ci_high"] - r["coef_sd_per_stage"]]],
                    fmt="o", ms=2.4, color=col, ecolor=col, elinewidth=0.8, capsize=0, zorder=3)
        rich_text(ax, 3.5 / sf.w_mm, y, [(lab, "")], size=SMALL, va="center", transform=tr, gid="pn-l%d" % y)
        rich_text(ax, 63.0 / sf.w_mm, y, p_runs(r["P"]), size=SMALL, va="center", transform=tr, gid="pn-p%d" % y)
    ax.axvline(0, color="#808080", lw=0.5, ls=(0, (3, 2)), zorder=1).set_gid("ref-zero")
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(-0.05, 0.5)
    ax.set_xticks([0, 0.2, 0.4])
    plots.plain_ticks(ax.xaxis)
    ax.tick_params(axis="x", labelsize=SMALL)
    ax.set_xlabel("Change per stage step (SD)", fontsize=SMALL)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    source("D", eff)


def main():
    eff = pd.read_csv(paths.result("stage_and_progression", "stage_effects_genetics.tsv"), sep="\t")
    n_pt = pd.read_csv(paths.result("stage_and_progression", "bm_tumor_units.tsv"), sep="\t")["patient_id"].nunique()
    n = int(eff["n_units"].iloc[0])
    assert (n, n_pt, eff["outcome"].nunique()) == (181, 179, 6)
    mst = eff[eff["outcome"] == "MST score"].set_index("model")["coef_sd_per_stage"]
    assert round(mst["unadjusted"], 2) == 0.53 and round(mst["adjusted"], 2) == 0.46        # Results: 0.53 -> 0.46 SD
    m = pd.read_csv(paths.result("stage_and_progression", "histone_dna_content_units.tsv"), sep="\t")
    T = pd.read_csv(paths.result("stage_and_progression", "histone_dna_content_tests.tsv"), sep="\t")
    hd = pd.read_csv(paths.result("stage_and_progression", "histone_dna_content.tsv"), sep="\t")
    assert len(m) == int(hd.loc[0, "n_units"])
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_genetics(pn["stage_effects_adjusted"], eff, n, n_pt)
        panel_ploidy_by_stage(pn["ploidy_by_stage"], m, T)
        panel_histone_vs_ploidy(pn["histone_vs_ploidy"], m, T)
        panel_histone_stage_effects(pn["histone_stage_effects"], hd)
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("EDFig10", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
