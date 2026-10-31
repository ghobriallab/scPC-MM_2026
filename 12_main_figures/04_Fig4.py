#!/usr/bin/env python3
"""Figure 4: random forest classifiers for malignancy and MGUS-to-MM state prediction in plasma cells.

Purpose:      A, design of the two classifiers and the features each uses. B, malignancy classifier: test-set
              confusion matrix and SHAP values of the 20 features with the largest mean absolute SHAP value. C, the same
              for the MGUS-MM classifier. D, UMAPs of all PCs by PC status and by malignancy score, and the mean score of
              each patient's PCs by disease stage, tissue and PC status (two-sided Wilcoxon rank-sum tests,
              Benjamini-Hochberg q; figlib/stage_tests.py). E, the same for disease stage and the MST score.

Inputs:       results/figure_inputs: confusion_counts.tsv, shap_malignancy.npz, shap_mgus_mm.npz, pcs_cells.npz,
              patient_id_mean_probability.csv.gz; results/stage_and_progression/stage_comparisons.tsv.

Outputs:      figures/Fig4.svg, Fig4.pdf; figures/source_data/Fig4_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths). Also icons.py.
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

import icons
import names
import palettes as P
import paths
import plots
import stage_tests as ST
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, SMALL, BASE, TITLE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("Fig4", [
    Slot("classifier_schematic", "A", 0, 0, 183, 40),
    Slot("malignancy_classifier", "B", 0, 42, 90, 56),
    Slot("mgus_mm_classifier", "C", 93, 42, 90, 56),
    Slot("malignancy_scores", "D", 0, 100, 90, 70),
    Slot("mst_scores", "E", 93, 100, 90, 70),
], max_height=170)
PT_MM = 25.4 / 72.0


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/Fig4_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center", **kw):
    """Text at (x, y) mm from the panel's top-left corner."""
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs, size=size, ha=ha, va=va, transform=sf.transSubfigure,
                     gid=gid, **kw)


def cells():
    """All PCs of the atlas in the order of the embedding: UMAP, tissue, PC status, stage, malignancy and MST scores.
    SMM risk groups of the IMWG 2/20/20 model (risk.py)."""
    z = np.load(paths.cache("pcs_cells.npz"))
    cat = lambda k: z[k + "__cats"].astype(str)[z[k + "__codes"]]
    d = pd.DataFrame({"x": z["umap"][:, 0], "y": z["umap"][:, 1], "patient_id": cat("patient_id"),
                      "tissue": cat("sample_type"), "annotation": cat("annotation"), "stage": cat("disease_status"),
                      "malignancy": z["malignancy"], "mst": z["mst"]})
    d["stage"] = ST.imwg_stages(d["patient_id"], d["tissue"], d["stage"])      # IMWG 2/20/20 risk groups (risk.py)
    d["status"] = d["tissue"] + " " + d["annotation"].map({"normal": "normal", "tumor": "malignant"})
    return d


# ---- B, C: test-set confusion matrices and SHAP values -----------------------------------------------------------
def confusion_pair(sf, tier, labels, label_names):
    cc = pd.read_csv(paths.cache("confusion_counts.tsv"), sep="\t")
    cc = cc[cc["tier"] == tier]
    assert (cc["cells"] == cc["expected"]).all()
    M = cc.pivot(index="true", columns="predicted", values="cells").loc[labels, labels].values.astype(float)
    frac = M / M.sum(1, keepdims=True)
    comma = FuncFormatter(lambda v, _: "{:,.0f}".format(v))
    for y, mat, cmap, fmt, kw in [(7.0, M, "viridis", lambda v: "{:,.0f}".format(v), dict(vmin=0, vmax=M.max())),
                                  (24.5, frac, P.CMAP_DIVERGING, lambda v: "%.2f" % v, dict(vmin=0, vmax=1, center=0.5))]:
        ax = axes_mm(sf, 14.5, y, 15, 14)
        mesh, _ = plots.confusion(ax, mat, label_names, "%s-%s" % (tier, "n" if mat is M else "f"), cmap, fmt, **kw)
        ax.tick_params(labelsize=SMALL)
        ax.set_yticklabels(label_names, rotation=0, va="center")
        ax.set_ylabel("True label", labelpad=1.5)
        if mat is M:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Predicted label", labelpad=1.5)
        cax = axes_mm(sf, 30.6, y, 1.3, 14)
        cb = plots.colorbar(sf, mesh, cax)
        if mat is M:
            cb.locator = MaxNLocator(nbins=3, integer=True)
            cb.formatter = comma
            cb.update_ticks()
        else:
            cb.set_ticks([0, 0.5, 1])
            plots.plain_ticks(cb.ax.yaxis)
    return cc, M


def shap_panel(sf, tier, key):
    z = np.load(paths.cache("shap_%s.npz" % tier))
    feats = [names.feature(str(f)) for f in z["feature"]]
    x0, w, y0, h = 60.5, 21.0, 5.5, 41.0
    ax = axes_mm(sf, x0, y0, w, h)
    order = plots.shap_violin(ax, z["shap"], z["value"], feats, "fig4-shap-" + tier)
    ax.tick_params(axis="y", labelsize=SMALL, pad=1.0)
    ax.tick_params(axis="x", labelsize=SMALL)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    plots.plain_ticks(ax.xaxis)
    lab = "SHAP value (impact on model output)"
    assert rich_width_pt(sf, [(lab, "")], BASE) * PT_MM / 2 <= min(x0 + w / 2, sf.w_mm - x0 - w / 2)
    ax.set_xlabel(lab, labelpad=1.5)
    cax = axes_mm(sf, 84.6, y0 + 4.0, 1.2, h - 8.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=Normalize(0, 1), cmap=plots.red_blue()), cax, ticks=[])
    at_mm(sf, 85.2, y0 + 2.4, [("High", "")], "pn-shaphigh-" + tier, size=SMALL, ha="center")
    at_mm(sf, 85.2, y0 + h - 2.4, [("Low", "")], "pn-shaplow-" + tier, size=SMALL, ha="center")
    cb.set_label("Feature value", fontsize=SMALL, labelpad=1.0)
    imp = np.abs(z["shap"]).mean(0)
    shown = set(order)
    return pd.DataFrame({"feature": feats, "mean_abs_shap": imp, "shown": [i in shown for i in range(len(feats))],
                         "cells": len(z["shap"])}).sort_values("mean_abs_shap", ascending=False)


def panel_classifier(sf, key, tier, title, labels, label_names):
    at_mm(sf, 5.0, 1.6, [(title, "")], "pn-title-" + tier, size=TITLE)
    cc, _ = confusion_pair(sf, tier, labels, label_names)
    imp = shap_panel(sf, tier, key)
    source(key + "_confusion", cc.drop(columns="expected"))
    source(key + "_shap_importance", imp)


# ---- D, E: scores across the cohort ------------------------------------------------------------------------------
def umap_pair(sf, c, cat_col, palette, order_cats, score_col, cmap, title_cat, title_score, key, ends):
    """Left, PCs by category; right, PCs by score, with the color bar's ends named after the two classes the classifier
    separates (`ends`)."""
    xy = c[["x", "y"]].values
    a1 = axes_mm(sf, 11.0, 4.5, 24.0, 24.0)
    plots.umap(a1, xy, np.array([palette[v] for v in c[cat_col]]), key + "-cat", s=0.02)
    a1.set_title(title_cat, fontsize=BASE, pad=2.0)
    a2 = axes_mm(sf, 47.0, 4.5, 24.0, 24.0)
    v = c[score_col].values
    norm = Normalize(0, 1)
    plots.umap(a2, xy, plots.matplotlib_cm(cmap)(norm(v)), key + "-score", s=0.02, order=np.argsort(v, kind="stable"))
    a2.set_title(title_score, fontsize=BASE, pad=2.0)
    cax = axes_mm(sf, 72.6, 7.0, 1.3, 19.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap=cmap), cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0 (%s)" % ends[0], "0.5", "1 (%s)" % ends[1]])
    items = [(g, palette[g]) for g in order_cats]
    plots.key(sf, 11.0 / sf.w_mm, 1 - 31.3 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.4,
              ncol=len(items), gid="pn-key" + key)


def risk_bracket(t, block, anchor, partners, cutoff):
    """One more stage bracket: the anchor stage against each partner stage of one block, if all differ at q < cutoff."""
    pos = {g: k for k, g in enumerate(ST.GROUPS)}
    df = t[block]
    q = []
    for p in partners:
        r = df[df["comparison"].isin(["%s,%s" % (anchor, p), "%s,%s" % (p, anchor)])]
        assert len(r) == 1, (anchor, p)
        q.append(float(r["fdr"].iloc[0]))
    if max(q) >= cutoff:
        return []
    return [dict(kind="stage", block=block, anchor_x=pos["%s:%s" % (anchor, block)],
                 partner_xs=[pos["%s:%s" % (p, block)] for p in partners], q=max(q), n=len(q))]


def panel_scores(sf, key, col, cat_col, palette, order_cats, cmap, title_cat, title_score, ylabel, panel_name, ends):
    c = cells()
    umap_pair(sf, c, cat_col, palette, order_cats, "malignancy" if col == "Malignancy Score" else "mst", cmap, title_cat,
              title_score, key, ends)
    units = ST.score_units()
    t = ST.tests(units, col)
    ST.check_against_saved(panel_name, t)
    specs = ST.brackets(units, col, t, 0.01, "score", score_rule=True)
    if col == "MST Score":                     # low-risk SMM vs intermediate- and high-risk SMM, malignant BM PCs (Results)
        specs += risk_bracket(t, "BM:tumor", "LRSMM", ["IRSMM", "HRSMM"], 0.01)
    ax = axes_mm(sf, 10.0, 39.0, 79.0, 19.0)
    ax.set_ylim(0, 1.0)
    plots.stage_box(ax, units, col, specs, key + "-box", refline_at=0.5, ylabel=ylabel)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"])
    out = units[["patient_id", "disease_status", "sample_type", "annotation", col]].rename(columns={col: "score"})
    source(key, out.sort_values(["sample_type", "annotation", "disease_status", "patient_id"]))
    source(key + "_brackets", pd.DataFrame([dict(block=b["block"], kind=b["kind"], q=b["q"], comparisons=b["n"])
                                            for b in specs]))


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        icons.classifier_schematic(pn["classifier_schematic"])
        panel_classifier(pn["malignancy_classifier"], "malignancy_classifier", "malignancy", "Malignancy classification",
                         ["normal", "tumor"], ["Normal", "Malignant"])
        panel_classifier(pn["mgus_mm_classifier"], "mgus_mm_classifier", "mgus_mm", "MGUS-MM classification",
                         ["MGUS", "MM"], ["MGUS", "MM"])
        panel_scores(pn["malignancy_scores"], "malignancy_scores", "Malignancy Score", "status", P.PC_STATUS,
                     ["BM normal", "PB normal", "BM malignant", "PB malignant"], P.CMAP_MALIGNANCY, "Plasma cell status",
                     "Malignancy score", "Malignancy score", "score Malignancy Score", ("normal", "malignant"))
        panel_scores(pn["mst_scores"], "mst_scores", "MST Score", "stage", P.STAGE, ST.STAGES, P.CMAP_MST,
                     "Disease stage", "MST score", "MST score", "score MST Score", ("MGUS-like", "MM-like"))
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("Fig4", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
