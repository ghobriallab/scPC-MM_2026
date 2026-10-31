#!/usr/bin/env python3
"""Supplementary Figure 12: feature selection and parameter tuning of the malignancy and MST classifiers.

Purpose:      Mean and SD of the accuracy over the cross-validation folds. A, malignancy classifier (random forest and
              logistic regression): accuracy of the 16 feature sets under three smoothing strategies; selected setting
              boxed. B, MGUS-MM random forest with the malignancy classifier at its selected setting: accuracy by feature
              set (top) and the MM and MGUS accuracy of every setting (bottom). C, accuracy against the number of
              neighbors k of the kNN smoothing; k = 30 marked. D, the selected settings. E, change in the out-of-fold
              AUROC of patient-level MST scores when one feature, or the six features with the lowest mean absolute
              SHAP value, is removed.
              Feature-set codes: auc = AUCell, mex = mean expression; gm16 = the 16 signatures (sig), age = senescence
              signature (sen), arm = CNV-like scores of driver arms (arm), cyto = genes overexpressed through
              translocations (cyto).

Inputs:       results/figure_inputs: tuning_tumor_normal.tsv, tuning_mgus_mm.tsv, tuning_knn.tsv;
              results/classifiers/feature_ablation_mst.tsv.

Outputs:      figures/SuppFig12.svg, SuppFig12.pdf; figures/source_data/SuppFig12_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

import names
import paths
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, text_advance_pt, LW, SMALL, BASE, TITLE
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("SuppFig12", [
    Slot("tuning_malignancy", "A", 0, 0, 183, 51),
    Slot("tuning_mgus_mm", "B", 0, 53, 92, 117),
    Slot("tuning_knn", "C", 95, 53, 88, 31),
    Slot("final_parameters", "D", 95, 86, 88, 20),
    Slot("feature_ablation", "E", 95, 108, 88, 62),
], max_height=247)
PT_MM = 25.4 / 72.0

PARTS = [("gm16", "sig"), ("age", "sen"), ("arm", "arm"), ("cyto", "cyto")]
COMBOS = ["gm16", "gm16+age", "gm16+arm", "gm16+cyto", "gm16+age+cyto", "gm16+arm+cyto", "gm16+age+arm",
          "gm16+age+arm+cyto"]                                        # order of the feature sets
CODES = ["%s:%s" % (n, c) for n in ("auc", "mex") for c in COMBOS]


def label(code):
    """'mex:gm16+age+cyto' -> 'MEX(sig+sen+cyto)'."""
    norm, combo = code.split(":")
    short = dict(PARTS)
    return "%s(%s)" % (norm.upper(), "+".join(short[p] for p in combo.split("+")))


LABELS = [label(c) for c in CODES]
SMOOTH = {"single-cell": ("Single-cell", "#999999"), "cohort-knn": ("Cohort-level kNN", "#0072B2"),
          "sample-knn": ("Sample-level kNN", "#D55E00")}
CLASSIFIERS = {"RandomForest": "Random forest", "LogisticRegression": "Logistic regression"}
SELECTED_TN = "cohort-knn|mex:gm16+age+cyto"                     # selected setting
SELECTED_DS = "cohort-knn|auc:gm16+arm"
SELECTED_K = 30
MARKERS = dict(zip(["sig", "sig+sen", "sig+arm", "sig+cyto", "sig+sen+cyto", "sig+arm+cyto", "sig+sen+arm",
                    "sig+sen+arm+cyto"], ["o", "X", "s", "P", "D", "v", "^", "*"]))
NORM_FILL = {"AUC": "#FFFFFF", "MEX": "#A6A6A6"}
KNN = [("mean_acc", "Overall", "#000000"), ("MM_mean_acc", "MM", "#A11A5B"), ("MGUS_mean_acc", "MGUS", "#E69F00"),
       ("normal_mean_acc", "Normal PC", "#56B4E9")]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig12_%s.tsv" % key, sep="\t", index=False)


# ---- bump charts (A, B top) ------------------------------------------------------------------------------------------
def bump(ax, d, hue_col, feat_col, ylim, key):
    """Mean accuracy (line, dots) +- SD (band) of each feature set, one line per smoothing strategy."""
    xs = np.arange(len(CODES))
    for sm in SMOOTH:
        q = d[d[hue_col] == sm].set_index(feat_col).reindex(CODES)
        if q["mean_acc"].isna().all():
            continue
        assert q["mean_acc"].notna().all(), (key, sm)
        m, s = q["mean_acc"].values, q["std_acc"].values
        band = ax.fill_between(xs, m - s, m + s, color=SMOOTH[sm][1], alpha=0.18, linewidth=0, zorder=1)
        band.set_gid("band-%s-%s" % (key, sm))
        ax.plot(xs, m, color=SMOOTH[sm][1], lw=0.75, marker="o", ms=1.8, mew=0, zorder=3)
    ax.set_xlim(-0.6, len(CODES) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS, rotation=90, fontsize=SMALL)
    ax.yaxis.set_major_locator(plots.nice_locator(4))
    plots.plain_ticks(ax.yaxis)
    ax.tick_params(axis="y", labelsize=SMALL)
    ax.grid(axis="x", color="#D9D9D9", lw=0.3, ls=(0, (2, 2)), zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def box_selected(ax, code, key):
    """Outline the selected feature set: its column and its x label."""
    i = CODES.index(code)
    lab_pt = text_advance_pt(None, LABELS[i], SMALL, weight="bold") + ax.xaxis.get_tick_padding() + 3.0   # label is bold
    h_pt = plots.axes_size_pt(ax)[1]
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    r = Rectangle((i - 0.42, -lab_pt / h_pt), 0.84, 1.0 + lab_pt / h_pt, transform=tr, fill=False, lw=0.5,
                  edgecolor="black", clip_on=False, zorder=4)
    r.set_gid("box-selected-" + key)
    ax.add_patch(r)
    ax.get_xticklabels()[i].set_fontweight("bold")


def smoothing_key(sf, x, y, which, gid):
    items = [(SMOOTH[s][0], SMOOTH[s][1]) for s in which]
    col = max(rich_width_pt(sf, [(t, "")], SMALL) for t, _ in items) + 9.0
    plots.key(sf, x / sf.w_mm, 1 - y / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=3.0, line=True,
              ncol=len(items), col_pt=col, gid=gid)


def panel_malignancy(sf, T):
    d = T[T["feature_set"].isin(CODES)].copy()
    lo = (d["mean_acc"] - d["std_acc"]).min()
    hi = (d["mean_acc"] + d["std_acc"]).max()
    ylim = (lo - 0.005, hi + 0.005)
    at_mm(sf, 5.0, 1.6, "Malignancy classification", "pn-title", size=TITLE)
    smoothing_key(sf, 98.0, 1.6, list(SMOOTH), "pn-key")
    for k, (clf, name) in enumerate(CLASSIFIERS.items()):
        x0 = 12.0 + k * 92.0
        ax = axes_mm(sf, x0, 8.0, 76.0, 22.0)
        bump(ax, d[d["classifier"] == clf], "train_set", "feature_set", ylim, "A%d" % k)
        at_mm(sf, x0 + 38.0, 5.6, name, "pn-clf%d" % k, ha="center")
        if k == 0:
            ax.set_ylabel("Mean accuracy")
            box_selected(ax, SELECTED_TN.split("|")[1], "A")
    sel = d[(d["classifier"] == "RandomForest") & (d["train_set"] == "cohort-knn") &
            (d["feature_set"] == SELECTED_TN.split("|")[1])]
    assert len(sel) == 1
    out = d.assign(feature_set=d["feature_set"].map(label), train_set=d["train_set"].map(lambda s: SMOOTH[s][0]),
                   classifier=d["classifier"].map(CLASSIFIERS))
    source("A", out[["classifier", "train_set", "feature_set", "mean_acc", "std_acc"]])
    return float(sel["mean_acc"].iloc[0])


def panel_mgus_mm(sf, C):
    d = C[C["tn_feature"] == SELECTED_TN].copy()
    d["ds_train_set"], d["ds_feature_set"] = zip(*d["ds_feature"].str.split("|"))
    d = d[d["ds_feature_set"].isin(CODES)].copy()
    assert len(d) == 32 and set(d["ds_train_set"]) == {"cohort-knn", "sample-knn"}
    at_mm(sf, 5.0, 1.6, "MGUS-MM classification", "pn-title", size=TITLE)
    smoothing_key(sf, 12.0, 6.0, ["cohort-knn", "sample-knn"], "pn-key")
    lo = (d["mean_acc"] - d["std_acc"]).min()
    hi = (d["mean_acc"] + d["std_acc"]).max()
    ax = axes_mm(sf, 12.0, 13.0, 76.0, 22.0)
    bump(ax, d, "ds_train_set", "ds_feature_set", (lo - 0.005, hi + 0.005), "B")
    ax.set_ylabel("Mean accuracy")
    at_mm(sf, 50.0, 10.6, "Random forest", "pn-clf", ha="center")
    box_selected(ax, SELECTED_DS.split("|")[1], "B")
    sel = d[d["ds_feature"] == SELECTED_DS]
    assert len(sel) == 1
    # class-specific accuracy of every setting (bottom)
    at_mm(sf, 5.0, 61.0, "Accuracy by class, random forest", "pn-title2")
    ax2 = axes_mm(sf, 12.0, 64.5, 38.0, 38.0)
    d["norm"] = d["ds_feature_set"].str.split(":").str[0].str.upper()
    d["combo"] = d["ds_feature_set"].map(lambda c: label(c).split("(")[1].rstrip(")"))
    for _, rw in d.iterrows():
        ax2.plot(rw["MM_mean_acc"], rw["MGUS_mean_acc"], ls="none", marker=MARKERS[rw["combo"]], ms=3.4,
                 mfc=NORM_FILL[rw["norm"]], mec=SMOOTH[rw["ds_train_set"]][1], mew=0.7, zorder=3)
    xlo, xhi = d["MM_mean_acc"].min() * 0.97, d["MM_mean_acc"].max() * 1.02
    ylo, yhi = d["MGUS_mean_acc"].min() * 0.93, d["MGUS_mean_acc"].max() * 1.08
    ax2.set_xlim(xlo, xhi)
    ax2.set_ylim(ylo, yhi)
    for axis in (ax2.xaxis, ax2.yaxis):
        axis.set_major_locator(plots.nice_locator(4))
        plots.plain_ticks(axis)
    ax2.tick_params(labelsize=SMALL)
    ax2.set_xlabel("MM accuracy")
    ax2.set_ylabel("MGUS accuracy")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    sx, sy = float(sel["MM_mean_acc"].iloc[0]), float(sel["MGUS_mean_acc"].iloc[0])
    dx, dy = (xhi - xlo) * 0.14, (yhi - ylo) * 0.03
    a = FancyArrowPatch((sx + dx, sy + dy), (sx, sy), arrowstyle="-|>,head_length=2.4,head_width=1.2",
                        mutation_scale=1.0, lw=0.6, color="black", shrinkA=0, shrinkB=3.2, zorder=4)
    a.set_gid("arrow-selected")
    ax2.add_patch(a)
    scatter_key(sf, 55.0, 63.0)
    out = d.assign(smoothing=d["ds_train_set"].map(lambda s: SMOOTH[s][0]), feature_set=d["ds_feature_set"].map(label))
    source("B", out[["smoothing", "feature_set", "mean_acc", "std_acc", "MM_mean_acc", "MM_std_acc", "MGUS_mean_acc",
                     "MGUS_std_acc", "normal_mean_acc", "normal_std_acc"]])
    return sel.iloc[0]


def scatter_key(sf, x, y):
    """Three blocks: fill = normalization, shape = feature combination, outline = smoothing."""
    fig = plots.root(sf)
    row = 2.15

    def mark(xm, ym, **kw):
        ln = Line2D([xm / sf.w_mm], [1 - ym / sf.h_mm], ls="none", transform=sf.transSubfigure, clip_on=False, **kw)
        sf.add_artist(ln)

    blocks = [("Normalization (fill)", [(n, dict(marker="s", mfc=NORM_FILL[n], mec="black")) for n in ("AUC", "MEX")]),
              ("Feature combination (shape)", [(c, dict(marker=m, mfc="white", mec="black")) for c, m in MARKERS.items()]),
              ("Smoothing (outline)", [(SMOOTH[s][0], dict(marker="o", mfc="white", mec=SMOOTH[s][1]))
                                       for s in ("cohort-knn", "sample-knn")])]
    yy = y
    for b, (title, items) in enumerate(blocks):
        at_mm(sf, x, yy, title, "pn-sk%d" % b, size=SMALL)
        yy += row
        for k, (lab, kw) in enumerate(items):
            mark(x + 1.2, yy, ms=3.0, mew=0.7, **kw)
            at_mm(sf, x + 3.0, yy, lab, "pn-sk%d-%d" % (b, k), size=SMALL)
            yy += row
        yy += 0.9
    return yy


def panel_knn(sf, K):
    K = K.sort_values("k")
    at_mm(sf, 5.0, 1.6, [("k", "i"), ("-nearest neighbors", "")], "pn-title", size=TITLE)
    ax = axes_mm(sf, 12.0, 5.5, 40.0, 17.5)
    for col, lab, c in KNN:
        m, s = K[col].values, K[col.replace("mean", "std")].values
        band = ax.fill_between(K["k"], m - s, m + s, color=c, alpha=0.15, linewidth=0, zorder=1)
        band.set_gid("band-knn-" + lab)
        ax.plot(K["k"], m, color=c, lw=0.75, marker="o", ms=1.8, mew=0, zorder=3)
    ln = ax.axvline(SELECTED_K, color="#808080", lw=0.5, ls=(0, (3, 2)), zorder=2)
    ln.set_gid("ref-k")
    ax.set_xlim(-3, 103)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 0.5, 1.0])
    plots.plain_ticks(ax.yaxis)
    ax.tick_params(labelsize=SMALL)
    ax.set_xlabel("k", fontstyle="italic", labelpad=1.0)
    ax.set_ylabel("Mean accuracy")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    a = FancyArrowPatch((SELECTED_K, -0.20), (SELECTED_K, -0.03), transform=tr,
                        arrowstyle="-|>,head_length=2.2,head_width=1.1", mutation_scale=1.0, lw=0.6, color="black",
                        shrinkA=0, shrinkB=0, clip_on=False, zorder=4)
    a.set_gid("arrow-k")
    ax.add_patch(a)
    plots.key(sf, 57.0 / sf.w_mm, 1 - 7.0 / sf.h_mm, [("%s accuracy" % lab, c) for _, lab, c in KNN],
              transform=sf.transSubfigure, size=SMALL, ms=3.0, line=True, row_pt=2.1 / PT_MM, gid="pn-key")
    sel = K[K["k"] == SELECTED_K]
    assert len(sel) == 1
    cols = ["k"] + [c for c, _, _ in KNN] + [c.replace("mean", "std") for c, _, _ in KNN]
    source("C", K[cols])
    return sel.iloc[0]


def panel_table(sf):
    at_mm(sf, 5.0, 1.6, "Selected parameters", "pn-title", size=TITLE)
    rows = [("Parameter", "Selected"),
            ("Malignancy classifier features", label(SELECTED_TN.split("|")[1])),
            ("MGUS-MM classifier features", label(SELECTED_DS.split("|")[1])),
            ("Smoothing", [("Cohort-level kNN, ", ""), ("k", "i"), (" = %d" % SELECTED_K, "")])]
    x0, x1, xm = 5.0, 85.0, 45.0
    y0, rh = 4.6, 3.3
    ax = axes_mm(sf, 0, 0, sf.w_mm, sf.h_mm)
    ax.set_xlim(0, sf.w_mm)
    ax.set_ylim(sf.h_mm, 0)
    ax.axis("off")
    head = Rectangle((x0, y0), x1 - x0, rh, facecolor="#E6E6E6", edgecolor="none", zorder=0)
    head.set_gid("bg-header")
    ax.add_patch(head)
    for yl, w in ((y0, 0.6), (y0 + rh, 0.4), (y0 + rh * len(rows), 0.6)):
        ax.plot([x0, x1], [yl, yl], color="black", lw=w, zorder=1)[0].set_gid("rule")
    for i, (a, b) in enumerate(rows):
        yc = y0 + rh * (i + 0.5)
        for j, (xx, t) in enumerate(((x0 + 1.5, a), (xm, b))):
            runs = t if isinstance(t, list) else [(t, "b" if i == 0 else "")]
            rich_text(ax, xx, yc, runs, size=SMALL, ha="left", va="center", transform=ax.transData, gid="cell-t%d%d" % (i, j))
    flat = lambda t: "".join(r[0] for r in t) if isinstance(t, list) else t
    source("D", pd.DataFrame([[flat(a), flat(b)] for a, b in rows[1:]], columns=list(rows[0])))


def panel_ablation(sf):
    R = pd.read_csv(paths.result("classifiers", "feature_ablation_mst.tsv"), sep="\t")
    full = R.iloc[0]
    assert full["model"].startswith("all 21 features") and round(full["oof_AUROC"], 2) == 0.86
    R = R.iloc[1:].copy()
    grp = R["model"].str.startswith("without the")
    assert int((~grp).sum()) == 21 and int(grp.sum()) == 1

    def name(m):
        if m.startswith("without the"):
            return "%s lowest-SHAP features\ntogether" % m.split()[2]
        f = m[len("without "):]
        return f.replace(" arm", " CNV-like score") if f.endswith(" arm") else names.feature(f)

    R["label"] = R["model"].map(name)
    R = pd.concat([R[~grp].sort_values("oof_difference", kind="stable"), R[grp]])
    yy = np.r_[np.arange(int((~grp).sum())), (~grp).sum() + 0.9]
    at_mm(sf, 5.0, 1.6, "Feature ablation, MGUS-MM classifier", "pn-title", size=TITLE)
    ax = axes_mm(sf, 42.0, 6.0, 40.0, 46.0)
    ax.errorbar(R["oof_difference"], yy, xerr=[R["oof_difference"] - R["oof_diff_lo"], R["oof_diff_hi"] - R["oof_difference"]],
                fmt="o", color="black", ms=2.0, lw=0.6, capsize=0, zorder=3)
    ax.axvline(0, color="#808080", lw=0.5, ls=(0, (3, 2)), zorder=1).set_gid("ref-zero")
    ax.set_yticks(yy)
    ax.set_yticklabels(R["label"], fontsize=SMALL, linespacing=0.95)
    ax.set_ylim(yy[-1] + 0.9, -0.7)
    ax.set_xlim(-0.13, 0.08)
    ax.set_xticks([-0.1, -0.05, 0, 0.05])
    plots.plain_ticks(ax.xaxis)
    ax.tick_params(axis="x", labelsize=SMALL)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("Change in AUROC (95% CI)\n12 patients with MGUS and 27 with MM", fontsize=SMALL, labelpad=1.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    at_mm(sf, 41.0, 4.6, "Feature removed", "pn-head", size=SMALL, ha="right")
    out = R[["label", "oof_AUROC", "oof_difference", "oof_diff_lo", "oof_diff_hi"]].copy()
    out["label"] = out["label"].str.replace("\n", " ")
    source("E", out.rename(columns={"label": "feature removed"}))


def main():
    T = pd.read_csv(paths.cache("tuning_tumor_normal.tsv"), sep="\t")
    C = pd.read_csv(paths.cache("tuning_mgus_mm.tsv"), sep="\t")
    K = pd.read_csv(paths.cache("tuning_knn.tsv"), sep="\t")
    with style():
        fig, pn = new_figure(LAYOUT)
        acc_tn = panel_malignancy(pn["tuning_malignancy"], T)
        sel_ds = panel_mgus_mm(pn["tuning_mgus_mm"], C)
        sel_k = panel_knn(pn["tuning_knn"], K)
        panel_table(pn["final_parameters"])
        panel_ablation(pn["feature_ablation"])
        # the selected MGUS-MM setting
        assert abs(sel_ds["mean_acc"] - 0.857631) < 1e-6 and abs(sel_ds["MGUS_mean_acc"] - 0.684614) < 1e-6
        print("selected: malignancy %.4f, MGUS-MM %.4f, k = 30 %.4f" % (acc_tn, sel_ds["mean_acc"], sel_k["mean_acc"]))
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig12", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
