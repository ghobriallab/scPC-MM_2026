#!/usr/bin/env python3
"""Supplementary Figure 9: the circulatory capacity signature.

Purpose:      A, the two SMM patients sampled at two IMWG 2/20/20 risk levels (Supplementary Fig. 14C; risk.py): signature activity of every PC on their UMAP, their PC
              status, and per sample (BM PCs and malignant PB PCs); each comparison of malignant PCs of BM and PB at the
              same risk level is labeled with the AUROC of the two-sided Mann-Whitney test between cells, U / (n1 n2),
              the probability that a malignant PB PC has the higher activity (exact P in the source data). B, CoMMpass:
              % PCs in blood (flow cytometry; 0 = not detected) by tertile of the signature in BM; Benjamini-Hochberg q
              of two-sided Wilcoxon rank-sum tests and Spearman's rho with its bootstrap 95% CI. C, CoMMpass: signature in
              CD138+ PB and BM samples of the same visit; two-sided Wilcoxon signed-rank P. D, our cohort: mean activity
              per individual, disease stage, tissue and PC status (figlib/stage_tests.py units); PB versus BM for malignant and for normal PCs, and
              malignant versus normal PCs within PB; two-sided Wilcoxon signed-rank P.
              Panels B and C use CoMMpass patient-level data (controlled access): no source data are written for them.
              Runs in the analysis environment (requirements.txt); the PDF is made from the SVG with Inkscape on a
              machine with Arial (inkscape SuppFig9.svg --export-type=pdf).

Inputs:       results/figure_inputs: longitudinal_cells.npz, longitudinal_signatures.npz,
              per-sample-mean-expr-score.csv.gz; results/commpass: circulatory_signature_patients.tsv,
              circulatory_signature_pairs.tsv, circulatory_signature_tests.tsv, circulatory_paired_samples.tsv.

Outputs:      figures/SuppFig9.svg; figures/source_data/SuppFig9_A_tests.tsv, SuppFig9_D.tsv.

Dependencies: Python 3.8 with requirements.txt (matplotlib 3.5); figlib; Inkscape for the PDF.
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import os
import sys

import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from scipy import stats

import palettes as P
import paths
import stage_tests as ST
import plots
from checks import figure_checks, report
from figstyle import style, rich_text, p_runs, rho_runs, join_runs, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("SuppFig9", [
    Slot("longitudinal", "A", 0, 0, 112, 98),
    Slot("commpass_blood", "B", 115, 0, 68, 64),
    Slot("commpass_pairs", "C", 115, 66, 68, 50),
    Slot("cohort_pairs", "D", 0, 100, 112, 50),
], max_height=247)
SIG = "mean_expr:circulatory_capacity"
PATIENTS = ["Pt-28", "Pt-160"]                     # the two SMM patients sampled at two IMWG 2/20/20 risk levels
GROUP_ORDER = ["Pt-28:BM:IRSMM:normal", "Pt-28:BM:LRSMM:tumor", "Pt-28:BM:IRSMM:tumor", "Pt-160:BM:LRSMM:normal",
               "Pt-160:BM:IRSMM:normal", "Pt-160:BM:LRSMM:tumor", "Pt-160:BM:IRSMM:tumor",
               "Pt-28:PB:IRSMM:tumor", "Pt-160:PB:LRSMM:tumor", "Pt-160:PB:IRSMM:tumor"]
# pairs of the figure, with P and U of the original test
TESTED_PAIRS = {("Pt-28:BM:IRSMM:tumor", "Pt-28:PB:IRSMM:tumor"): (3.085e-18, 6.739e5),
             ("Pt-160:BM:IRSMM:tumor", "Pt-160:PB:IRSMM:tumor"): (3.150e-40, 7.490e4),
             ("Pt-160:BM:LRSMM:tumor", "Pt-160:PB:LRSMM:tumor"): (1.981e-64, 2.075e4)}
TISSUE_BG = {"BM": "#E5E5E5", "PB": "#FBE1CF"}
TERTILES = ["Low", "Medium", "High"]
PCS = [("BM", "normal"), ("PB", "normal"), ("BM", "tumor"), ("PB", "tumor")]
PC_LABEL = {"normal": "normal", "tumor": "malignant"}
BRACKET_TISSUE = "#1A1A1A"


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig9_%s.tsv" % key, sep="\t", index=False)


def close(a, b, rel=1e-6):
    assert abs(a - b) <= rel * max(1e-300, abs(b)), (a, b)


# ---- A ---------------------------------------------------------------------------------------------------------------
def longitudinal():
    c = np.load(paths.cache("longitudinal_cells.npz"))
    sg = np.load(paths.cache("longitudinal_signatures.npz"))
    cat = lambda k: c[k + "__cats"].astype(str)[c[k + "__codes"]]
    d = pd.DataFrame({"x": c["umap"][:, 0], "y": c["umap"][:, 1], "patient": cat("patient_id"), "tissue": cat("sample_type"),
                      "annotation": cat("annotation"), "risk": cat("disease_status")})
    d["value"] = sg["scores"][:, [str(s) for s in sg["names"]].index(SIG)].astype(float)
    d["risk"] = ST.imwg_stages(d["patient"], d["tissue"], d["risk"])      # IMWG 2/20/20 risk groups
    d = d[d["patient"].isin(PATIENTS)].reset_index(drop=True)
    d["group"] = d["patient"] + ":" + d["tissue"] + ":" + d["risk"] + ":" + d["annotation"]
    assert d["value"].notna().all()
    return d


def panel_longitudinal(sf, d):
    xy = d[["x", "y"]].values
    # UMAPs: signature activity (scanpy: high values on top, color range = data range) and PC status
    ax = axes_mm(sf, 6.0, 5.0, 30.0, 30.0)
    norm = Normalize(d["value"].min(), d["value"].max())
    plots.umap(ax, xy, plots.matplotlib_cm("RdPu")(norm(d["value"].values)), "sf9a-sig", s=0.35,
               order=np.argsort(d["value"].values, kind="stable"))
    ax.set_title("Circulatory capacity", fontsize=BASE, pad=2.0)
    cax = axes_mm(sf, 37.5, 12.0, 1.4, 17.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap="RdPu"), cax)
    cb.ax.yaxis.set_major_locator(plots.nice_locator(3))
    plots.plain_ticks(cb.ax.yaxis)
    rich_text(cax, 0.0, 1, [("Activity", "")], size=SMALL, ha="left", va="bottom",
              transform=plots.pts(cax, 0, 3.8, cax.transAxes), gid="pn-cbt")
    a2 = axes_mm(sf, 53.0, 5.0, 30.0, 30.0)
    col = np.array([P.PC_STATUS["%s %s" % (t, "malignant" if a == "tumor" else "normal")]
                    for t, a in zip(d["tissue"], d["annotation"])])
    plots.umap(a2, xy, col, "sf9a-status", s=0.35)
    a2.set_title("Plasma cell status", fontsize=BASE, pad=2.0)
    plots.key(sf, 85.0 / sf.w_mm, 1 - 12.0 / sf.h_mm,
              [("%s %s PCs" % (t, PC_LABEL[a]), P.PC_STATUS["%s %s" % (t, "malignant" if a == "tumor" else "normal")])
               for t, a in PCS], transform=sf.transSubfigure, size=SMALL, ms=2.4, row_pt=2.2 * 72 / 25.4, gid="pn-pckey")
    # violins: BM PCs and malignant PB PCs of each patient, by sample
    shown = d[(d["tissue"] == "BM") | (d["annotation"] == "tumor")]
    groups = {p: [g for g in GROUP_ORDER if g.startswith(p + ":") and (shown["group"] == g).any()] for p in PATIENTS}
    widths = {p: 5.6 * len(g) for p, g in groups.items()}
    xs, x = {}, 13.0
    for p in PATIENTS:
        xs[p] = x
        x += widths[p] + 2.5
    tests, axes, jobs = [], [], []
    for j, p in enumerate(PATIENTS):
        df = shown[shown["patient"] == p]
        gs = groups[p]
        ax = axes_mm(sf, xs[p], 50.0, widths[p], 18.0)
        first_pb = [k for k, g in enumerate(gs) if ":PB:" in g][0]
        for t, (a, b) in (("BM", (-0.5, first_pb - 0.5)), ("PB", (first_pb - 0.5, len(gs) - 0.5))):
            ax.axvspan(a, b, color=TISSUE_BG[t], lw=0, zorder=0).set_gid("bg-%s" % t)
        for k, g in enumerate(gs):
            risk, ann = g.split(":")[2], g.split(":")[3]
            plots.violin_box(ax, df.loc[df["group"] == g, "value"].values, k,
                             (P.TUMOR if ann == "tumor" else P.NORMAL)[risk], "sf9a-%s" % g, width=0.82)
        comps = []
        for risk in df["risk"].unique():                          # pairs: malignant PCs of BM and PB, same risk group
            pair = [g for g in gs if g.endswith(":tumor") and (":%s:" % risk) in g]
            if len(pair) != 2:
                continue
            a, b = pair
            l10, u = plots.mwu_log10p(df.loc[df["group"] == a, "value"].values, df.loc[df["group"] == b, "value"].values)
            ref_p, ref_u = TESTED_PAIRS[(a, b)]
            close(u, ref_u, 6e-4)
            close(10 ** l10, ref_p, 1e-3)
            n_a, n_b = int((df["group"] == a).sum()), int((df["group"] == b).sum())
            auc = 1.0 - u / (n_a * n_b)                         # P(a malignant PB PC is higher than a BM one)
            tests.append(dict(first=a, second=b, n_first=n_a, n_second=n_b, U=u, auroc=auc, log10_p=l10))
            comps.append(dict(i=gs.index(a), j=gs.index(b), runs=[("AUROC %.2f" % auc, "")], color=BRACKET_TISSUE))
        ax.set_xlim(-0.5, len(gs) - 0.5)
        jobs.append((ax, comps, {k: float(df.loc[df["group"] == g, "value"].max()) for k, g in enumerate(gs)}, "sf9a%d" % j))
        axes.append(ax)
        ax.set_yticks([0, 0.5, 1.0])
        plots.plain_ticks(ax.yaxis)
        ax.tick_params(axis="y", labelsize=SMALL)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Circulatory capacity\nactivity", fontsize=SMALL)
        ax.set_xticks(range(len(gs)))
        ax.set_xticklabels(["%s %s" % (g.split(":")[2], PC_LABEL[g.split(":")[3]]) for g in gs], rotation=90, fontsize=SMALL)
        plots.n_row(ax, [int((df["group"] == g).sum()) for g in gs], rotate=True)
        at_mm(sf, xs[p] + widths[p] / 2.0, 46.8, p, "pn-pt%d" % j, ha="center")
    plots.brackets_shared(jobs, -0.03)                          # one y scale, brackets placed at that scale
    assert len(tests) == len(TESTED_PAIRS)
    for k, t in enumerate(("BM", "PB")):
        rect = Rectangle(((1.0 + 7.0 * k) / sf.w_mm, 1 - 47.9 / sf.h_mm), 2.2 / sf.w_mm, 2.2 / sf.h_mm,
                         transform=sf.transSubfigure, facecolor=TISSUE_BG[t], edgecolor="#333333", lw=0.3)
        rect.set_gid("key-%s" % t)
        sf.add_artist(rect)
        at_mm(sf, 1.0 + 7.0 * k + 3.0, 46.8, t, "pn-tk%d" % k, size=SMALL)
    source("A_tests", pd.DataFrame(tests))


# ---- B ---------------------------------------------------------------------------------------------------------------
def panel_blood(sf, tests):
    d = pd.read_csv(paths.result("commpass", "circulatory_signature_patients.tsv"), sep="\t")
    sig = [c for c in d.columns if c.endswith("circulatory_capacity")][0]
    assert len(d) == 633
    # the tests are those of circulatory_signature_tests.tsv; recomputed here, they agree to the printed digits
    # (blood PC percentages are means of a low and a high value: equal percentages can differ in the last bit, and the
    # text table shifts some of those bits, which reorders a few ties)
    t = tests[tests["analysis"] == "CoMMpass % PCs in blood by tertile"].set_index("comparison")
    for a, b in (("Low", "Medium"), ("Medium", "High"), ("Low", "High")):
        p = stats.mannwhitneyu(d.loc[d["tertile"] == a, "pb_pct"], d.loc[d["tertile"] == b, "pb_pct"],
                               alternative="two-sided").pvalue
        assert p_runs(p) == p_runs(float(t.loc["%s vs %s" % (a, b), "p"])), (a, b, p)
    sp = tests[(tests["analysis"] == "CoMMpass") & (tests["comparison"] == "signature vs % PCs in blood")].iloc[0]
    assert round(stats.spearmanr(d[sig], d["pb_pct"]).correlation, 2) == round(float(sp["value"]), 2)
    at_mm(sf, 34.0, 1.8, [("CoMMpass (", ""), ("n", "i"), (" = %d)" % len(d), "")], "pn-title", ha="center")
    at_mm(sf, 34.0, 5.0, join_runs(rho_runs(float(sp["value"])) + [(" (95%% CI, %.2f–%.2f)" % (sp["ci_low"], sp["ci_high"]),
                                                                       "")], p_runs(float(sp["p"]))),
          "pn-rho", size=SMALL, ha="center")
    ab = axes_mm(sf, 14.0, 7.0, 44.0, 13.5)                      # brackets above the box plot (its y axis is symlog)
    ab.set_xlim(-0.6, 2.4)
    ab.set_ylim(0, 1)
    ab.axis("off")
    ax = axes_mm(sf, 14.0, 21.0, 44.0, 29.0, sharex=ab)
    plots.box_strip(ax, d.rename(columns={"tertile": "group", "pb_pct": "value"}), TERTILES, P.LOW_MED_HIGH, "sf9b",
                    point=1.0, width=0.6, edge=0.15, jitter=0.25, labels=TERTILES, label_size=BASE)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
    ax.set_ylim(-0.003, 60)
    ax.set_yticks([0, 0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0", "0.01", "0.1", "1", "10"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", labelsize=SMALL)
    ax.set_ylabel("PCs in blood (%, flow cytometry)")
    ax.set_xlabel("Circulatory capacity activity in BM (tertile)", labelpad=8.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plots.brackets(ab, [dict(i=0, j=1, runs=p_runs(float(t.loc["Low vs Medium", "q"]), "q"), color=BRACKET_TISSUE),
                        dict(i=1, j=2, runs=p_runs(float(t.loc["Medium vs High", "q"]), "q"), color=BRACKET_TISSUE),
                        dict(i=0, j=2, runs=p_runs(float(t.loc["Low vs High", "q"]), "q"), color=BRACKET_TISSUE)],
                   {0: 0.0, 1: 0.0, 2: 0.0}, ymin=-1.0, key="sf9b")


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_pairs(sf, tests):
    pr = pd.read_csv(paths.result("commpass", "circulatory_signature_pairs.tsv"), sep="\t")
    assert len(pr) == 12
    w = tests[tests["analysis"] == "CoMMpass CD138+ PB vs BM, same visit"].iloc[0]
    p = stats.wilcoxon(pr["PB"], pr["BM"]).pvalue
    close(p, float(w["p"]))
    at_mm(sf, 34.0, 1.8, "CoMMpass, CD138+ samples of the same visit", "pn-title", ha="center")
    ax = axes_mm(sf, 18.0, 6.0, 30.0, 33.0)
    for _, r in pr.iterrows():
        ax.plot([0, 1], [r["BM"], r["PB"]], color="#999999", lw=0.5, zorder=1)[0].set_gid("pair")
    for i, t_ in enumerate(("BM", "PB")):
        ax.scatter(np.full(len(pr), i), pr[t_], s=2.4 ** 2, c=P.TISSUE[t_], edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(10, 600)
    ax.set_yticks([10, 20, 50, 100, 200, 500])
    ax.set_yticklabels(["10", "20", "50", "100", "200", "500"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", labelsize=SMALL)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BM", "PB"])
    plots.n_below(ax, [len(pr)] * 2)
    ax.set_ylabel("Circulatory capacity activity\n(mean TPM)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    rich_text(ax, 0.5, 1, p_runs(p), size=SMALL, ha="center", va="top", transform=plots.pts(ax, 0, -2.5, ax.transAxes),
              gid="ax-p")                                 # va top refers to the main run: room for the exponent


# ---- D ---------------------------------------------------------------------------------------------------------------
def panel_cohort(sf):
    u = ST.signature_units().rename(columns={"disease_status": "stage"})   # IMWG risk groups, units pooled by cells
    u = u.dropna(subset=[SIG])
    ref = pd.read_csv(paths.result("commpass", "circulatory_paired_samples.tsv"), sep="\t").set_index("comparison")
    panels = []
    for cells, title, comp in (("tumor", "Malignant PCs", "PB vs BM, malignant PCs"),
                               ("normal", "Normal PCs", "PB vs BM, normal PCs")):
        x = u[u["annotation"] == cells].pivot_table(index=["patient_id", "stage"], columns="sample_type", values=SIG).dropna()
        panels.append((title, comp, ("BM", cells), ("PB", cells), x.rename(columns={"BM": "left", "PB": "right"}),
                       ["BM", "PB"]))
    x = u[u["sample_type"] == "PB"].pivot_table(index=["patient_id", "stage"], columns="annotation", values=SIG).dropna()
    panels.append(("Within PB", "PB malignant vs PB normal PCs", ("PB", "normal"), ("PB", "tumor"),
                   x.rename(columns={"normal": "left", "tumor": "right"}), ["Normal", "CTCs"]))
    rows, axes, jobs = [], [], []
    for k, (title, comp, kl, kr, x, labels) in enumerate(panels):
        p = stats.wilcoxon(x["right"], x["left"]).pvalue
        assert len(x) == ref.loc[comp, "n_pairs"] and (x["right"] > x["left"]).sum() == ref.loc[comp, "first_higher"]
        close(p, float(ref.loc[comp, "p"]))
        ax = axes_mm(sf, 14.0 + 32.0 * k, 6.0, 22.0, 30.0)
        for _, r in x.iterrows():
            ax.plot([0, 1], [r["left"], r["right"]], color="#B3B3B3", lw=0.35, zorder=1)[0].set_gid("pair")
        for i, (key_, kk) in enumerate((("left", kl), ("right", kr))):
            ax.scatter(np.full(len(x), i), x[key_], s=1.5 ** 2, c=P.PC_STATUS["%s %s" % (kk[0], "malignant" if kk[1] == "tumor"
                                                                                        else "normal")],
                       edgecolors="black", linewidths=0.25, zorder=3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        plots.n_below(ax, [len(x)] * 2)
        col = P.BRACKET_MAL if title == "Within PB" else BRACKET_TISSUE       # malignant vs normal PCs: red, as elsewhere
        jobs.append((ax, [dict(i=0, j=1, p=p, color=col)], {0: float(x["left"].max()), 1: float(x["right"].max())},
                     "sf9d%d" % k))
        ax.set_title(title, fontsize=BASE, pad=2.0)
        axes.append(ax)
        rows += [dict(comparison=comp, patient_id=i_[0], stage=i_[1], first=r["left"], second=r["right"])
                 for i_, r in x.iterrows()]
    plots.brackets_shared(jobs, 0)                              # one y scale, brackets placed at that scale
    for k, a in enumerate(axes):
        a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        plots.plain_ticks(a.yaxis)
        a.tick_params(axis="y", labelsize=SMALL)
        if k:
            a.set_yticklabels([])
        else:
            a.set_ylabel("Circulatory capacity activity\n(mean log-normalized expression)", fontsize=SMALL)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    at_mm(sf, 56.0, 45.5, "Paired samples of the same individual and disease stage, this cohort", "pn-foot", size=SMALL,
          ha="center")
    source("D", pd.DataFrame(rows))


def main():
    tests = pd.read_csv(paths.result("commpass", "circulatory_signature_tests.tsv"), sep="\t")
    d = longitudinal()
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_longitudinal(pn["longitudinal"], d)
        panel_blood(pn["commpass_blood"], tests)
        panel_pairs(pn["commpass_pairs"], tests)
        panel_cohort(pn["cohort_pairs"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT, preview_dpi=0)
        os.remove(pdf)                                   # the PDF is made from the SVG where Arial is installed
        print(report("SuppFig9", issues, "file checks run once the PDF is made from the SVG"))
        print("\n".join(issues) or "ALL CHECKS PASS")


if __name__ == "__main__":
    sys.exit(main())
