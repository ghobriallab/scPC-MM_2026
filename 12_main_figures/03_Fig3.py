#!/usr/bin/env python3
"""Figure 3: sixteen transcriptional signatures dissecting tumorigenesis and disease progression.

Purpose:      A, Hotspot local-correlation Z-scores of the 1,463 signature genes with signature membership. B, genes per
              signature. C, MHC-I gene module, histone/chaperone and B cell activation: the 15 genes with the highest
              Hotspot Z-score and the mean activity per patient by disease stage, tissue and PC status, with two-sided
              Wilcoxon rank-sum tests and Benjamini-Hochberg q (figlib/stage_tests.py). D, E, CoMMpass progression-free
              survival by thirds of histone/chaperone and B cell activation (cut-offs at the 33rd and 66th percentiles,
              log-rank P). F, multivariable Cox model of progression-free survival.

Inputs:       results/figure_inputs: hotspot_z.npz, hotspot_genes.tsv, pcs_cells.npz, per-sample-mean-expr-score.csv.gz;
              results/stage_and_progression/
              stage_comparisons.tsv; results/commpass/signature_cox_age_sex.tsv; data/commpass/
              MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv.

Outputs:      figures/Fig3.svg, Fig3.pdf; figures/source_data/Fig3_*.tsv (none for CoMMpass patient-level data).

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

import names
import palettes as P
import paths
import plots
import stage_tests as ST
from checks import figure_checks, file_checks, report
from figstyle import (style, rich_text, rich_width_pt, p_runs, n_runs, join_runs, SMALL, BASE, LW,
                      MINUS)
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("Fig3", [
    Slot("hotspot_matrix", "A", 0, 0, 56, 56),
    Slot("genes_per_signature", "B", 57, 0, 36, 56),
    Slot("signature_rows", "C", 95, 0, 88, 118),
    Slot("km_histone", "D", 0, 58, 45, 47),
    Slot("km_bcell", "E", 47, 58, 45, 47),
    Slot("cox_signatures", "F", 0, 107, 93, 48),
], max_height=170)
PT_MM = 25.4 / 72.0
INK = "#333333"
THREE = ["MHC-I_gene_module", "histone_chaperone_gene", "B_cell_activation"]      # panel C, in this order


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/Fig3_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center", **kw):
    """Text at (x, y) mm from the panel's top-left corner."""
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs, size=size, ha=ha, va=va, transform=sf.transSubfigure,
                     gid=gid, **kw)


def close(a, b, rel=1e-6):
    assert abs(a - b) <= rel * max(1.0, abs(b)), (a, b)


def signature_genes():
    """Hotspot genes with their signature; a gene in two signatures is colored as circulatory capacity."""
    g = pd.read_csv(paths.cache("hotspot_genes.tsv"), sep="\t")
    ctx = g.groupby("gene")["context"].agg(lambda s: "circulatory_capacity" if len(s) > 1 else s.iloc[0])
    return g, ctx


# ---- A ---------------------------------------------------------------------------------------------------------------
def panel_hotspot_matrix(sf):
    z = np.load(paths.cache("hotspot_z.npz"), allow_pickle=True)
    g, ctx = signature_genes()
    rows = np.array([str(x) for x in z["rows"]])
    keep = np.isin(rows, ctx.index)
    assert keep.sum() == 1463 == len(ctx)
    Z = z["z"][np.ix_(keep, keep)]
    genes = rows[keep]
    x0, y0, w = 8.0, 4.5, 44.0
    at_mm(sf, x0 + w / 2, 2.2, [("%s genes" % format(len(genes), ","), "")], "pn-title", ha="center")
    ax = axes_mm(sf, x0, y0, w, w)
    im = plots.heatmap(ax, Z, "fig3-hotspot", P.CMAP_DIVERGING, vmin=-200, vmax=200, raster=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(LW)
    # signature membership: one colored segment per run of genes of the same signature (vector)
    st = axes_mm(sf, x0 + w + 0.4, y0, 2.0, w)
    lab = ctx.reindex(genes).values
    start = 0
    for k in range(1, len(lab) + 1):
        if k == len(lab) or lab[k] != lab[start]:
            r = Rectangle((0, start), 1, k - start, fc=P.SIGNATURE[lab[start]], ec="none")
            r.set_gid("strip-" + lab[start])
            st.add_patch(r)
            start = k
    st.set_xlim(0, 1)
    st.set_ylim(len(lab), 0)
    st.axis("off")
    cax = axes_mm(sf, 1.2, 12.0, 1.3, 18.0)
    plots.colorbar(sf, im, cax, ticks=[-200, 0, 200])
    cax.set_yticklabels([MINUS + "200", "0", "200"])
    at_mm(sf, 0.6, 7.3, [("Hotspot", "")], "pn-zlab0", size=SMALL)
    at_mm(sf, 0.6, 9.6, [("Z-score", "")], "pn-zlab", size=SMALL)
    order = list(dict.fromkeys(lab))
    return order


# ---- B ---------------------------------------------------------------------------------------------------------------
def panel_genes_per_signature(sf, order):
    g, _ = signature_genes()
    counts = g["context"].value_counts().reindex(order)
    assert counts.sum() == 1467 and sorted(order) == sorted(names.SIGNATURE_ORDER)     # bar order = matrix order
    ax = axes_mm(sf, 24.6, 4.5, 7.4, 44.0)
    bars = plots.hbar(ax, [names.SIGNATURES[k] for k in order], counts.values, [P.SIGNATURE[k] for k in order],
                      zero=False)
    for b in bars:
        b.set_edgecolor(INK)
        b.set_linewidth(0.3)
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 100, 200])
    ax.tick_params(axis="y", labelsize=SMALL, length=0, pad=1.0)
    ax.tick_params(axis="x", labelsize=SMALL)
    ax.set_xlabel("Genes", labelpad=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for y, v in enumerate(counts.values):
        rich_text(ax, v, y, [("%d" % v, "")], size=SMALL, ha="left", va="center", transform=plots.pts(ax, 1.0, 0),
                  gid="pn-count%d" % y)
    source("genes_per_signature", pd.DataFrame({"signature": [names.SIGNATURES[k] for k in order],
                                                "genes": counts.values}))


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_signature_rows(sf):
    g, _ = signature_genes()
    units = ST.signature_units()
    rows_out, top_out, stage_axes = [], [], []
    y_rows = [0.4, 37.8, 75.2]
    for r, sig in enumerate(THREE):
        y = y_rows[r]
        col = "mean_expr:" + sig
        at_mm(sf, 4.5 if r == 0 else 0.5, y + 1.5, [(names.SIGNATURES[sig], "")], "pn-row%d" % r)
        # the 15 genes with the highest Hotspot Z-score
        top = g[g["context"] == sig].sort_values("Z", ascending=False).head(15)
        ab = axes_mm(sf, 13.0, y + 3.4, 13.0, 30.6)
        bars = plots.hbar(ab, list(top["gene"]), top["Z"].values, P.SIGNATURE[sig], zero=False)
        for b in bars:
            b.set_edgecolor(INK)
            b.set_linewidth(0.3)
        ab.tick_params(axis="y", labelsize=SMALL, length=0, pad=1.0)
        ab.tick_params(axis="x", labelsize=SMALL)
        ab.xaxis.set_major_locator(plots.nice_locator(2))
        for sp in ("top", "right"):
            ab.spines[sp].set_visible(False)
        if r == 2:
            ab.set_xlabel("Hotspot Z-score", labelpad=1.0)
        top_out.append(top.assign(signature=names.SIGNATURES[sig])[["signature", "gene", "Z"]])
        # mean activity per patient by disease stage, tissue and PC status
        t = ST.tests(units, col)
        ST.check_against_saved("signature " + sig, t)
        specs = ST.brackets(units, col, t, 0.05, "signature")
        ax = axes_mm(sf, 36.0, y + 5.0, 51.5, 25.0 if r == 2 else 27.0)
        lo_, hi_ = units[col].min(), units[col].max()
        ax.set_ylim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.05 * (hi_ - lo_))
        plots.stage_box(ax, units, col, specs, "fig3-c%d" % r, ylabel="Mean activity", point=1.2, headers=(r == 0),
                        labels=(r == 2))
        ax.yaxis.set_major_locator(plots.nice_locator(3))
        ax.tick_params(axis="y", labelsize=SMALL)
        stage_axes.append(ax)
        rows_out.append(units[["patient_id", "disease_status", "sample_type", "annotation", col]]
                        .rename(columns={col: "mean_activity"}).assign(signature=names.SIGNATURES[sig]))
        source("signature_rows_brackets_%d" % r, pd.DataFrame([dict(signature=names.SIGNATURES[sig], block=b["block"],
                                                                    kind=b["kind"], q=b["q"], comparisons=b["n"])
                                                               for b in specs]))
    sf.align_ylabels(stage_axes)                                  # the three "Mean activity" labels on one line
    source("signature_rows_genes", pd.concat(top_out))
    source("signature_rows_activity", pd.concat(rows_out))


# ---- D, E ------------------------------------------------------------------------------------------------------------
# numbers at risk and log-rank P of the original survminer analysis (surv_pvalue); None = not printed
KM_EXPECT = {"histone_chaperone_gene": ({"Low": [249, 116, 80, 54, 9], "Medium": [248, 125, 63, 35, 5],
                                         "High": [257, 97, 48, 30, 4]}, 9.311140e-03),
             "B_cell_activation": ({"Low": [249, 118, 73, 41, 9], "Medium": [248, 123, 69, 48, 5],
                                    "High": [257, 97, 49, 30, 4]}, 3.119466e-04),
             "cell_cycling_G1S": (None, 3.245262e-07),
             "cell_cycling_G2M": (None, 7.157504e-06)}


def panel_km(sf, sig):
    e = pd.read_csv(paths.inputs("commpass", "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv"), sep="\t")
    v = e["mean_expr:" + sig]
    q33, q66 = np.quantile(v, [0.33, 0.66])                     # R quantile(type 7), probs 0.33 and 0.66
    k = pd.DataFrame({"group": np.where(v > q66, "High", np.where(v > q33, "Medium", "Low")),
                      "time": e["ttcpfs1"] / 365.0, "event": e["censpfs1"]})
    grp = ["Low", "Medium", "High"]
    assert len(k) == 754
    at_mm(sf, 4.5, 1.8, [(names.SIGNATURES[sig], "")], "pn-title")
    pv = plots.logrank_p(k)
    exp_risk, exp_p = KM_EXPECT[sig]
    assert abs(pv - exp_p) <= 1e-6 * exp_p, (pv, exp_p)
    at_mm(sf, 43.0, 1.8, [("Log-rank ", "")] + p_runs(pv), "pn-logrank", size=SMALL, ha="right")
    at_mm(sf, 4.5, 4.75, [("Tertile", "")], "pn-tert", size=SMALL)
    plots.key(sf, 13.0 / sf.w_mm, 1 - 3.6 / sf.h_mm, [(g_, P.LOW_MED_HIGH[g_]) for g_ in grp], line=True, size=SMALL,
              ms=3.2, ncol=3, col_pt=rich_width_pt(sf, [("Medium", "")], SMALL) + 13.0, transform=sf.transSubfigure,
              gid="pn-kmkey")
    ak = axes_mm(sf, 13.0, 7.0, 29.5, 20.0)
    tabs = plots.km(ak, k, grp, P.LOW_MED_HIGH)
    ak.set_xlim(0, 8.3)
    ak.set_xticks([0, 2, 4, 6, 8])
    ak.set_yticks([0, 0.5, 1.0])
    ak.set_yticklabels(["0", "0.5", "1"])
    ak.tick_params(labelsize=SMALL)
    ak.set_xlabel("Years", labelpad=1.0)
    ak.set_ylabel("Progression-free\nsurvival")
    risk = {g_: [int(T[T[:, 0] >= t, 1][0]) if np.any(T[:, 0] >= t) else 0 for t in (0, 2, 4, 6, 8)]
            for g_, T in tabs.items()}
    assert exp_risk is None or risk == exp_risk, risk
    at_mm(sf, 13.0, 34.6, [("Patients at risk", "")], "pn-atrisk", size=SMALL)
    ar = axes_mm(sf, 13.0, 36.0, 29.5, 8.0, sharex=ak)
    plots.at_risk(ar, tabs, grp, P.LOW_MED_HIGH, [0, 2, 4, 6, 8])


# ---- F ---------------------------------------------------------------------------------------------------------------
def panel_cox(sf):
    c = pd.read_csv(paths.result("commpass", "signature_cox_age_sex.tsv"), sep="\t")
    c = c[(c["endpoint"] == "PFS") & (c["model"] == "paper")].set_index("term")
    n, ev = int(c["n"].iloc[0]), int(c["events"].iloc[0])
    assert (n, ev) == (735, 304)
    terms = ["B_cell_activation", "cell_cycling_G1S", "cell_cycling_G2M", "interferon_inducible", "histone_chaperone_gene",
             "PI", "ISS"]
    label = dict(names.SIGNATURES, PI="Proliferation index", ISS="ISS stage")
    close(c.loc["B_cell_activation", "HR"], 1.15681734052679)
    rows = [dict(label=label[t], hr=c.loc[t, "HR"], lo=c.loc[t, "lo"], hi=c.loc[t, "hi"], p=c.loc[t, "P"],
                 stat="%.2f (%.2f%s%.2f)" % (c.loc[t, "HR"], c.loc[t, "lo"], "–", c.loc[t, "hi"])) for t in terms]
    at_mm(sf, 4.5, 1.8, [("Progression-free survival, CoMMpass (", ""), ("n", "i"), (" = %d; %d events)" % (n, ev), "")],
          "pn-title")
    at_mm(sf, 4.5, 5.0, [("Multivariable Cox model: each factor is adjusted for all other factors", "")], "pn-sub",
          size=SMALL)
    # forest table as forestmodel draws it: variable, n, hazard ratio per SD on a log axis, the same as text, exact P
    x0, x1 = 0.8, sf.w_mm - 0.8                                  # frame, mm
    xn, xf0, xf1, xs, xp = 28.4, 31.2, 52.6, 55.4, x1 - 1.2       # n (right edge), plot, HR column, P (right edge)
    rh, top = 3.9, 11.6                                          # row height and top of the first row, mm
    ax = axes_mm(sf, xf0, top, xf1 - xf0, rh * len(rows))
    plots.forest(ax, rows, xlim=(0.6, 2.0), xticks=[0.6, 1, 1.5, 2], label="HR per SD (log scale)")
    ax.tick_params(labelsize=SMALL)
    tr = blended_transform_factory(sf.transSubfigure, ax.transData)
    X = lambda mm: mm / sf.w_mm
    for y in range(1, len(rows), 2):                              # alternate rows shaded, behind the text
        band = Rectangle((X(x0), y - 0.5), X(x1) - X(x0), 1.0, transform=tr, facecolor="#F0F0F0", edgecolor="none",
                         zorder=0, clip_on=False)
        band.set_gid("bg-band%d" % y)
        ax.add_patch(band)
    plots.table_rules(ax, sf, x0, x1, -1.5, -0.5, len(rows) - 0.5, seps=((xn + xf0) / 2, (xf1 + xs) / 2))
    w = lambda runs: rich_width_pt(sf, runs, BASE) * PT_MM
    for x_mm, runs, ha in ((x0 + 1.0, [("Variable", "b")], "left"), (xn, [("n", "bi")], "right"),
                           ((xf0 + xf1) / 2, [("Hazard ratio", "b")], "center"), (xs, [("HR (95% CI)", "b")], "left"),
                           (xp, [("P", "bi")], "right")):
        rich_text(ax, X(x_mm), -1.0, runs, size=BASE, ha=ha, va="center", transform=tr, gid="pn-h" + runs[0][0][0])
    for y, rw in enumerate(rows):
        pr = p_runs(rw["p"])[1:]                                  # the column header is P: values only
        pr[0] = (pr[0][0].replace(" = ", ""), pr[0][1])
        nr = [("%d" % n, "")]
        assert x0 + 1.0 + w([(rw["label"], "b")]) + 1.0 <= xn - w(nr), rw["label"]
        assert xs + w([(rw["stat"], "")]) + 1.0 <= xp - w(pr), rw["stat"]
        rich_text(ax, X(x0 + 1.0), y, [(rw["label"], "b")], size=BASE, va="center", transform=tr, gid="pn-v%d" % y)
        rich_text(ax, X(xn), y, nr, size=BASE, ha="right", va="center", transform=tr, gid="pn-n%d" % y)
        rich_text(ax, X(xs), y, [(rw["stat"], "")], size=BASE, va="center", transform=tr, gid="pn-s%d" % y)
        rich_text(ax, X(xp), y, pr, size=BASE, ha="right", va="center", transform=tr, gid="pn-p%d" % y)
    source("cox_signatures", pd.DataFrame([dict(variable=r["label"], n=n, hr=r["hr"], lo=r["lo"], hi=r["hi"], p=r["p"])
                                           for r in rows]))


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        order = panel_hotspot_matrix(pn["hotspot_matrix"])
        panel_genes_per_signature(pn["genes_per_signature"], order)
        panel_signature_rows(pn["signature_rows"])
        panel_km(pn["km_histone"], "histone_chaperone_gene")
        panel_km(pn["km_bcell"], "B_cell_activation")
        panel_cox(pn["cox_signatures"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("Fig3", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
