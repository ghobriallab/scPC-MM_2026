#!/usr/bin/env python3
"""Extended Data Figure 9: multivariable survival models of the MST score in CoMMpass, and regressions between
mutation-based and transcription-based progression scores.

Purpose:      A, B, Cox models of progression-free and overall survival (MST score adjusted for age, sex, R2-ISS and
              IMS/IMWG 2025 risk), exact Wald P. C, MST score against the WGS MM-like score with the least-squares fit
              (points jittered horizontally), residuals by disease stage (Spearman correlation with stage coded 0-4, from
              03_wgs_residuals_by_stage.py), and the classifier features whose Spearman
              correlation with the residuals has q < 0.05 (Benjamini-Hochberg). D, the same for the LASSO model of the
              MST score on 27 genomic alterations: observed against predicted scores (dashed line, predicted =
              observed), residuals and correlated features.

Inputs:       results/commpass: mst_cox_coefficients.tsv, mst_cox_covariates.tsv, mst_cox_models.tsv;
              results/figure_inputs: wgs_samples.tsv, wgs_residual_features_lm.tsv, wgs_residual_features_lasso.tsv,
              wgs_residual_stage.tsv.

Outputs:      figures/EDFig9.svg, EDFig9.pdf; figures/source_data/EDFig9_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

import names
import palettes as P
import paths
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, p_runs, rho_runs, join_runs, n_runs, rng, MINUS, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("EDFig9", [
    Slot("cox_pfs", "A", 0, 0, 90, 52),
    Slot("cox_os", "B", 93, 0, 90, 52),
    Slot("wgs_regression", "C", 0, 54, 183, 47),
    Slot("lasso_regression", "D", 0, 103, 183, 47),
], max_height=247)
MODEL = "age, sex, R2-ISS, IMS/IMWG 2025"
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
# rows of the forest: (variable, level, term in the Cox table or None for the reference level, covariate key for N)
ROWS = [("MST score (per SD)", "", "MST_sd", None), ("Age (per 10 years)", "", "age10", None),
        ("Sex", "Male", None, ("sex", "Male")), ("", "Female", "sexFemale", ("sex", "Female")),
        ("R2-ISS", "I", None, ("R2-ISS", "I")), ("", "II", "r2issII", ("R2-ISS", "II")),
        ("", "III", "r2issIII", ("R2-ISS", "III")), ("", "IV", "r2issIV", ("R2-ISS", "IV")),
        ("IMS/IMWG 2025", "Standard risk", None, ("IMS/IMWG 2025 high risk", "No")),
        ("", "High risk", "imwgYes", ("IMS/IMWG 2025 high risk", "Yes"))]
SHADE = [0, 2, 3, 8, 9]                                   # alternate variables shaded


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/EDFig9_%s.tsv" % key, sep="\t", index=False)


# ---- A, B ------------------------------------------------------------------------------------------------------------
def panel_cox(sf, endpoint, title, xlim, xticks):
    co = pd.read_csv(paths.result("commpass", "mst_cox_coefficients.tsv"), sep="\t")
    co = co[(co["endpoint"] == endpoint) & (co["reference_model"] == MODEL)].set_index("term")
    cov = pd.read_csv(paths.result("commpass", "mst_cox_covariates.tsv"), sep="\t").set_index(["variable", "level"])["n"]
    mo = pd.read_csv(paths.result("commpass", "mst_cox_models.tsv"), sep="\t")
    mo = mo[(mo["endpoint"] == endpoint) & (mo["reference_model"] == MODEL)].iloc[0]
    n, ev = int(mo["n"]), int(mo["events"])
    assert n == 541 and abs(co.loc["MST_sd", "hr"] - mo["adj_hr_per_sd"]) < 1e-12
    at_mm(sf, 4.5, 1.8, [(title + ", CoMMpass (", ""), ("n", "i"), (" = %d; %d events)" % (n, ev), "")], "pn-title")
    at_mm(sf, 4.5, 5.0, "Multivariable Cox model: each factor is adjusted for all other factors", "pn-sub", size=SMALL)
    ax = axes_mm(sf, 42.0, 11.0, 18.0, 30.0)
    rows = []
    for var, lev, term, ck in ROWS:
        nn = n if ck is None else int(cov.loc[ck])
        if term is None:
            rows.append(dict(var=var, lev=lev, n=nn, hr=1.0, lo=None, hi=None, p=None, stat="Reference"))
        else:
            r = co.loc[term]
            rows.append(dict(var=var, lev=lev, n=nn, hr=r["hr"], lo=r["lo"], hi=r["hi"], p=r["p"],
                             stat="%.2f (%.2f–%.2f)" % (r["hr"], r["lo"], r["hi"])))
    assert sum(r["n"] for r in rows if r["var"] == "Sex" or r["lev"] == "Female") == n
    for y, rw in enumerate(rows):
        if rw["lo"] is not None:
            ax.plot([rw["lo"], rw["hi"]], [y, y], color="black", lw=0.6, zorder=2, solid_capstyle="butt")
        ax.plot([rw["hr"]], [y], marker="s", ms=2.4, color="black", ls="none", zorder=3)
    ax.axvline(1.0, color=P.REFLINE, lw=0.5, ls=(0, (3, 2)), zorder=1).set_gid("ref-one")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["%g" % t for t in xticks])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=SMALL)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Adjusted hazard ratio (log scale)", fontsize=SMALL)
    tr = blended_transform_factory(sf.transSubfigure, ax.transData)
    X = lambda mm: mm / sf.w_mm
    x0, x1 = 0.6, sf.w_mm - 0.6                                  # table frame, as Fig. 3F and forestmodel
    for y in range(len(rows)):                                   # shading of alternate variables, behind the text
        if y in SHADE:
            band = Rectangle((X(x0), y - 0.5), X(x1) - X(x0), 1.0, transform=tr, facecolor="#F0F0F0", edgecolor="none",
                             zorder=0, clip_on=False)
            band.set_gid("bg-band%d" % y)
            ax.add_patch(band)
    plots.table_rules(ax, sf, x0, x1, -1.8, -0.5, len(rows) - 0.5, seps=(41.0, 61.0))
    head = -1.15
    for x_mm, txt, ha in ((1.5, "Variable", "left"), (40.0, "n", "right"), (51.0, "Hazard ratio", "center"),
                          (62.0, "HR (95% CI)", "left"),
                          (88.0, "P", "right")):
        rich_text(ax, X(x_mm), head, [(txt, "bi" if txt in ("P", "n") else "b")], size=SMALL, ha=ha, va="center",
                  transform=tr, gid="pn-h%s" % txt[0])
    for y, rw in enumerate(rows):
        if rw["var"]:
            rich_text(ax, X(1.5), y, [(rw["var"], "b")], size=SMALL, va="center", transform=tr, gid="pn-v%d" % y)
        if rw["lev"]:
            rich_text(ax, X(22.0), y, [(rw["lev"], "")], size=SMALL, va="center", transform=tr, gid="pn-l%d" % y)
        rich_text(ax, X(40.0), y, [("%d" % rw["n"], "")], size=SMALL, ha="right", va="center", transform=tr,
                  gid="pn-n%d" % y)
        rich_text(ax, X(62.0), y, [(rw["stat"], "")], size=SMALL, va="center", transform=tr, gid="pn-s%d" % y)
        if rw["p"] is not None:                                   # the column header is P: values only
            runs = p_runs(rw["p"])[1:]
            runs[0] = (runs[0][0].replace(" = ", ""), runs[0][1])
            rich_text(ax, X(88.0), y, runs, size=SMALL, ha="right", va="center", transform=tr, gid="pn-p%d" % y)
    out = pd.DataFrame(rows)[["var", "lev", "n", "hr", "lo", "hi", "p"]]
    out["var"] = out["var"].replace("", np.nan).ffill()
    source(endpoint.split()[-1], out.rename(columns={"var": "variable", "lev": "level"}))


# ---- C, D ------------------------------------------------------------------------------------------------------------
def feature_color(f):
    """Signatures in their color of Fig. 3A; CNV-like scores gray."""
    base = f.split(":")[-1]
    return P.SIGNATURE.get(base, "#808080")

def regression_row(sf, w, feats, kind, stage_stat):
    lm = kind == "lm"
    x = w["wgs_mm_like_score"].values.astype(float) if lm else w["lasso_pred"].values
    resid = w["lm_resid"].values if lm else w["lasso_resid"].values
    title = "MST score against the WGS MM-like score" if lm else "MST score predicted from 27 mutations (LASSO)"
    at_mm(sf, 4.5, 1.8, title, "pn-title")
    ax = axes_mm(sf, 12.0, 7.0, 34.0, 29.0)
    xj = x + (rng("sf15-" + kind).uniform(-0.15, 0.15, len(x)) if lm else 0.0)
    plots.scatter_groups(ax, xj, w["mst"].values, w["disease_status"].values, P.TUMOR, "sf15" + kind, s=2.2,
                         order=STAGES)
    if lm:
        xs = np.array([x.min(), x.max()])
        fit = np.polyfit(x, w["lm_fitted"].values, 1)          # the fitted values lie on one line
        assert np.allclose(np.polyval(fit, x), w["lm_fitted"].values)
        ln, = ax.plot(xs, np.polyval(fit, xs), color="black", lw=0.6, ls=(0, (3, 2)), zorder=4)
        ax.set_xticks(range(int(x.min()), int(x.max()) + 1))
        ax.set_xticklabels([("%d" % v).replace("-", MINUS) for v in range(int(x.min()), int(x.max()) + 1)])
        ax.set_xlabel("WGS MM-like score")
        line_lab = "Least-squares fit"
    else:
        xs = np.array([x.min(), x.max()])
        ln, = ax.plot(xs, xs, color="black", lw=0.6, ls=(0, (3, 2)), zorder=4)
        ax.xaxis.set_major_locator(plots.nice_locator(4))
        plots.plain_ticks(ax.xaxis)
        ax.set_xlabel("MST score predicted from mutations")
        line_lab = "Predicted = observed"
    ln.set_gid("ref-fit")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    plots.plain_ticks(ax.yaxis)
    ax.tick_params(labelsize=SMALL)
    ax.set_ylabel("MST score")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plots.key(sf, 49.0 / sf.w_mm, 1 - 9.0 / sf.h_mm, [(s, P.TUMOR[s]) for s in STAGES], transform=sf.transSubfigure,
              size=SMALL, ms=2.4, row_pt=2.1 * 72 / 25.4, gid="pn-skey")
    plots.key(sf, 49.0 / sf.w_mm, 1 - 21.5 / sf.h_mm, [(line_lab, "black")], transform=sf.transSubfigure, size=SMALL,
              ms=3.2, line=True, ls=(0, (3, 2)), gid="pn-lkey")
    # residuals by disease stage
    at_mm(sf, 95.0, 1.8, "Residuals by disease stage", "pn-rtitle", ha="center")
    rho, p = stage_stat                                         # Spearman correlation with disease stage (0-4)
    at_mm(sf, 95.0, 4.6, join_runs([("Spearman ", "")] + rho_runs(rho), p_runs(p)), "pn-rstat", size=SMALL, ha="center")
    ar = axes_mm(sf, 83.0, 7.0, 25.0, 24.0)
    d = pd.DataFrame({"group": w["disease_status"].values, "value": resid})
    plots.box_strip(ar, d, STAGES, P.TUMOR, "sf15r" + kind, point=1.5, width=0.62, labels=STAGES, label_size=SMALL,
                    rotate=90, n_labels="row")
    ar.axhline(0, color=P.REFLINE, lw=0.5, ls=(0, (3, 2)), zorder=1).set_gid("ref-zero")
    ar.yaxis.set_major_locator(plots.nice_locator(4))
    plots.plain_ticks(ar.yaxis)
    ar.tick_params(axis="y", labelsize=SMALL)
    ar.set_ylabel("Residual MST score")
    for s in ("top", "right"):
        ar.spines[s].set_visible(False)
    # classifier features correlated with the residuals (q < 0.05), ordered by q
    f = feats[feats["fdr"] < 0.05].sort_values("fdr").reset_index(drop=True)
    assert len(f) == 3
    at_mm(sf, 147.0, 1.8, "Features correlated with the residuals", "pn-ftitle", ha="center")
    ab = axes_mm(sf, 133.0, 9.0, 30.0, 13.5)
    plots.hbar(ab, [names.feature(v) for v in f["feature"]], f["rho"].values, [feature_color(v) for v in f["feature"]],
               height=0.68)                                     # one color per feature, the same in C and D
    ab.set_xlim(-0.6, 0.6)
    ab.set_xticks([-0.5, 0, 0.5])
    plots.plain_ticks(ab.xaxis)
    ab.tick_params(axis="y", length=0, labelsize=SMALL)
    ab.tick_params(axis="x", labelsize=SMALL)
    ab.set_xlabel("Spearman’s ρ with the residuals", fontsize=SMALL)
    tr = blended_transform_factory(ab.transAxes, ab.transData)
    for y, q in enumerate(f["fdr"].values):
        rich_text(ab, 1, y, p_runs(q, "q"), size=SMALL, va="center", transform=plots.pts(ab, 2.0, 0, tr), gid="pn-q%d" % y)
    for s in ("top", "right"):
        ab.spines[s].set_visible(False)
    return f


def main():
    w = pd.read_csv(paths.cache("wgs_samples.tsv"), sep="\t")
    assert len(w) == 84 and w["disease_status"].value_counts().reindex(STAGES).tolist() == [4, 33, 23, 13, 11]
    lm = pd.read_csv(paths.cache("wgs_residual_features_lm.tsv"), sep="\t")
    la = pd.read_csv(paths.cache("wgs_residual_features_lasso.tsv"), sep="\t")
    # expected values
    assert abs(lm.set_index("feature").loc["auc:circulatory_capacity", "rho"] - (-0.528946)) < 1e-6
    assert abs(la.set_index("feature").loc["auc:circulatory_capacity", "rho"] - (-0.500294)) < 1e-6
    st = pd.read_csv(paths.cache("wgs_residual_stage.tsv"), sep="\t").set_index("residual")
    assert abs(st.loc["lm_resid", "rho"] - 0.608735) < 1e-5 and abs(st.loc["lasso_resid", "rho"] - 0.589916) < 1e-5
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_cox(pn["cox_pfs"], "IA22 PFS", "Progression-free survival", (0.45, 7.5), [0.5, 1, 2, 5])
        panel_cox(pn["cox_os"], "IA22 OS", "Overall survival", (0.4, 11.0), [0.5, 1, 2, 5, 10])
        f1 = regression_row(pn["wgs_regression"], w, lm, "lm", st.loc["lm_resid", ["rho", "p"]].tolist())
        f2 = regression_row(pn["lasso_regression"], w, la, "lasso", st.loc["lasso_resid", ["rho", "p"]].tolist())
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("EDFig9", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")
    source("CD_samples", w[["patient_id", "disease_status", "wgs_mm_like_score", "mst", "lm_fitted", "lm_resid", "lasso_pred",
                            "lasso_resid"]])
    source("CD_residuals_by_stage", st.reset_index())
    source("CD_features", pd.concat([f1.assign(model="linear (WGS MM-like score)"), f2.assign(model="LASSO (27 mutations)")]))


if __name__ == "__main__":
    main()
