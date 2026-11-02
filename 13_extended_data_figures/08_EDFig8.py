#!/usr/bin/env python3
"""Extended Data Figure 8: prediction of progression in SMM, serially sampled patients, scores by cytogenetic
group and the MST score in external single-cell datasets.

Purpose:      A, MST scores of the 115 SMM patients by later progression to MM; two-sided Wilcoxon rank-sum P. B, ROC
              curves of the mean MST score, of the percentage of MM-like malignant PCs (MST score > 0.5) and of the IMWG
              2/20/20 risk group for progression, with AUROC and 95% CI. C,
              the two SMM patients sampled at two IMWG 2/20/20 risk levels (risk.py): malignancy and MST scores of every
              PC on their UMAP and by sample (BM PCs and malignant PB PCs); UMAPs by patient and by risk group and PC
              status; single-cell activity of the MHC-I, histone/chaperone and B cell activation signatures in the
              malignant BM PCs of each sample. Each comparison is labeled with the AUROC of the two-sided Mann-Whitney
              test between cells, U / (n1 n2), the probability that a cell of the right-hand group (scores) or of the
              higher-risk sample (signatures) has the higher value (exact P in the source data). D, mean malignancy and
              MST scores of the malignant BM PCs of each tumor by cytogenetic group. E, PCs of three external studies on
              their UMAP, colored by the MST score of each cell and by disease stage and PC status as labeled by the
              authors.

Inputs:       results/stage_and_progression: smm_progression_patient_scores.tsv, smm_progression_results.tsv,
              mm_like_cells_tumors.tsv, mm_like_cells_smm.tsv, bm_tumor_units.tsv, stage_effects_genetics_tests.tsv; results/figure_inputs: longitudinal_cells.npz,
              longitudinal_signatures.npz, external_<study>.npz, patient_id_mean_probability.csv.gz.

Outputs:      figures/EDFig8.svg, EDFig8.pdf; figures/source_data/EDFig8_*.tsv.

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
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

import names
import palettes as P
import paths
import plots
import stage_tests as ST
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, p_runs, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save
from plots import at_mm

LAYOUT = FigureLayout("EDFig8", [
    Slot("smm_progressors", "A", 0, 0, 34, 47),
    Slot("progression_roc", "B", 36, 0, 50, 47),
    Slot("serial_umaps", "C", 88, 0, 95, 47),
    Slot("serial_signatures", "", 0, 49, 93, 66),
    Slot("serial_violins", "", 95, 49, 88, 66),
    Slot("scores_by_cytogenetics", "D", 0, 117, 183, 42),
    Slot("external_umaps", "E", 0, 161, 183, 40),
], max_height=247)
PT_MM = 25.4 / 72.0
RISK = {"LRSMM": 0, "IRSMM": 1, "HRSMM": 2}
MST_COLOR, MMLIKE_COLOR, RISK_COLOR = "#54278F", "#D55E00", "#808080"
INK = "#333333"
THREE = ["MHC-I_gene_module", "histone_chaperone_gene", "B_cell_activation"]      # the signatures of Fig. 3C
PATIENTS = ["Pt-28", "Pt-160"]                     # the two SMM patients sampled at two IMWG 2/20/20 risk levels
GROUP_ORDER = ["Pt-28:BM:IRSMM:normal", "Pt-28:BM:LRSMM:tumor", "Pt-28:BM:IRSMM:tumor", "Pt-160:BM:LRSMM:normal",
               "Pt-160:BM:IRSMM:normal", "Pt-160:BM:LRSMM:tumor", "Pt-160:BM:IRSMM:tumor",
               "Pt-28:PB:IRSMM:tumor", "Pt-160:PB:LRSMM:tumor", "Pt-160:PB:IRSMM:tumor"]
# pairs of the figure, with P (0 = below the floating-point range) and U of the original test
TESTED_PAIRS = {("malignancy", "Pt-28:BM:IRSMM:normal", "Pt-28:BM:IRSMM:tumor"): (0.0, 2.063e5),
             ("malignancy", "Pt-160:BM:IRSMM:normal", "Pt-160:BM:IRSMM:tumor"): (0.0, 7.512e4),
             ("malignancy", "Pt-160:BM:LRSMM:normal", "Pt-160:BM:LRSMM:tumor"): (1.040e-90, 1.122e3),
             ("mst", "Pt-28:BM:LRSMM:tumor", "Pt-28:BM:IRSMM:tumor"): (0.0, 4.623e7),
             ("mst", "Pt-160:BM:LRSMM:tumor", "Pt-160:BM:IRSMM:tumor"): (2.556e-84, 1.141e6)}
TISSUE_BG = {"BM": ("#E5E5E5", "BM"), "PB": ("#FBE1CF", "PB")}
STAGE3 = {"Donor": "Donor", "MGUS": "MGUS", "SMM": "IRSMM", "MM": "MM"}       # SMM is not split by risk in these studies
EXT_GROUPS = ["Donor", "MGUS normal", "SMM normal", "MM normal", "MGUS malignant", "SMM malignant", "MM malignant"]
EXT_COLORS = {"Donor": P.DONOR}
EXT_COLORS.update({"%s normal" % s: P.NORMAL[STAGE3[s]] for s in ("MGUS", "SMM", "MM")})
EXT_COLORS.update({"%s malignant" % s: P.TUMOR[STAGE3[s]] for s in ("MGUS", "SMM", "MM")})
STUDIES = [("ledergor", "Ledergor et al."), ("boiarsky", "Boiarsky et al."), ("dang", "Dang et al.")]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/EDFig8_%s.tsv" % key, sep="\t", index=False)


def close(a, b, rel=1e-6):
    assert abs(a - b) <= rel * max(1e-300, abs(b)), (a, b)


# ---- A, B ------------------------------------------------------------------------------------------------------------
def panel_progressors(sf, d, res):
    d = d.assign(group=np.where(d["progressed"] == 1, "Yes", "No"), value=d["mst"])
    p = stats.mannwhitneyu(d.loc[d["group"] == "Yes", "mst"], d.loc[d["group"] == "No", "mst"],
                           alternative="two-sided").pvalue
    close(p, float(res["wilcoxon_p"]))
    at_mm(sf, 18.5, 1.8, "Patients with SMM", "pn-title", ha="center")
    ax = axes_mm(sf, 10.0, 5.0, 21.0, 29.0)
    order = ["No", "Yes"]
    tops = plots.box_strip(ax, d, order, {"No": "#BDBDBD", "Yes": P.TUMOR["MM"]}, "sf14a", point=1.6, width=0.6,
                           labels=["No", "Yes"], label_size=BASE)
    plots.refline(ax, 0.5)
    ax.set_ylim(0, 1.0)
    plots.brackets(ax, [dict(i=0, j=1, p=p, color=P.BRACKET_STAGE)], tops, ymin=0, key="sf14a")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    plots.plain_ticks(ax.yaxis)
    ax.set_ylabel("MST score")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    at_mm(sf, 20.5, 44.5, "Progressed to MM", "pn-xlab", ha="center")
    source("A", d[["patient_id", "disease_status", "progressed", "mst"]])


def panel_roc(sf, d, res, mm):
    y = d["progressed"].values
    risk = d["disease_status"].map(RISK).values
    at_mm(sf, 26.0, 1.8, "Progression to MM", "pn-title", ha="center")
    ax = axes_mm(sf, 12.0, 5.0, 29.0, 29.0)
    rows = []
    for values, col, lab, a, lo, hi, key in [
            (d["mst"].values, MST_COLOR, "MST score", "auroc_mst", "auroc_mst_lo", "auroc_mst_hi", "mst"),
            (d["pct"].values, MMLIKE_COLOR, "MM-like cells (%)", "auroc_pct", "auroc_pct_lo", "auroc_pct_hi", "pct"),
            (risk, RISK_COLOR, "2/20/20 risk group", "auroc_2_20_20", "auroc_2_20_20_lo", "auroc_2_20_20_hi", "risk")]:
        close(roc_auc_score(y, values), float((mm if key == "pct" else res)[a]))
        fpr, tpr, _ = roc_curve(y, values)
        ax.plot(fpr, tpr, color=col, lw=0.8, drawstyle="default", zorder=3)
        rows += [dict(score=lab, fpr=f, tpr=t) for f, t in zip(fpr, tpr)]
    ax.plot([0, 1], [0, 1], color="#808080", lw=0.5, ls=(0, (3, 2)), zorder=1)[0].set_gid("ref-diag")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        plots.plain_ticks(axis)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    items = [("%s, AUROC %.2f (%.2f–%.2f)" % (lab, float(src[a]), float(src[lo]), float(src[hi])), col)
             for lab, col, src, a, lo, hi in [
                 ("MST score", MST_COLOR, res, "auroc_mst", "auroc_mst_lo", "auroc_mst_hi"),
                 ("MM-like cells", MMLIKE_COLOR, mm, "auroc_pct", "auroc_pct_lo", "auroc_pct_hi"),
                 ("2/20/20", RISK_COLOR, res, "auroc_2_20_20", "auroc_2_20_20_lo", "auroc_2_20_20_hi")]]
    plots.key(sf, 2.0 / sf.w_mm, 1 - 40.5 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=3.0, line=True,
              row_pt=2.2 / PT_MM, gid="pn-key")
    source("B", pd.DataFrame(rows))


# ---- C ---------------------------------------------------------------------------------------------------------------
def longitudinal():
    c = np.load(paths.cache("longitudinal_cells.npz"))
    cat = lambda k: c[k + "__cats"].astype(str)[c[k + "__codes"]]
    d = pd.DataFrame({"x": c["umap"][:, 0], "y": c["umap"][:, 1], "patient": cat("patient_id"), "tissue": cat("sample_type"),
                      "annotation": cat("annotation"), "risk": cat("disease_status"),
                      "malignancy": c["malignancy"].astype(float), "mst": c["mst"].astype(float)})
    d["risk"] = ST.imwg_stages(d["patient"], d["tissue"], d["risk"])      # IMWG 2/20/20 risk groups
    d = d[d["patient"].isin(PATIENTS)].reset_index(drop=True)
    d["group"] = d["patient"] + ":" + d["tissue"] + ":" + d["risk"] + ":" + d["annotation"]
    return d


def panel_serial_umaps(sf, d):
    xy = d[["x", "y"]].values
    for k, (col, title, cmap) in enumerate((("malignancy", "Malignancy score", P.CMAP_MALIGNANCY),
                                             ("mst", "MST score", P.CMAP_MST))):
        x0 = 6.0 + 47.0 * k
        ax = axes_mm(sf, x0, 6.0, 33.0, 33.0)
        v = d[col].values
        ok = ~np.isnan(v)
        order = np.r_[np.where(~ok)[0], np.where(ok)[0][np.argsort(v[ok], kind="stable")]]   # scanpy: high values on top
        norm = Normalize(0, 1)
        colr = np.tile([0.85, 0.85, 0.85, 1.0], (len(v), 1))
        colr[ok] = plots.matplotlib_cm(cmap)(norm(v[ok]))
        plots.umap(ax, xy, colr, "sf14c-" + col, s=0.35, order=order)
        ax.set_title(title, fontsize=BASE, pad=2.0)
        cax = axes_mm(sf, x0 + 34.5, 14.0, 1.4, 17.0)
        cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap=cmap), cax, ticks=[0, 0.5, 1])
        plots.plain_ticks(cb.ax.yaxis)


def panel_serial_signatures(sf):
    """Schematic of the risk group at the two samples, UMAPs by patient and by risk group and PC status, and the
    single-cell activity of the three signatures of Fig. 3C in the malignant BM PCs of the two samples of each patient."""
    c = np.load(paths.cache("longitudinal_cells.npz"), allow_pickle=True)
    sg = np.load(paths.cache("longitudinal_signatures.npz"), allow_pickle=True)
    cat = lambda k: c[k + "__cats"].astype(str)[c[k + "__codes"]]
    d = pd.DataFrame({"x": c["umap"][:, 0], "y": c["umap"][:, 1], "patient": cat("patient_id"),
                      "tissue": cat("sample_type"), "annotation": cat("annotation"), "risk": cat("disease_status")})
    d["risk"] = ST.imwg_stages(d["patient"], d["tissue"], d["risk"])     # IMWG 2/20/20 risk groups
    snames = [str(s_) for s_ in sg["names"]]
    for sig in THREE:
        d[sig] = sg["scores"][:, snames.index("mean_expr:" + sig)]
    pairs = {"Pt-28": ("LRSMM", "IRSMM"), "Pt-160": ("IRSMM", "LRSMM")}   # first -> second
    d = d[d["patient"].isin(PATIENTS).values].reset_index(drop=True)
    # schematic: IMWG 2/20/20 risk group of the first and the second sample of each patient (arrow: first -> second)
    ax0 = axes_mm(sf, 1.0, 2.0, 38.0, 15.0)
    ax0.set_xlim(0, 38)
    ax0.set_ylim(15, 0)
    ax0.axis("off")
    xr = {"LRSMM": 6.0, "IRSMM": 16.0, "HRSMM": 26.0}
    ax0.plot([2.5, 29.5], [10.8, 10.8], color=INK, lw=0.6, zorder=1)[0].set_gid("schema-axis")
    for rk, xx in xr.items():
        ax0.plot([xx], [10.8], marker="o", ms=3.0, mfc=P.TUMOR[rk], mec="black", mew=0.3, ls="none", zorder=3)
        rich_text(ax0, xx, 13.4, [(rk, "")], size=SMALL, ha="center", va="center", transform=ax0.transData,
                  gid="pn-risk" + rk)
    for k, pt in enumerate(PATIENTS):
        a_, b_ = pairs[pt]
        yy = 3.0 + 2.5 * k
        plots.arrow_line(ax0, xr[a_], yy, xr[b_], yy, P.PATIENT3[pt], "schema-" + pt)
        rich_text(ax0, 28.5, yy, [(pt, "")], size=SMALL, ha="left", va="center", transform=ax0.transData,
                  gid="pn-pt%d" % k)
    rich_text(ax0, 0.2, 0.6, [("IMWG 2/20/20 risk at the two samples", "")], size=SMALL, ha="left", va="center",
              transform=ax0.transData, gid="pn-schematitle")
    # UMAPs of all PCs of the two patients: by patient, and by risk group and PC status
    xy = d[["x", "y"]].values
    au = axes_mm(sf, 45.0, 4.0, 15.0, 15.0)
    plots.umap(au, xy, np.array([P.PATIENT3[p_] for p_ in d["patient"]]), "sf14c-long-pt", s=0.25, labels=False)
    au.set_title("Patient", fontsize=SMALL, pad=1.5)
    plots.key(sf, 61.0 / sf.w_mm, 1 - 5.5 / sf.h_mm, [(p_, P.PATIENT3[p_]) for p_ in PATIENTS], size=SMALL, ms=2.2,
              transform=sf.transSubfigure, gid="pn-ptkey")
    av = axes_mm(sf, 75.5, 4.0, 15.0, 15.0)
    colr = [(P.TUMOR if a_ == "tumor" else P.NORMAL)[r_] for a_, r_ in zip(d["annotation"], d["risk"])]
    plots.umap(av, xy, np.array(colr), "sf14c-long-risk", s=0.25, labels=False)
    av.set_title("Risk group, PC status", fontsize=SMALL, pad=1.5)
    shown = [r_ for r_ in RISK if any(r_ in pairs[p_] for p_ in PATIENTS)]          # risk groups of the samples
    for k, (lab, pal) in enumerate((("Normal PCs", P.NORMAL), ("Malignant PCs", P.TUMOR))):
        yy = 21.6 + 2.5 * k
        at_mm(sf, 45.0, yy, [(lab, "")], "pn-kl%d" % k, size=SMALL)
        plots.key(sf, 58.5 / sf.w_mm, 1 - (yy - 1.2) / sf.h_mm, [(r_, pal[r_]) for r_ in shown], size=SMALL, ms=2.2,
                  ncol=3, col_pt=rich_width_pt(sf, [("HRSMM", "")], SMALL) / PT_MM * PT_MM + 7.5, row_pt=2.4 / PT_MM,
                  transform=sf.transSubfigure, gid="pn-kr%d" % k)
    # violins: three signatures x two patients, malignant BM PCs of the two samples
    xs, ys, w, h = [15.5, 55.0], [33.5, 42.5, 51.5], 33.0, 7.0
    rows = []
    for i, sig in enumerate(THREE):
        lab = names.SIGNATURES[sig]
        lines = {"MHC-I_gene_module": ["MHC-I gene", "module"], "histone_chaperone_gene": ["Histone/", "chaperone"],
                 "B_cell_activation": ["B cell", "activation"]}[sig]
        for li, ln in enumerate(lines):
            at_mm(sf, 0.4, ys[i] + h / 2 + (li - 0.5) * 2.2, [(ln, "")], "pn-vrow%d_%d" % (i, li), size=SMALL)
        for j, pt in enumerate(PATIENTS):
            ax = axes_mm(sf, xs[j], ys[i], w, h)
            groups = sorted(pairs[pt], key=RISK.get)
            vals = [d.loc[(d["patient"] == pt) & (d["tissue"] == "BM") & (d["annotation"] == "tumor") & (d["risk"] == r_),
                          sig].values.astype(float) for r_ in groups]
            for k, (r_, v) in enumerate(zip(groups, vals)):
                plots.violin_box(ax, v, k, P.TUMOR[r_], "%s-%s-%s" % (sig, pt, r_))
            l10, u = plots.mwu_log10p(vals[0], vals[1])
            p = stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided").pvalue
            if p > 0:
                close(10 ** l10, p, 1e-6)
            auc = 1.0 - u / (len(vals[0]) * len(vals[1]))          # P(a cell of the higher-risk sample is higher)
            rows.append(dict(signature=lab, patient=pt, first=pairs[pt][0], second=pairs[pt][1],
                             n_cells_lower_risk=len(vals[0]), n_cells_higher_risk=len(vals[1]), auroc=auc, log10_p=l10))
            top = max(v.max() for v in vals)
            bot = min(v.min() for v in vals)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(bot - 0.03 * (top - bot), top + 0.55 * (top - bot))
            rich_text(ax, 0.5, 1, [("AUROC %.2f" % auc, "")], size=SMALL, ha="center", va="top",
                      transform=plots.pts(ax, 0, -2.0, ax.transAxes), gid="ax-auc%d%d" % (i, j))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(groups if i == 2 else [], fontsize=SMALL)
            ax.tick_params(axis="x", length=0, pad=1.0)
            if i == 2:                                             # cells per sample (the same for the three rows)
                plots.n_row(ax, [len(v) for v in vals])
            ax.tick_params(axis="y", labelsize=SMALL)
            ax.yaxis.set_major_locator(plots.nice_locator(2))
            plots.plain_ticks(ax.yaxis)
            if i == 0:
                at_mm(sf, xs[j] + w / 2, ys[0] - 1.5, [(pt, "")], "pn-vcol%d" % j, size=SMALL, ha="center")
    at_mm(sf, 0.4, 29.0, [("Malignant BM PCs, signature activity", "")], "pn-vtitle", size=SMALL)
    source("C_signature_tests", pd.DataFrame(rows))


def tested_pairs(df, col):
    """Malignancy score: malignant vs normal BM PCs of each sample. MST score: the malignant PCs of the two samples of a
    patient in the same tissue, tested only if a sample's mean is above 0.5; the figure shows the significant pairs
    (TESTED_PAIRS)."""
    pairs = []
    if col == "malignancy":
        for g in df.loc[df["annotation"] == "normal", "group"].unique():
            t = g.replace(":normal", ":tumor")
            if (df["group"] == t).any():
                pairs.append((g, t))
    else:
        for tissue in ("BM", "PB"):
            ts = list(df.loc[(df["annotation"] == "tumor") & (df["tissue"] == tissue), "group"].unique())
            if len(ts) == 2:
                pairs.append(tuple(ts))
    return pairs


def panel_serial_violins(sf, d):
    shown = d[(d["tissue"] == "BM") | (d["annotation"] == "tumor")]
    groups = {p: [g for g in GROUP_ORDER if g.startswith(p + ":") and (shown["group"] == g).any()] for p in PATIENTS}
    assert sum(len(v) for v in groups.values()) == 10
    widths = {p: 5.35 * len(g) for p, g in groups.items()}
    xs, x = {}, 13.0
    for p in PATIENTS:
        xs[p] = x
        x += widths[p] + 2.5
    rows, tests = [("malignancy", "Malignancy score", 8.0), ("mst", "MST score", 26.5)], []
    for r, (col, ylab, y0) in enumerate(rows):
        row_axes, jobs = [], []
        for j, p in enumerate(PATIENTS):
            df = shown[shown["patient"] == p]
            gs = groups[p]
            ax = axes_mm(sf, xs[p], y0, widths[p], 15.0)
            first_pb = [k for k, g in enumerate(gs) if ":PB:" in g][0]
            for t, (a, b) in (("BM", (-0.5, first_pb - 0.5)), ("PB", (first_pb - 0.5, len(gs) - 0.5))):
                ax.axvspan(a, b, color=TISSUE_BG[t][0], lw=0, zorder=0).set_gid("bg-%s" % t)
            for k, g in enumerate(gs):
                v = df.loc[df["group"] == g, col].values
                risk, ann = g.split(":")[2], g.split(":")[3]
                plots.violin_box(ax, v, k, (P.TUMOR if ann == "tumor" else P.NORMAL)[risk], "sf14c-%s-%s" % (col, g),
                                 width=0.82)
            plots.refline(ax, 0.5)
            comps, top = [], {k: float(df.loc[df["group"] == g, col].max()) for k, g in enumerate(gs)}
            for a, b in [sorted(pr, key=gs.index) for pr in tested_pairs(df, col)]:   # statannotations: x order
                if (col, a, b) not in TESTED_PAIRS:          # the pairs the figure shows
                    continue
                l10, u = plots.mwu_log10p(df.loc[df["group"] == a, col].values, df.loc[df["group"] == b, col].values)
                ref_p, ref_u = TESTED_PAIRS[(col, a, b)]
                close(u, ref_u, 6e-4)
                if ref_p > 0:
                    close(10 ** l10, ref_p, 1e-3)
                n_a, n_b = int((df["group"] == a).sum()), int((df["group"] == b).sum())
                auc = 1.0 - u / (n_a * n_b)                  # P(a cell of the right-hand group scores higher)
                tests.append(dict(score=ylab, first=a, second=b, n_first=n_a, n_second=n_b, U=u, auroc=auc, log10_p=l10))
                comps.append(dict(i=gs.index(a), j=gs.index(b), runs=[("AUROC %.2f" % auc, "")],
                                  color=P.BRACKET_MAL if col == "malignancy" else P.BRACKET_STAGE))
            ax.set_xlim(-0.5, len(gs) - 0.5)
            jobs.append((ax, comps, top, "sf14c%d%d" % (r, j)))
            row_axes.append(ax)
            ax.set_yticks([0, 0.5, 1.0])
            plots.plain_ticks(ax.yaxis)
            ax.tick_params(axis="y", labelsize=SMALL)
            if j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab)
            ax.set_xticks(range(len(gs)))
            if r == len(rows) - 1:
                labels = ["%s %s" % (g.split(":")[2], "malignant" if g.endswith("tumor") else "normal") for g in gs]
                ax.set_xticklabels(labels, rotation=90, fontsize=SMALL)
                plots.n_row(ax, [int((df["group"] == g).sum()) for g in gs], rotate=True)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
                at_mm(sf, xs[p] + widths[p] / 2.0, y0 - 6.2, p, "pn-pt%d" % j, ha="center")
        plots.brackets_shared(jobs, -0.03)                      # one y scale per row, brackets placed at that scale
    keyx = 1.0
    for k, t in enumerate(("BM", "PB")):
        rect = Rectangle(((keyx + 7.0 * k) / sf.w_mm, 1 - 2.9 / sf.h_mm), 2.2 / sf.w_mm, 2.2 / sf.h_mm,
                         transform=sf.transSubfigure, facecolor=TISSUE_BG[t][0], edgecolor="#333333", lw=0.3)
        rect.set_gid("key-%s" % t)
        sf.add_artist(rect)
        at_mm(sf, keyx + 7.0 * k + 3.0, 1.8, t, "pn-tk%d" % k, size=SMALL)
    assert len(tests) == len(TESTED_PAIRS)
    source("C_tests", pd.DataFrame(tests))
    source("C_cells", shown.groupby("group").agg(n_cells=("mst", "size"), median_malignancy=("malignancy", "median"),
                                                 median_mst=("mst", "median")).reindex(GROUP_ORDER).reset_index())


# ---- D ---------------------------------------------------------------------------------------------------------------
def panel_cytogenetics(sf):
    """The 181 BM tumors (mean scores of the malignant BM PCs of a patient at one stage), the units of the Kruskal-Wallis
    tests reported in the Results (stage_effects_genetics_tests.tsv; recomputed and asserted)."""
    u = pd.read_csv(paths.result("stage_and_progression", "bm_tumor_units.tsv"), sep="\t")
    u["group"] = u["cytogenetics"].map(ST.GENE)
    assert len(u) == 181 and u["group"].notna().all()
    T = pd.read_csv(paths.result("stage_and_progression", "stage_effects_genetics_tests.tsv"), sep="\t")
    at_mm(sf, 91.5, 1.6, "Malignant BM PCs by cytogenetic group", "pn-title", ha="center")
    for k, (col, lab) in enumerate((("Malignancy Score", "Malignancy score"), ("MST Score", "MST score"))):
        groups = [g for g in P.CYTO_ORDER if (u["group"] == g).any()]
        p = stats.kruskal(*[u.loc[u["group"] == g, col] for g in groups]).pvalue
        ref = T[(T["test"] == "Kruskal-Wallis") & T["detail"].str.startswith(col)]
        assert len(ref) == 1
        close(p, float(ref["P"].iloc[0]))
        ax = axes_mm(sf, 12.0 + 92.0 * k, 6.5, 76.0, 18.0)
        plots.box_strip(ax, u.assign(value=u[col].astype(float)), P.CYTO_ORDER, P.CYTO, "sf14d%d" % k, point=1.2,
                        rotate=90, label_size=SMALL, n_labels="row", width=0.62, jitter=0.18, edge=0.25)
        plots.refline(ax, 0.5)
        ax.set_ylim(-0.03, 1.06)
        ax.set_yticks([0, 0.5, 1.0])
        plots.plain_ticks(ax.yaxis)
        ax.tick_params(axis="y", labelsize=SMALL)
        ax.set_ylabel(lab)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        rich_text(ax, 1.0, 1, [("Kruskal-Wallis ", "")] + p_runs(p), size=SMALL, ha="right", va="bottom",
                  transform=plots.pts(ax, 0, 1.2, ax.transAxes), gid="pn-kw%d" % k)
    out = u[["patient_id", "disease_status", "group", "Malignancy Score", "MST Score"]]
    source("D", out.rename(columns={"group": "cytogenetic_group"}))


# ---- E ---------------------------------------------------------------------------------------------------------------
def external(study):
    z = np.load(paths.cache("external_%s.npz" % study))
    cat = lambda k: z[k + "__cats"].astype(str)[z[k + "__codes"]]
    d = pd.DataFrame({"x": z["umap"][:, 0], "y": z["umap"][:, 1], "stage": cat("disease_status"),
                      "annotation": cat("annotation"), "mst": z["mst"].astype(float), "kept": z["kept"]})
    d = d[d["kept"] & d["annotation"].isin(["tumor", "normal"])].copy()        # other labels dropped
    d["group"] = d["stage"] + " " + np.where(d["annotation"] == "tumor", "malignant", "normal")
    d.loc[d["stage"] == "Donor", "group"] = "Donor"
    assert d["mst"].notna().all() and set(d["group"]) <= set(EXT_GROUPS)
    return d


def panel_external(sf):
    norm = Normalize(0, 1)
    counts = []
    for k, (study, name) in enumerate(STUDIES):
        d = external(study)
        x0 = 4.0 + 57.5 * k
        at_mm(sf, x0 + 26.0, 1.6, name, "pn-study%d" % k, ha="center")
        xy = d[["x", "y"]].values
        a1 = axes_mm(sf, x0, 6.5, 25.0, 25.0)
        order = np.argsort(d["mst"].values, kind="stable")
        plots.umap(a1, xy, plots.matplotlib_cm(P.CMAP_MST)(norm(d["mst"].values)), "sf14e-mst-%s" % study, s=0.12,
                   order=order, labels=(k == 0))
        a1.set_title("MST score", fontsize=SMALL, pad=1.2)
        a2 = axes_mm(sf, x0 + 27.0, 6.5, 25.0, 25.0)
        plots.umap(a2, xy, np.array([EXT_COLORS[g] for g in d["group"]]), "sf14e-grp-%s" % study, s=0.12, labels=False)
        a2.set_title("Disease stage, PC status", fontsize=SMALL, pad=1.2)
        counts.append(d.groupby("group").size().reindex(EXT_GROUPS).fillna(0).astype(int).rename(name))
    cax = axes_mm(sf, 176.0, 11.0, 1.4, 17.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap=P.CMAP_MST), cax, ticks=[0, 0.5, 1])
    plots.plain_ticks(cb.ax.yaxis)
    for k, word in enumerate(("MST", "score")):
        rich_text(cax, 0, 1, [(word, "")], size=SMALL, ha="left", va="bottom",
                  transform=plots.pts(cax, 0, 3.8 + (1 - k) * SMALL * 1.2, cax.transAxes), gid="pn-cbt%d" % k)
    items = [(g, EXT_COLORS[g]) for g in EXT_GROUPS]
    plots.key(sf, 4.0 / sf.w_mm, 1 - 35.5 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.4, ncol=7,
              gid="pn-key")
    source("E_cells_per_group", pd.concat(counts, axis=1).reset_index().rename(columns={"index": "group"}))


def main():
    d = pd.read_csv(paths.result("stage_and_progression", "smm_progression_patient_scores.tsv"), sep="\t")
    res = pd.read_csv(paths.result("stage_and_progression", "smm_progression_results.tsv"), sep="\t").iloc[0]
    assert len(d) == int(res["n_patients"]) == 115 and int(d["progressed"].sum()) == int(res["n_progressed"]) == 13
    # percentage of MM-like malignant PCs of the same tumor and its AUROC (09_stage_and_progression/09)
    t = pd.read_csv(paths.result("stage_and_progression", "mm_like_cells_tumors.tsv"), sep="\t")
    d = d.merge(t[["patient_id", "disease_status", "frac_mm_like"]], on=["patient_id", "disease_status"], how="left")
    d["pct"] = 100 * d["frac_mm_like"]
    assert d["pct"].notna().all()
    mm = pd.read_csv(paths.result("stage_and_progression", "mm_like_cells_smm.tsv"), sep="\t").set_index("group").loc["all SMM"]
    lg = longitudinal()
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_progressors(pn["smm_progressors"], d, res)
        panel_roc(pn["progression_roc"], d, res, mm)
        panel_serial_umaps(pn["serial_umaps"], lg)
        panel_serial_signatures(pn["serial_signatures"])
        panel_serial_violins(pn["serial_violins"], lg)
        panel_cytogenetics(pn["scores_by_cytogenetics"])
        panel_external(pn["external_umaps"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("EDFig8", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
