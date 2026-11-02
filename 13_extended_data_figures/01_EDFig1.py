#!/usr/bin/env python3
"""Extended Data Figure 1: cohort structure and overview of malignant PCs across the cohort.

Purpose:      A, cohort, sampling and analysis flow diagram (every row adds up, asserted). B, percentage of malignant PCs
              per individual and stage in BM and in PB: two-sided Wilcoxon rank-sum tests, Benjamini-Hochberg within
              tissue, exact q (consecutive stages in BM, MGUS against the significant stages in PB), checked against the
              saved stage comparisons. C, CNVs of each Numbat clone by disease stage in BM tumors, with the progression
              track.

Inputs:       results/cohort: cohort_by_stage.tsv, sampling_design.tsv, analysis_flow.tsv; results/stage_and_progression/
              stage_comparisons.tsv; data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (via figlib/stage_tests.py);
              results/figure_inputs: clones_<stage>.npz, clones_rows.tsv, clones_bands.tsv.

Outputs:      figures/EDFig1.svg, EDFig1.pdf; figures/source_data/EDFig1_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths). Also icons.py.
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest as multi
from matplotlib.colors import ListedColormap

import icons
import palettes as P
import paths
import plots
import stage_tests as ST
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, p_runs, SMALL, BASE, TITLE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("EDFig1", [
    Slot("flow", "A", 0, 0, 183, 70),
    Slot("tumor_frac", "B", 0, 72, 183, 40),
    Slot("clone_cnv", "C", 0, 114, 183, 86),
], max_height=247)
PT_MM = 25.4 / 72.0
STAGES5 = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
GENE = {"t(11;14)": "CCND1", "t(12;14)": "CCND2", "t(6;14)": "CCND3", "t(4;14)": "NSD2", "t(14;16)": "MAF",
        "t(8;14)": "MAFA", "t(14;20)": "MAFB", "Other": "Other", "HRD": "HRD"}
CNV = {-1: "#2166AC", 0: "#F2F2F2", 1: "#B2182B"}             # loss, neutral, gain


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/EDFig1_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center"):
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs if isinstance(runs, list) else [(runs, "")], size=size, ha=ha,
                     va=va, transform=sf.transSubfigure, gid=gid)


# ---- A ---------------------------------------------------------------------------------------------------------------
def panel_flow(sf):
    res = lambda n: paths.result("cohort", n)
    st = pd.read_csv(res("cohort_by_stage.tsv"), sep="\t").set_index("stage")
    des = pd.read_csv(res("sampling_design.tsv"), sep="\t").set_index("design")["All"]
    fl = pd.read_csv(res("analysis_flow.tsv"), sep="\t").set_index("key")["n"]
    ax = icons.canvas(sf, "flow")
    a = st.loc["All"]
    n_donor = int(st.loc["Donor", "individuals"])
    smm = st.loc["SMM (any risk)"]
    stage_n = [int(st.loc[s, "individuals"]) for s in ("Donor", "MGUS")] + [int(smm["individuals"]),
                                                                              int(st.loc["MM", "individuals"])]
    design = [int(des[k]) for k in ("BM only, one time point", "BM and PB, one time point", "PB only, one time point",
                                    "several time points")]
    assert sum(stage_n) == int(a["individuals"]) == sum(design) == 226
    risk = [int(st.loc[s, "individuals"]) for s in ("LRSMM", "IRSMM", "HRSMM")]
    assert sum(risk) == int(smm["individuals"]) + int(fl["SMM_at_two_risk_levels"])

    def fbox(x, y, w, h, lines, key, edge="#4D4D4D", lw=0.5, size=BASE, first=TITLE, sizes=None):
        """A rounded box with centered lines: the first bold at `first` pt, the others at `size` pt (or `sizes`, per line)."""
        p = icons.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=edge, lw=lw, zorder=0)
        p.set_gid("bg-box-" + key)
        ax.add_patch(p)
        sizes = sizes or [first] + [size] * (len(lines) - 1)
        lh = [s * 1.25 * PT_MM for s in sizes]
        yy = y + (h - sum(lh)) / 2
        assert sum(lh) <= h - 1.0, (key, sum(lh), h)
        for i, (ln, s, l) in enumerate(zip(lines, sizes, lh)):
            icons.label(ax, x + w / 2, yy + l / 2, ln, "%s%d" % (key, i), size=s, bold=(i == 0), width=w - 2.0)
            yy += l

    X0 = 1.5                                                    # boxes clear of the page edges
    W = sf.w_mm - 2 * X0
    fmt = "{:,}".format
    fbox(X0, 3.4, W, 10.8, ["%d individuals (%d patients, %d healthy donors)" % (a["individuals"],
                                                                              a["individuals"] - n_donor, n_donor),
                             "%d samples (%d BM, %d PB)" % (a["samples"], a["BM_samples"], a["PB_samples"]),
                             "%s PCs passing quality control" % fmt(int(a["malignant_PCs"] + a["normal_PCs"]))],
         "top", size=TITLE)
    gap = 2.5
    w4 = (W - 3 * gap) / 4
    xs4 = [X0 + i * (w4 + gap) for i in range(4)]
    icons.label(ax, X0, 16.3, [("Disease stage at sampling (individuals)", "b")], "l1", ha="left", width=W)
    # stage name, then the number of individuals on its own line, as in the sampling-design boxes below
    stage_boxes = [("Donor", ["Healthy donors", "%d" % stage_n[0]]), ("MGUS", ["MGUS", "%d" % stage_n[1]]),
                   ("IRSMM", ["SMM", "%d" % stage_n[2], "LR %d, IR %d, HR %d (%d at two risk levels)"
                              % (tuple(risk) + (int(fl["SMM_at_two_risk_levels"]),))]),
                   ("MM", ["MM", "%d" % stage_n[3]])]
    for x, (col, lines) in zip(xs4, stage_boxes):
        fbox(x, 18.4, w4, 9.6, lines, "st" + col, edge=P.STAGE[col], lw=1.0, sizes=[BASE, TITLE, 5.5][:len(lines)])
    icons.label(ax, X0, 30.1, [("Sampling design (individuals)", "b")], "l2", ha="left", width=W)
    for i, (x, t) in enumerate(zip(xs4, ["One BM sample", "BM + PB, one visit", "PB only", "Two or three visits"])):
        fbox(x, 32.2, w4, 7.2, [t, "%d" % design[i]], "ds%d" % i, size=TITLE, first=BASE)
    icons.label(ax, X0, 41.4, [("Use in the analyses", "b")], "l3", ha="left", width=W)
    cls = lambda k: sum(int(fl["classifier_%s:%s" % (k, s)]) for s in ("Donor", "MGUS", "MM"))
    use = [["Stage comparisons", "mean of all PCs per individual, stage,",
            "tissue and cell type: all %d individuals;" % a["individuals"],
            "%d BM tumors with ≥ 30 malignant PCs" % fl["BM_tumors_at_least_30_malignant"]],
           ["Signature discovery", "all %d individuals, BM and PB;" % a["individuals"],
            "at most 1,000 malignant and 1,000 normal", "PCs per individual and tissue"],
           ["Malignant vs normal PCs", "BM, per patient and stage, ≥ 30 PCs of each:",
            "%d of %d tumors; %d excluded" % (fl["BM_tumors_at_least_30_malignant_and_normal"], fl["BM_tumors"],
                                             fl["BM_tumors_fewer_than_30_normal"]),
            "(fewer than 30 normal PCs)"],
           ["CNV and cytogenetics", "BM and PB pooled per patient,",
            "≥ 30 malignant PCs: %d of %d patients;" % (fl["patients_with_at_least_30_malignant_PCs"], fl["patients"]),
            "%d excluded (no malignant PCs)" % fl["patients_without_malignant_PCs"]],
           ["Classifiers (BM)", "≥ 500 malignant (MGUS, MM) or normal",
            "(donor) PCs: %d of %d individuals;" % (cls("eligible"), cls("BM_units")),
            "%d training, %d test" % (cls("training"), cls("test"))],
           ["Longitudinal", "%d individuals with two or three visits," % fl["individuals_several_visits"],
            "each visit analyzed separately;", "%d patients with SMM at two risk levels" % fl["SMM_patients_two_risk_levels"]]]
    w3 = (W - 2 * gap) / 3
    for i, lines in enumerate(use):
        fbox(X0 + (i % 3) * (w3 + gap), 43.4 + (i // 3) * 13.8, w3, 12.6, lines, "u%d" % i)


# ---- B ---------------------------------------------------------------------------------------------------------------
def tumor_fraction():
    c = ST.cells()
    stage = ST.imwg_stages(c["patient_id"], c["sample_type"], c["disease_status"])     # IMWG 2/20/20 risk groups
    c = c.assign(stage=stage)
    out = {}
    for tissue in ("BM", "PB"):
        x = c[(c["stage"] != "Donor") & (c["sample_type"] == tissue)]
        f = x.groupby(["patient_id", "stage"]).agg(tumor=("annotation", lambda s: (s == "tumor").sum()),
                                                   N=("annotation", "size")).reset_index()
        f["value"] = f["tumor"] / f["N"] * 100
        pairs = list(combinations(f["stage"].unique(), 2))
        p = [scipy.stats.ranksums(f.loc[f["stage"] == a, "value"], f.loc[f["stage"] == b, "value"])[1] for a, b in pairs]
        q = multi.fdrcorrection(p, alpha=0.05, method="indep", is_sorted=False)[1]
        out[tissue] = (f, {tuple(sorted(pr, key=STAGES5.index)): qq for pr, qq in zip(pairs, q)})
    ref = pd.read_csv(paths.result("stage_and_progression", "stage_comparisons.tsv"), sep="\t")
    ref = ref[ref["panel"] == "tumor fraction"]
    for _, r in ref.iterrows():
        assert abs(out[r["test"]][1][tuple(r["comparison"].split(","))] - r["fdr"]) < 1e-12
    return out


def panel_tumor_frac(sf):
    tf = tumor_fraction()
    rows = []
    for k, tissue in enumerate(("BM", "PB")):
        f, q = tf[tissue]
        ax = axes_mm(sf, 14.0 + k * 92.0, 6.0, 72.0, 26.0)
        d = f.assign(group=f["stage"])
        ax.set_ylim(-4, 104)
        tops = plots.box_strip(ax, d, STAGES5, P.TUMOR, "sf1b-" + tissue, point=1.6, label_size=SMALL, width=0.6,
                               n_labels="below")
        ax.set_ylabel("Malignant PCs (%)")
        ax.set_title("%s samples" % tissue, fontsize=BASE, pad=2.0)
        ax.set_yticks([0, 50, 100])
        ax.tick_params(axis="y", labelsize=SMALL)
        # brackets: consecutive stages in BM; MGUS against every significant stage in PB
        if tissue == "BM":
            pairs = list(zip(STAGES5[:-1], STAGES5[1:]))
        else:
            pairs = [("MGUS", s) for s in STAGES5[1:] if q[("MGUS", s)] < 0.05]
        comps = [dict(i=STAGES5.index(a), j=STAGES5.index(b), runs=p_runs(q[(a, b)], "q"), color=P.BRACKET_STAGE)
                 for a, b in pairs]
        plots.brackets(ax, comps, tops, ymin=-4, key="sf1b" + tissue)
        rows += [dict(tissue=tissue, first=a, second=b, q=qq) for (a, b), qq in q.items()]
        f.insert(0, "tissue", tissue)
        source("B_" + tissue, f.rename(columns={"value": "malignant_PC_percent"}))
    source("B_tests", pd.DataFrame(rows))


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_clone_cnv(sf):
    rows = pd.read_csv(paths.cache("clones_rows.tsv"), sep="\t")
    bands = pd.read_csv(paths.cache("clones_bands.tsv"), sep="\t")["band"].astype(str).values
    chrom = np.array([int("".join(ch for ch in b.split("p")[0].split("q")[0] if ch.isdigit())) for b in bands])
    stages = ["MM", "HRSMM", "IRSMM", "LRSMM", "MGUS"]
    expected = {"MM": (84, 55), "HRSMM": (59, 42), "IRSMM": (70, 43), "LRSMM": (88, 50), "MGUS": (20, 10)}
    pcol = pd.read_csv(paths.cache("colors_patient_colors.tsv"), sep="\t").set_index("patient_id")["color"].to_dict()
    tracks = ["Progression", "HRD", "Translocation", "Clonality", "Patient"]
    x0, x1 = 22.0, 152.0
    tw = 2.2
    y = 3.0
    rh = 0.19                                                  # mm per clone
    cmap = ListedColormap([CNV[-1], CNV[0], CNV[1]])
    for s in stages:
        z = np.load(paths.cache("clones_%s.npz" % s))
        M, clones = z["cnv"].astype(int), [str(c) for c in z["clone"]]
        r = rows.set_index("clone").loc[clones]
        assert (len(clones), int((r["clonality"] == "subclonal").sum())) == expected[s], s
        h = len(clones) * rh
        ax = axes_mm(sf, x0, y, x1 - x0, h)
        rep = 3                                                # integer upsampling keeps the cell edges sharp at >= 300 ppi
        img = np.repeat(np.repeat(M + 1, rep, axis=1), max(1, int(np.ceil(300 / 25.4 * rh))), axis=0)
        im = ax.imshow(img, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="none",
                       extent=(0, M.shape[1], len(clones), 0))
        im.set_gid("raster-cnv-" + s)
        for b in np.where(np.diff(chrom) != 0)[0]:
            ln = ax.axvline(b + 1, color="black", lw=0.3, ls=(0, (2, 1.5)), zorder=3)
            ln.set_gid("ref-chr")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        at_mm(sf, 0.4, y + h / 2, [(s, "b")], "pn-stage-" + s, size=SMALL)
        at_mm(sf, 0.4, y + h / 2 + 2.2, "%d clones" % len(clones), "pn-n-" + s, size=SMALL)
        tvals = [[(P.LOW_MED_HIGH["High"] if v == 1 else ("#D9D9D9" if v == 0 else "#FFFFFF")) for v in r["progressor"]],
                 [P.CYTO["HRD"] if v == "HRD" else "#FFFFFF" for v in r["hrd_status"]],
                 [P.CYTO.get(GENE.get(str(v), ""), "#FFFFFF") if str(v) != "HRD" else "#FFFFFF" for v in r["cytogenetics"]],
                 ["#4D4D4D" if v == "clonal" else "#BDBDBD" for v in r["clonality"]],
                 [pcol.get(v, "#FFFFFF") for v in r["patient_id"]]]
        for k, colors in enumerate(tvals):
            ta = axes_mm(sf, x0 - (len(tracks) - k) * (tw + 0.3) - 0.4, y, tw, h)
            cols = list(dict.fromkeys(colors))
            m = ta.pcolormesh([0, 1], np.arange(len(colors) + 1), np.array([[cols.index(c)] for c in colors]),
                              cmap=ListedColormap(cols), vmin=-0.5, vmax=len(cols) - 0.5, edgecolors="face", linewidth=0)
            m.set_gid("mesh-track-%s%d" % (s, k))
            ta.set_ylim(len(colors), 0)
            ta.axis("off")
        y += h + 2.4
    # chromosome numbers under the last heatmap
    mids = [(np.where(chrom == c)[0].min() + np.where(chrom == c)[0].max() + 1) / 2 for c in range(1, 23)]
    below = plots.pts(ax, 0, -1.2, plots.blended_transform_factory(ax.transData, ax.transAxes))
    below2 = plots.pts(ax, 0, -6.4, plots.blended_transform_factory(ax.transData, ax.transAxes))
    for c, m in zip(range(1, 23), mids):                         # 21 is too narrow next to 20 and 22: on a second line
        ax.text(m, 0, str(c), transform=below2 if c == 21 else below, ha="center", va="top", fontsize=SMALL,
                gid="pn-chr%d" % c)
    at_mm(sf, (x0 + x1) / 2, y + 4.4, "Chromosome", "pn-chrlab", size=SMALL, ha="center")
    # track names (rotated, under the tracks) and keys
    for k, name in enumerate(tracks):
        xx = x0 - (len(tracks) - k) * (tw + 0.3) - 0.4 + tw / 2
        t = sf.text(xx / sf.w_mm, 1 - (y - 1.6) / sf.h_mm, name, rotation=90, ha="center", va="top", fontsize=SMALL,
                    transform=sf.transSubfigure)
        t.set_gid("pn-trk%d" % k)
    kx = x1 + 3.0
    keys = [("CNV", [("Gain", CNV[1]), ("Loss", CNV[-1]), ("Neutral", CNV[0])]),
            ("Progression", [("To MM", P.LOW_MED_HIGH["High"]), ("No", "#D9D9D9")]),
            ("Clonality", [("Clonal", "#4D4D4D"), ("Subclonal", "#BDBDBD")]),
            ("Translocation, HRD", [(g, P.CYTO[g]) for g in ["CCND1", "CCND2", "CCND3", "NSD2", "MAF", "MAFA", "MAFB", "Other",
                                                             "HRD"]])]
    yy = 3.5
    for title, items in keys:
        at_mm(sf, kx, yy, [(title, "b")], "pn-kt-" + title.replace(" ", ""), size=SMALL)
        yy += 2.1
        plots.key(sf, kx / sf.w_mm, 1 - (yy - 1.05) / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.3,
                  marker="s", row_pt=2.1 / PT_MM, gid="pn-k-" + title.replace(" ", ""))
        yy += 2.1 * len(items) + 1.0
    source("C_clones", rows)


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_flow(pn["flow"])
        panel_tumor_frac(pn["tumor_frac"])
        panel_clone_cnv(pn["clone_cnv"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("EDFig1", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
