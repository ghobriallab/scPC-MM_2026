#!/usr/bin/env python3
"""Supplementary Figure 4: cytogenetic classification at the single-cell level.

Purpose:      A, data used to call the cytogenetic group of the 182 patients (UpSet plot: a call counts for FISH and WGS
              when it names a group, for scRNA-seq when it equals the final call; patients of the group Other count for
              none; stage = most advanced stage at sampling). B, scaled expression (z-score per gene, clipped at 10) of
              the nine translocation partner genes in the malignant BM PCs of each patient, by cytogenetic group (stage,
              the most advanced in BM). C, the same values on the tumor-cell UMAP of Fig. 2C, all malignant PCs. D, the
              four patients with two clones of different cytogenetic groups.

Inputs:       data/paper: cytogenetics-status.csv, cohort-clinical-metadata.csv; results/figure_inputs/tumors_cells.npz.

Outputs:      figures/SuppFig4.svg, SuppFig4.pdf; figures/source_data/SuppFig4_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgba_array
from matplotlib.lines import Line2D

import palettes as P
import paths
import stage_tests as ST
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, SMALL, BASE, LW
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("SuppFig4", [
    Slot("assay_upset", "A", 0, 0, 183, 44),
    Slot("partner_boxes", "B", 0, 46, 183, 92),
    Slot("partner_umaps", "C", 0, 140, 135, 62),
    Slot("two_clone_umap", "D", 137, 140, 46, 62),
], max_height=247)
PT_MM = 25.4 / 72.0
GENE = {"t(11;14)": "CCND1", "t(12;14)": "CCND2", "t(6;14)": "CCND3", "t(4;14)": "NSD2", "t(14;16)": "MAF",
        "t(8;14)": "MAFA", "t(14;20)": "MAFB", "Other": "Other", "HRD": "HRD"}
GROUPS = ["CCND1", "CCND2", "CCND3", "NSD2", "MAF", "MAFA", "MAFB", "HRD", "Other"]
STAGES5 = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig4_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center"):
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs if isinstance(runs, list) else [(runs, "")], size=size, ha=ha,
                     va=va, transform=sf.transSubfigure, gid=gid)


# ---- A ---------------------------------------------------------------------------------------------------------------
def panel_upset(sf):
    a = pd.read_csv(paths.inputs("paper", "cytogenetics-status.csv"), index_col=0)
    a = a.loc[a.index.str.endswith("tumor:0")]
    fish = set(a.loc[~a["FISH"].isin(["Unclassified", "Unavailable"]), "patient_id"])
    wgs = set(a.loc[~a["WGS"].isin(["Other", "Unavailable"]), "patient_id"])
    scr = set(a.loc[(a["scRNA"] != "no_translocation_detected") & (a["scRNA"] == a["cytogenetics"]), "patient_id"])
    pts = sorted(set(a["patient_id"]))
    assert len(pts) == 182 and len(fish | wgs | scr) == 168
    meta = pd.read_csv(paths.inputs("paper", "cohort-clinical-metadata.csv"))
    meta = meta[meta["disease_status"].isin(STAGES5)].copy()
    meta["disease_status"] = ST.imwg_stages(meta["patient_id"], meta["sample_type"], meta["disease_status"])   # IMWG groups
    first = meta.groupby("patient_id")["disease_status"].agg(lambda s: max(s, key=STAGES5.index))   # most advanced stage
    d = pd.DataFrame({"patient_id": pts, "FISH": [p in fish for p in pts], "WGS": [p in wgs for p in pts],
                      "scRNA-seq": [p in scr for p in pts]})
    d["stage"] = d["patient_id"].map(first)
    assert d["stage"].notna().all()
    sets = ["FISH", "WGS", "scRNA-seq"]
    combos = d.groupby(sets).size().sort_values(ascending=False, kind="stable")       # sorted by cardinality
    n = len(combos)
    rh = 3.6
    am = axes_mm(sf, 22.0, 6.0, 14.0, n * rh)                   # dot matrix
    ab = axes_mm(sf, 38.0, 6.0, 60.0, n * rh, sharey=am)        # stacked bars by stage
    for r, (combo, cnt) in enumerate(combos.items()):
        y = r
        on = [k for k, v in enumerate(combo) if v]
        am.plot(range(3), [y] * 3, ls="none", marker="o", ms=3.0, mfc="#DDDDDD", mec="none", zorder=1)
        if on:
            am.plot(on, [y] * len(on), ls="none", marker="o", ms=3.0, mfc="#1A1A1A", mec="none", zorder=3)
            if len(on) > 1:
                am.plot([min(on), max(on)], [y, y], color="#1A1A1A", lw=1.0, zorder=2)
        sub = d[(d[sets] == list(combo)).all(1)]
        left = 0
        for s in STAGES5:
            k = int((sub["stage"] == s).sum())
            if k:
                b = ab.barh(y, k, left=left, height=0.62, color=P.TUMOR[s], edgecolor="none", zorder=2)
                b[0].set_gid("bar")
                left += k
        rich_text(ab, left, y, [("%d" % cnt, "")], size=SMALL, ha="left", va="center", transform=plots.pts(ab, 1.2, 0),
                  gid="ax-cnt%d" % r)
    am.set_xlim(-0.6, 2.6)
    am.set_ylim(n - 0.5, -0.5)
    am.set_xticks(range(3))
    am.set_xticklabels(sets, rotation=90, fontsize=SMALL)
    am.set_yticks([])
    am.tick_params(axis="x", length=0, pad=1.0)
    for sp in am.spines.values():
        sp.set_visible(False)
    ab.set_xlim(0, combos.max() * 1.12)
    ab.tick_params(axis="y", left=False, labelleft=False)
    ab.tick_params(axis="x", labelsize=SMALL)
    ab.set_xlabel("Patients", labelpad=1.0)
    for sp in ("top", "right", "left"):
        ab.spines[sp].set_visible(False)
    at_mm(sf, 29.0, 2.4, "Data used", "pn-atitle", size=SMALL, ha="center")
    at_mm(sf, 38.0, 2.4, [("Patients by disease stage (", ""), ("n", "i"), (" = 182)", "")], "pn-btitle", size=SMALL)
    empty = [r for r, combo in enumerate(combos.index) if not any(combo)]
    assert len(empty) == 1 and combos.iloc[empty[0]] == 14       # the 14 patients of group Other: no assay named a group
    at_mm(sf, 21.0, 6.0 + (empty[0] + 0.5) * rh, "None (group Other)", "pn-anone", size=SMALL, ha="right")
    plots.key(sf, 104.0 / sf.w_mm, 1 - 6.0 / sf.h_mm, [(s, P.TUMOR[s]) for s in STAGES5], transform=sf.transSubfigure,
              size=SMALL, ms=2.3, marker="s", row_pt=2.1 / PT_MM, gid="pn-kA")
    source("A", d)


# ---- B ---------------------------------------------------------------------------------------------------------------
def spread(xs, ws, gap, lo, hi):
    """Positions as close as possible to xs (increasing) such that neighbors are at least (w_i + w_j) / 2 + gap apart
    and every label stays within [lo, hi]."""
    pos = np.array(xs, float)
    for _ in range(500):
        moved = False
        for i in range(len(pos) - 1):
            need = (ws[i] + ws[i + 1]) / 2.0 + gap - (pos[i + 1] - pos[i])
            if need > 1e-9:
                pos[i] -= need / 2.0
                pos[i + 1] += need / 2.0
                moved = True
        pos[0] = max(pos[0], lo + ws[0] / 2.0)
        pos[-1] = min(pos[-1], hi - ws[-1] / 2.0)
        if not moved:
            break
    assert all(pos[i + 1] - pos[i] >= (ws[i] + ws[i + 1]) / 2.0 + gap - 1e-6 for i in range(len(pos) - 1))
    return pos


def group_labels(ax, bounds, counts, mm_per, rh, lift_pt=8.0):
    """Group names above the first row, in one line, each joined by a thin line to the center of its columns."""
    tr = plots.blended_transform_factory(ax.transData, ax.transAxes)
    fig = plots.root(ax)
    runs = [[("%s (%d)" % (g, counts[g]), "")] for g in GROUPS]
    widths = [rich_width_pt(fig, r, SMALL) * PT_MM / mm_per for r in runs]           # in columns (patients)
    centers = [(lo + hi - 1) / 2.0 for lo, hi in zip(bounds[:-1], bounds[1:])]
    pos = spread(centers, widths, 1.6 / mm_per, -0.5, bounds[-1] - 0.5)
    up = lambda pt: 1 + pt * PT_MM / rh                                              # points above the plot, in axes units
    for g, c, x, r in zip(GROUPS, centers, pos, runs):
        rich_text(ax, x, 1, r, size=SMALL, ha="center", va="bottom", transform=plots.pts(ax, 0, lift_pt, tr),
                  gid="pn-grp-" + g)
        ln = Line2D([c, x], [up(0.6), up(lift_pt - 0.9)], color="black", lw=0.3, transform=tr, clip_on=False)
        ln.set_gid("ref-leader-" + g)
        ax.add_line(ln)


def tumor_cells():
    z = np.load(paths.cache("tumors_cells.npz"), allow_pickle=True)
    cat = lambda k: np.where(z[k + "__codes"] < 0, "nan", z[k + "__cats"].astype(str)[np.maximum(z[k + "__codes"], 0)])
    obs = pd.DataFrame({k: cat(k) for k in ("patient_id", "sample_type", "disease_status", "tumor_id", "cytogenetics",
                                              "hrd_status")})
    genes = [str(g) for g in z["partner_genes"]]
    return z["umap"], obs, z["partner_expr"], genes


def panel_partner_boxes(sf, obs, X, genes):
    keep = ((obs["sample_type"] == "BM") & ~obs["tumor_id"].str.endswith("1")).values           # BM, main tumors
    d = obs[keep].reset_index(drop=True)
    E = X[keep]
    d["disease_status"] = ST.imwg_stages(d["patient_id"], d["sample_type"], d["disease_status"])   # IMWG risk groups
    top = d.groupby("patient_id")["disease_status"].agg(lambda s: max(s.unique(), key=STAGES5.index))
    d["disease_status"] = d["patient_id"].map(top)                     # one stage per patient: the highest
    d["group"] = pd.Categorical(d["cytogenetics"].map(GENE), categories=GROUPS)
    Pt = d[["patient_id", "group"]].drop_duplicates().sort_values("group", kind="stable")
    order = list(Pt["patient_id"])
    counts = Pt["group"].value_counts().reindex(GROUPS)
    assert counts.to_dict() == {"CCND1": 45, "CCND2": 2, "CCND3": 4, "NSD2": 15, "MAF": 14, "MAFA": 4, "MAFB": 8, "HRD": 73,
                                "Other": 14}                     # 179 patients
    stage_of = d.drop_duplicates("patient_id").set_index("patient_id")["disease_status"]
    idx = d.groupby("patient_id").indices
    bounds = np.cumsum([0] + list(counts.values))
    x0, x1 = 16.0, 181.0
    rh, gap = 6.8, 1.2
    y0 = 12.5
    for gi, g in enumerate(genes):
        ax = axes_mm(sf, x0, y0 + gi * (rh + gap), x1 - x0, rh)
        vals = [E[idx[p], gi] for p in order]
        bp = ax.boxplot(vals, positions=np.arange(len(order)), widths=0.75, patch_artist=True, showfliers=False,
                        showcaps=False, whis=1.5, manage_ticks=False,
                        medianprops=dict(color="black", linewidth=0.3), whiskerprops=dict(color="#333333", linewidth=0.3),
                        boxprops=dict(linewidth=0.3, edgecolor="#333333"))
        for k, box in enumerate(bp["boxes"]):
            box.set_facecolor(P.TUMOR[stage_of[order[k]]])
            box.set_gid("box-sf4b-%d-%d" % (gi, k))
        ymax = float(np.ceil(E[:, gi].max()) + 1)
        ax.set_ylim(-0.6, ymax)
        ax.set_yticks([0, ymax])
        ax.set_yticklabels(["0", "%g" % ymax])
        ax.tick_params(axis="y", labelsize=SMALL, length=1.5, pad=1.0)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_xticks([])
        for b in bounds[1:-1]:
            ln = ax.axvline(b - 0.5, color="black", lw=0.4, zorder=1)
            ln.set_gid("ref-group")
        rich_text(ax, 0, 0.5, [(g, "")], size=SMALL, ha="right", va="center",
                  transform=plots.pts(ax, -9.0, 0, ax.transAxes), gid="pn-gene%d" % gi)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if gi == 0:
            group_labels(ax, bounds, counts, (x1 - x0) / len(order), rh)
    at_mm(sf, x0 + (x1 - x0) / 2, 2.3, [("Cytogenetic group of the patient (", "b"), ("n", "bi"), (", patients)", "b")],
          "pn-btitle", size=SMALL, ha="center")
    at_mm(sf, x0 - 3.2, y0 - 1.9, [("Gene", "b")], "pn-genehdr", size=SMALL, ha="right")
    yb = y0 + len(genes) * (rh + gap) - gap
    t = sf.text(1.6 / sf.w_mm, 1 - (y0 + (yb - y0) / 2) / sf.h_mm, "Scaled expression (z-score)", rotation=90, ha="center",
                va="center", fontsize=SMALL, transform=sf.transSubfigure)
    t.set_gid("pn-ylab")
    at_mm(sf, x0 + (x1 - x0) / 2, yb + 2.4, "Patients (one box per patient, malignant BM PCs)", "pn-xlab", size=SMALL,
          ha="center")
    items = [(s, P.TUMOR[s]) for s in STAGES5]
    at_mm(sf, x0 + 30.0, yb + 5.6, "Disease stage", "pn-kst", size=SMALL, ha="right")
    plots.key(sf, (x0 + 31.0) / sf.w_mm, 1 - (yb + 4.6) / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.3,
              marker="s", ncol=5, row_pt=2.0 / PT_MM, gid="pn-kB")
    med = pd.DataFrame([dict(patient_id=p, group=str(Pt.set_index("patient_id").loc[p, "group"]), stage=stage_of[p],
                             **{g: float(np.median(E[idx[p], gi])) for gi, g in enumerate(genes)}) for p in order])
    source("B_patient_medians", med)


# ---- C, D ------------------------------------------------------------------------------------------------------------
def panel_partner_umaps(sf, xy, X, genes):
    for gi, g in enumerate(genes):
        r, c = divmod(gi, 5)
        x, y = 1.0 + c * 27.0, 3.5 + r * 29.0
        ax = axes_mm(sf, x, y + 2.6, 20.0, 20.0)
        v = X[:, gi]
        norm = Normalize(float(v.min()), float(v.max()))
        order = np.argsort(v, kind="stable")
        plots.umap(ax, xy, plots.matplotlib_cm("viridis")(norm(v)), "sf4c-" + g, s=0.01, order=order, labels=False)
        rich_text(ax, 0.5, 1, [(g, "")], size=BASE, ha="center", va="bottom", transform=plots.pts(ax, 0, 1.0, ax.transAxes),
                  gid="pn-c%d" % gi)
        cax = axes_mm(sf, x + 20.8, y + 5.1, 1.0, 15.0)
        cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap="viridis"), cax,
                            ticks=plots.range_ticks(float(v.min()), float(v.max())))
        plots.plain_ticks(cb.ax.yaxis)
    x, y = 1.0 + 4 * 27.0, 3.5 + 29.0 + 9.0
    for k, line in enumerate(["Color: scaled expression", "(z-score) of the gene;", "each scale spans the", "range of that gene"]):
        at_mm(sf, x, y + k * 2.4, line, "pn-cnote%d" % k, size=SMALL)


def panel_two_clones(sf, xy, obs):
    cyto = obs["cytogenetics"].map(lambda x: GENE.get(x, "nan")).values
    cy = np.where((obs["hrd_status"].values == "Y") & (cyto != "HRD"), np.char.add(cyto.astype(str), "+HRD"), cyto)
    lab = np.char.add(np.char.add(obs["patient_id"].values.astype(str), ": "), cy.astype(str))
    pal = {"Pt-129: MAF": "#0072B2", "Pt-129: HRD": "#56B4E9", "Pt-141: CCND1": "#9E3D7A", "Pt-141: CCND1+HRD": "#D98CBE",
           "Pt-155: CCND1": "#006B4F", "Pt-155: Other": "#3CC79A", "Pt-185: CCND1": "#B34700", "Pt-185: MAFB": "#F09A4E"}
    hit = np.isin(lab, list(pal))
    assert set(lab[hit]) == set(pal), set(pal) - set(lab[hit])
    ax = axes_mm(sf, 3.0, 4.0, 40.0, 40.0)
    plots.umap(ax, xy[~hit], "#E3E3E3", "sf4d-bg", s=0.01, labels=False)
    col = to_rgba_array([pal[v] for v in lab[hit]])
    col[:, 3] = 0.9
    sc = ax.scatter(xy[hit, 0], xy[hit, 1], s=0.4, c=col, linewidths=0, rasterized=True, zorder=2)
    sc.set_gid("raster-sf4d-hit")
    ax.set_title("Patients with two clones", fontsize=BASE, pad=2.0)
    plots.key(sf, 3.0 / sf.w_mm, 1 - 46.0 / sf.h_mm, [(k, v) for k, v in pal.items()], transform=sf.transSubfigure,
              size=SMALL, ms=2.3, ncol=2, col_pt=21.0 / PT_MM, row_pt=2.0 / PT_MM, gid="pn-kD")
    plots.key(sf, 3.0 / sf.w_mm, 1 - 54.6 / sf.h_mm, [("Other malignant PCs", "#E3E3E3")], transform=sf.transSubfigure,
              size=SMALL, ms=2.3, row_pt=2.0 / PT_MM, gid="pn-kD2")
    source("D_cells", pd.Series(lab[hit]).value_counts().rename_axis("clone").reset_index(name="cells"))


def main():
    xy, obs, X, genes = tumor_cells()
    assert genes == ["CCND1", "CCND2", "CCND3", "NSD2", "FGFR3", "MAF", "MAFA", "MAFB", "MYC"]
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_upset(pn["assay_upset"])
        panel_partner_boxes(pn["partner_boxes"], obs, X, genes)
        panel_partner_umaps(pn["partner_umaps"], xy, X, genes)
        panel_two_clones(pn["two_clone_umap"], xy, obs)
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig4", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
