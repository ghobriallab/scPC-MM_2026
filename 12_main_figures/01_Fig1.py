#!/usr/bin/env python3
"""Figure 1: generation of a plasma cell atlas using scRNA-seq and scBCR-seq.

Purpose:      A, study design: individuals sampled in each tissue by disease stage, each individual counted once, and the
              two single-cell scores. B, cohort overview of the 226 individuals (one column each: disease stage, tissue,
              progression, clinical tracks, cytogenetic group, isotype, numbers of PCs); an individual with two values in
              a track gets a cell split into an upper and a lower half; gray, not available. C, the 1,649,290 PCs on the
              UMAP of the atlas colored by individual, plasma cell status, disease stage, isotype and cytogenetic group
              (cells without a value in light gray, drawn first).

Inputs:       results/cohort/cohort_by_stage.tsv; results/figure_inputs: cohort_tracks.tsv, cohort_isotypes.tsv,
              cohort_counts.tsv, colors_patient_colors.tsv, pcs_cells.npz.

Outputs:      figures/Fig1.svg, Fig1.pdf; figures/source_data/Fig1_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths, stage_tests).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter

import icons
import palettes as P
import paths
import plots
import stage_tests as ST
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, SMALL, BASE, LW
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("Fig1", [
    Slot("design", "A", 0, 0, 183, 46),
    Slot("cohort", "B", 0, 48, 183, 68),
    Slot("umaps", "C", 0, 118, 183, 52),
], max_height=170)
PT_MM = 25.4 / 72.0
STAGES = ST.STAGES
GENE = {"t(11;14)": "CCND1", "t(12;14)": "CCND2", "t(6;14)": "CCND3", "t(4;14)": "NSD2", "t(14;16)": "MAF",
        "t(8;14)": "MAFA", "t(14;20)": "MAFB", "Other": "Other", "None": "Other", "HRD": "HRD"}
TRANSLOCATION = ["CCND1", "CCND2", "CCND3", "NSD2", "MAF", "MAFA", "MAFB", "Other"]
ISOTYPES = ["IgG", "IgA", "IgD", "IgM", "IgM/IgD", "IgE", "Light chain only"]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/Fig1_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center"):
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs if isinstance(runs, list) else [(runs, "")], size=size, ha=ha,
                     va=va, transform=sf.transSubfigure, gid=gid)


# ---- A ---------------------------------------------------------------------------------------------------------------
def panel_design(sf):
    c = pd.read_csv(paths.result("cohort", "cohort_by_stage.tsv"), sep="\t").set_index("stage")
    counts = {t: {s: int(c.loc["SMM (any risk)" if s == "SMM" else s, "individuals_with_" + t])
                  for s in ["MM", "SMM", "MGUS", "Donor"]} for t in ("BM", "PB")}
    n, donors = int(c.loc["All", "individuals"]), int(c.loc["Donor", "individuals"])
    pcs = int(c.loc["All", "malignant_PCs"] + c.loc["All", "normal_PCs"])
    assert sum(counts["BM"].values()) == 220 and sum(counts["PB"].values()) == 90
    assert (n, donors, pcs) == (226, 29, 1649290)                                # as the Results
    icons.study_design(sf, counts, n, donors, pcs)
    source("design", pd.DataFrame([dict(tissue=t, stage=s, individuals=n) for t in counts for s, n in counts[t].items()]))


# ---- B ---------------------------------------------------------------------------------------------------------------
def cohort_tables():
    t = pd.read_csv(paths.cache("cohort_tracks.tsv"), sep="\t")
    iso = pd.read_csv(paths.cache("cohort_isotypes.tsv"), sep="\t").set_index("patient_id")
    cnt = pd.read_csv(paths.cache("cohort_counts.tsv"), sep="\t").set_index("patient_id")
    st = t[t["track"] == "Disease stage"].copy()
    st["k"] = st["value"].map(STAGES.index)
    top = st.groupby("patient_id")["k"].max()
    ncell = cnt.sum(1)
    order = pd.DataFrame({"k": top, "n": ncell.reindex(top.index)}).sort_values(["k", "n"], ascending=[True, False]).index
    per = pd.Series([STAGES[k] for k in top.reindex(order)]).value_counts().reindex(STAGES).tolist()
    assert len(order) == 226 and per == [29, 36, 56, 41, 23, 41], per          # by most advanced stage
    prog = t.loc[t["track"] == "Progression", "value"].value_counts().to_dict()
    assert prog == {"MM": 13, "LRSMM": 8, "IRSMM": 4, "HRSMM": 2}, prog          # 27 patients (legend)
    return t, iso, cnt, list(order)


def criteria_by_stage(get):
    """2/20/20 criteria met per individual and SMM risk group: {criterion: {patient: {stage: "Yes"/"No"}}}; where all
    three values are known, the number met agrees with the risk group (0 low, 1 intermediate, 2-3 high risk)."""
    out = {}
    for name, cut in (("FLC ratio", 20), ("M spike", 2), ("BMBx %", 20)):
        out[name] = {}
        for p, v in get(name + " by stage").items():
            out[name][p] = {x.split("|")[0]: ("Yes" if float(x.split("|")[1]) > cut else "No") for x in v}
    need = {"LRSMM": (0,), "IRSMM": (1,), "HRSMM": (2, 3)}
    for p in out["BMBx %"]:
        for st in out["BMBx %"][p]:
            if all(st in out[c].get(p, {}) for c in out):
                n_met = sum(out[c][p][st] == "Yes" for c in out)
                assert n_met in need[st], (p, st, n_met)
    return out


def cells_track(ax, order, values, colors, y, h, key, w=0.9):
    """One categorical track: a cell per individual, split into an upper and a lower half when it has two values."""
    polys, cols = [], []
    for i, p in enumerate(order):
        v = values.get(p, [])
        x0, x1 = i - w / 2, i + w / 2
        if len(v) == 1:
            polys.append([(x0, y), (x1, y), (x1, y + h), (x0, y + h)])
            cols.append(colors[v[0]])
        elif len(v) == 2:
            polys += [[(x0, y), (x1, y), (x1, y + h / 2), (x0, y + h / 2)], [(x0, y + h / 2), (x1, y + h / 2), (x1, y + h),
                                                                            (x0, y + h)]]
            cols += [colors[v[0]], colors[v[1]]]
        elif len(v) > 2:
            raise ValueError("more than two values for one individual in track %s" % key)
    pc = PolyCollection(polys, facecolors=cols, edgecolors="none", zorder=2)
    pc.set_gid("track-" + key)
    ax.add_collection(pc)


def scale_track(ax, order, values, cmap, vmin, vmax, y, h, key):
    cm = plots.matplotlib_cm(cmap)
    vals = {p: [cm(min(1.0, max(0.0, (float(v[0]) - vmin) / (vmax - vmin))))] for p, v in values.items()}
    polys, cols = [], []
    for i, p in enumerate(order):
        if p in vals:
            polys.append([(i - 0.45, y), (i + 0.45, y), (i + 0.45, y + h), (i - 0.45, y + h)])
            cols.append(vals[p][0])
    pc = PolyCollection(polys, facecolors=cols, edgecolors="none", zorder=2)
    pc.set_gid("track-" + key)
    ax.add_collection(pc)


def panel_cohort(sf):
    t, iso, cnt, order = cohort_tables()
    get = lambda name: {p: list(g["value"]) for p, g in t[t["track"] == name].groupby("patient_id", sort=False)}
    n = len(order)
    x0, x1 = 17.0, 128.5                                   # tracks; labels left of x0; scales right of x1
    # bar tracks: PC counts (stacked by tissue and PC status) and isotype fractions
    ac = axes_mm(sf, x0, 1.5, x1 - x0, 13.0)
    stack = [("BM:tumor", "BM malignant"), ("PB:tumor", "PB malignant"), ("BM:normal", "BM normal"),
             ("PB:normal", "PB normal")]
    bottom = np.zeros(n)
    c = cnt.reindex(order).fillna(0)
    assert int(c.values.sum()) == 1649290
    for col, lab in stack:
        v = c[col].values
        bars = ac.bar(np.arange(n), v, bottom=bottom, width=0.9, color=P.PC_STATUS[lab], linewidth=0, zorder=2)
        for b in bars:
            b.set_gid("bar")
        bottom += v
    ac.set_xlim(-0.5, n - 0.5)
    ac.set_ylim(0, 37000)
    ac.set_yticks([0, 10000, 20000, 30000])
    ac.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:,.0f}".format(v)))
    ac.yaxis.tick_right()
    ac.tick_params(axis="y", labelsize=SMALL, length=1.5, pad=1.0)
    ac.set_xticks([])
    for s_ in ("top", "left", "bottom"):
        ac.spines[s_].set_visible(False)
    rich_text(ac, 0, 0.5, [("PCs", "")], size=SMALL, ha="right", va="center", transform=plots.pts(ac, -2.0, 0, ac.transAxes),
              gid="pn-lab-pcs")
    ai = axes_mm(sf, x0, 17.0, x1 - x0, 5.5, sharex=ac)
    fr = iso.reindex(order)[ISOTYPES].fillna(0)
    bottom = np.zeros(n)
    for k in ISOTYPES:
        v = fr[k].values
        bars = ai.bar(np.arange(n), v, bottom=bottom, width=0.9, color=P.ISOTYPE[k], linewidth=0, zorder=2)
        for b in bars:
            b.set_gid("bar")
        bottom += v
    ai.set_ylim(0, 1)
    ai.set_yticks([0, 1])
    ai.yaxis.tick_right()
    ai.tick_params(axis="y", labelsize=SMALL, length=1.5, pad=1.0)
    ai.tick_params(axis="x", length=0, labelbottom=False)
    for s_ in ("top", "left", "bottom"):
        ai.spines[s_].set_visible(False)
    rich_text(ai, 0, 0.5, [("Isotype fraction", "")], size=SMALL, ha="right", va="center",
              transform=plots.pts(ai, -2.0, 0, ai.transAxes), gid="pn-lab-iso")
    # thin tracks, top to bottom
    tracks = [("PB tumor fraction", "scale", ("Blues", 0, 1)), ("BM tumor fraction", "scale", ("Reds", 0, 1)),
              ("FLC ratio > 20", "crit", ("FLC ratio", 20)), ("M-spike > 2 g/dL", "crit", ("M spike", 2)),
              ("BMBx > 20%", "crit", ("BMBx %", 20)), ("FLC ratio", "scale", ("Oranges", 0, 400)),
              ("M-spike (g/dL)", "scale", ("Greens", 0, 5)), ("BMBx (%)", "scale", ("Purples", 0, 80)),
              ("HRD", "cat", None), ("Translocation", "cat", None), ("Tissue", "cat", None),
              ("Age (years)", "scale", ("viridis", 31, 89)),
              ("Progression", "cat", None), ("Disease stage", "cat", None)]
    at = axes_mm(sf, x0, 24.0, x1 - x0, 42.5, sharex=ac)
    at.set_ylim(len(tracks), 0)
    at.axis("off")
    frac = {}
    for tissue in ("BM", "PB"):
        tot = c[tissue + ":tumor"] + c[tissue + ":normal"]
        frac[tissue] = {p: [float(c.loc[p, tissue + ":tumor"] / tot[p])] for p in order if tot[p] > 0}
    smm = {p for p, v in get("Disease stage").items() if any(s in ("LRSMM", "IRSMM", "HRSMM") for s in v)}
    clinical = {"FLC ratio": get("FLC ratio"), "M spike": get("M spike"), "BMBx %": get("BMBx %")}
    crit = criteria_by_stage(get)
    h = 0.82
    for r, (lab, kind, arg) in enumerate(tracks):
        y = r + (1 - h) / 2
        key = lab.split(" ")[0].lower() + str(r)
        if lab in ("PB tumor fraction", "BM tumor fraction"):
            scale_track(at, order, frac[lab[:2]], arg[0], arg[1], arg[2], y, h, key)
        elif kind == "scale":
            src = {"FLC ratio": "FLC ratio", "M-spike (g/dL)": "M spike", "BMBx (%)": "BMBx %", "Age (years)": "Age"}[lab]
            scale_track(at, order, get(src) if src == "Age" else clinical[src], arg[0], arg[1], arg[2], y, h, key)
        elif kind == "crit":                                     # SMM only: value above the IMWG 2/20/20 cut-off,
            vals = {p: [d[st] for st in sorted(d, key=STAGES.index)]    # one per risk group (split as the stage track)
                    for p, d in crit[arg[0]].items() if p in smm}
            cells_track(at, order, vals, {"Yes": "#1A1A1A", "No": "#D3D3D3"}, y, h, key)
        elif lab == "HRD":
            cells_track(at, order, get("HRD status"), {"HRD": P.CYTO["HRD"]}, y, h, key)
        elif lab == "Translocation":
            vals = {p: [GENE[x] for x in v] for p, v in get("Translocation").items()}
            cells_track(at, order, vals, P.CYTO, y, h, key)
        elif lab == "Tissue":
            vals = {p: sorted(v) for p, v in get("Sample type").items()}
            cells_track(at, order, vals, P.TISSUE, y, h, key)
        elif lab == "Progression":
            cells_track(at, order, get("Progression"), P.STAGE, y, h, key)
        elif lab == "Disease stage":
            vals = {p: sorted(v, key=STAGES.index) for p, v in get("Disease stage").items()}
            cells_track(at, order, vals, P.STAGE, y, h, key)
        rich_text(at, 0, r + 0.5, [(lab, "")], size=SMALL, ha="right", va="center",
                  transform=plots.pts(at, -2.0, 0, plots.blended_transform_factory(at.transAxes, at.transData)),
                  gid="pn-trk%d" % r)
    # color scales of the continuous tracks, on their rows at the right
    tr_y = lambda r: 24.0 + (r + 0.5) * 42.5 / len(tracks)
    for r, (lab, kind, arg) in enumerate(tracks):
        if kind != "scale":
            continue
        cmap, vmin, vmax = arg
        yy = tr_y(r)
        cax = axes_mm(sf, x1 + 5.2, yy - 0.75, 7.0, 1.5)
        import matplotlib.cm as mcm
        from matplotlib.colors import Normalize
        cb = plots.colorbar(sf, mcm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), cax, orientation="horizontal",
                            ticks=[])
        lo = "%g" % vmin
        hi = ("%g" % vmax) + ("+" if lab == "FLC ratio" else "")
        at_mm(sf, x1 + 4.6, yy, lo, "pn-sc-lo%d" % r, size=SMALL, ha="right")
        at_mm(sf, x1 + 12.8, yy, hi, "pn-sc-hi%d" % r, size=SMALL, ha="left")
    # categorical keys
    lx1, lx2 = 147.5, 166.0
    groups1 = [("Disease stage", [(s, P.STAGE[s]) for s in STAGES]), ("Tissue", [(s, P.TISSUE[s]) for s in ("BM", "PB")]),
               ("2/20/20 criterion", [("Yes", "#1A1A1A"), ("No", "#D3D3D3")]),
               ("PCs", [(s, P.PC_STATUS[s]) for s in ("BM malignant", "PB malignant", "BM normal", "PB normal")]),
               ("Progression (stage reached)", [(s, P.STAGE[s]) for s in ("LRSMM", "IRSMM", "HRSMM", "MM")]),
               ("Clinical tracks", [("Not available", "#FFFFFF")])]
    groups2 = [("Translocation", [(g, P.CYTO[g]) for g in TRANSLOCATION]), ("HRD", [("HRD", P.CYTO["HRD"])]),
               ("Isotype", [(k, P.ISOTYPE[k]) for k in ISOTYPES])]
    row = 2.05
    for gx, groups in ((lx1, groups1), (lx2, groups2)):
        y = 2.0
        for title, items in groups:
            at_mm(sf, gx, y, [(title, "b")], "pn-kt-%s" % title.replace(" ", "").replace("/", ""), size=SMALL)
            y += row
            plots.key(sf, gx / sf.w_mm, 1 - (y - row / 2) / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.3,
                      marker="s", row_pt=row / PT_MM, gid="pn-k-%s" % title.replace(" ", "").replace("/", ""))
            y += row * len(items) + 0.9
        assert y <= sf.h_mm + 0.5, (title, y)
    source("cohort_counts", c.reset_index().rename(columns={"index": "patient_id"}))


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_umaps(sf):
    z = np.load(paths.cache("pcs_cells.npz"))
    xy = z["umap"]
    cat = lambda k: np.where(z[k + "__codes"] < 0, "nan", z[k + "__cats"].astype(str)[np.maximum(z[k + "__codes"], 0)])
    pt, tissue, ann = cat("patient_id"), cat("sample_type"), cat("annotation")
    stage = cat("disease_status")
    stage = np.array(ST.imwg_stages(pt, tissue, stage))           # IMWG 2/20/20 risk groups (risk.py)
    status = np.char.add(np.char.add(tissue, " "), np.where(ann == "tumor", "malignant", "normal"))
    iso = cat("isotype_status")
    iso = np.where(iso == "None", "Light chain only", iso)
    cyto = np.array([GENE.get(x, "nan") for x in cat("cytogenetics")])
    hrd = cat("hrd_status")
    cy = np.where((hrd == "Y") & (cyto != "HRD") & (cyto != "nan"), np.char.add(cyto, "+HRD"), cyto)
    pcol = pd.read_csv(paths.cache("colors_patient_colors.tsv"), sep="\t").set_index("patient_id")["color"].to_dict()
    cyto_cols = dict(P.CYTO, **P.CYTO_HRD)
    cyto_order = [g for k in TRANSLOCATION if k != "Other" for g in (k, k + "+HRD") if g in set(cy)] + ["HRD", "Other"]
    na = "#D3D3D3"                                  # cells without a value (scanpy's na_color), drawn first
    panels = [([("Individual (", ""), ("n", "i"), (" = 226)", "")], pt, pcol, None, 1.0, None),
              ([("Plasma cell status", "")], status, P.PC_STATUS, ["BM normal", "PB normal", "BM malignant",
                                                                   "PB malignant"], 1.0, None),
              ([("Disease stage", "")], stage, P.STAGE, STAGES, 1.0, None),
              ([("Isotype", "")], iso, P.ISOTYPE, ISOTYPES, 1.0, "No scBCR-seq"),
              ([("Cytogenetic group", "")], cy, cyto_cols, cyto_order, 0.6, "Normal plasma cells")]
    assert len(set(pt)) == 226
    W = 183.0 / 5
    for k, (title, lab, pal, order, alpha, na_lab) in enumerate(panels):
        x = 1.5 + k * W + 2.4
        ax = axes_mm(sf, x, 4.0, 28.5, 28.5)
        missing = np.array([v not in pal for v in lab])
        col = np.array([pal.get(v, na) for v in lab])
        if alpha < 1:
            from matplotlib.colors import to_rgba_array
            col = to_rgba_array(col)
            col[:, 3] = alpha
        draw = np.r_[np.where(missing)[0], np.where(~missing)[0]]
        plots.umap(ax, xy, col, "fig1c-%d" % k, s=0.02, labels=(k == 0), order=draw)
        rich_text(ax, 0.5, 1, title, size=BASE, ha="center", va="bottom", transform=plots.pts(ax, 0, 2.0, ax.transAxes),
                  gid="pn-ct%d" % k)
        if order:
            items = [(o, pal[o]) for o in order] + ([(na_lab, na)] if na_lab and missing.any() else [])
            ncol = 2
            nrow = int(np.ceil(len(items) / ncol))
            width = lambda its: max(rich_width_pt(sf, [(o, "")], SMALL) for o, _ in its) + 2.3 + 1.8
            colw = width(items[:nrow]) + 2.2                     # the last column may be as wide as the panel allows
            assert (colw + width(items[nrow:])) * PT_MM <= W - 1.0, (title, (colw + width(items[nrow:])) * PT_MM)
            plots.key(sf, x / sf.w_mm, 1 - 34.0 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.3,
                      ncol=ncol, col_pt=colw, row_pt=2.0 / PT_MM, gid="pn-ck%d" % k)
            assert 34.0 + nrow * 2.0 <= sf.h_mm


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_design(pn["design"])
        panel_cohort(pn["cohort"])
        panel_umaps(pn["umaps"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("Fig1", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
