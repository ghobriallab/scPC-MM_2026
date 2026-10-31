#!/usr/bin/env python3
"""Figure 2: transcriptional heterogeneity of sixteen known therapeutic targets by cytogenetic group.

Purpose:      A, per BM tumor, the log2 fold change of each target between malignant and normal PCs where significant
              (q <= 0.01 and |log2 fold change| >= 1), columns ordered by cytogenetic group with Ward clustering within
              each group, rows by complete-linkage clustering on the cosine distance. B, pseudobulk log2 fold changes by
              disease stage and by cytogenetic group. C, malignant BM PCs on the tumor-cell UMAP by cytogenetic group
              ("+HRD" groups as a darker shade). D-G, per malignant PC, the significant log2 fold change of its tumor for
              GPRC5D, TNFRSF17, FCRL5 and SLAMF7 (white, not tested).

Inputs:       results/differential_expression (per-tumor tables via figlib/targets.py;
              disease-tumor-matched-imwg-risk-degs.csv); data/paper/degs/pseudobulk/cytogenetics-tumor-matched-degs.csv;
              results/figure_inputs/tumors_cells.npz. SMM risk groups of the IMWG 2/20/20 model (risk.py).

Outputs:      figures/Fig2.svg, Fig2.pdf; figures/source_data/Fig2_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.colors import TwoSlopeNorm, to_rgba_array

import palettes as P
import paths
import plots
import stage_tests as ST
import targets as T
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("Fig2", [
    Slot("tumor_heatmap", "A", 0, 0, 124, 72),
    Slot("pseudobulk", "B", 126, 0, 57, 72),
    Slot("umap_cyto", "C", 0, 74, 70, 94),
    Slot("lfc_GPRC5D", "D", 72, 74, 55.5, 47),
    Slot("lfc_CD19", "E", 127.5, 74, 55.5, 47),
    Slot("lfc_CD48", "F", 72, 121, 55.5, 47),
    Slot("lfc_ITGB7", "G", 127.5, 121, 55.5, 47),
], max_height=170)
PT_MM = 25.4 / 72.0
GENE = {"t(11;14)": "CCND1", "t(12;14)": "CCND2", "t(6;14)": "CCND3", "t(4;14)": "NSD2", "t(14;16)": "MAF",
        "t(8;14)": "MAFA", "t(14;20)": "MAFB", "Other": "Other", "HRD": "HRD"}
TR = ["CCND1", "CCND2", "CCND3", "NSD2", "MAF", "MAFA", "MAFB", "Other"]
STAGES5 = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
NORM = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
Y0, Y1 = 17.5, 69.5                       # heatmap rows of A and B share these limits (mm), so the gene rows line up


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/Fig2_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center"):
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs if isinstance(runs, list) else [(runs, "")], size=size, ha=ha,
                     va=va, transform=sf.transSubfigure, gid=gid)


def mesh(ax, M, key):
    """Vector heatmap cells (one QuadMesh), coolwarm -3..3."""
    m = ax.pcolormesh(np.arange(M.shape[1] + 1), np.arange(M.shape[0] + 1), M, cmap="coolwarm", norm=NORM,
                      edgecolors="face", linewidth=0)
    m.set_gid("mesh-" + key)
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)
    return m


def track(ax, colors, key):
    """One row of colored cells (column annotations)."""
    from matplotlib.colors import ListedColormap
    cols = list(dict.fromkeys(colors))
    idx = np.array([[cols.index(c) for c in colors]])
    m = ax.pcolormesh(np.arange(len(colors) + 1), [0, 1], idx, cmap=ListedColormap(cols), vmin=-0.5,
                      vmax=len(cols) - 0.5, edgecolors="face", linewidth=0)
    m.set_gid("mesh-track-" + key)
    ax.set_xlim(0, len(colors))
    ax.set_ylim(1, 0)
    ax.axis("off")


# ---- A ---------------------------------------------------------------------------------------------------------------
def tumor_matrix():
    fdr, lfc = T.de_tables()
    cyto = pd.read_csv(paths.inputs("paper", "cytogenetics-status.csv"), index_col=0)
    meta = pd.DataFrame(index=lfc.columns)
    meta["patient_id"] = [c.split("_BM_")[0] for c in lfc.columns]
    meta["stage"] = [c.split("_BM_")[1].split("_")[0] for c in lfc.columns]
    tid = [p + ":tumor:" + c[-1] for p, c in zip(meta["patient_id"], lfc.columns)]
    meta["tumor"] = tid
    meta["hrd"] = cyto.loc[tid, "hrd_status"].values == "Y"
    tr = cyto.loc[tid, "cytogenetics"].map(GENE).values
    meta["translocation"] = [None if t == "HRD" else t for t in tr]                  # HRD only: no translocation
    X = T.significant_lfc(fdr, lfc, T.TARGETS)
    # columns: translocation groups in gene-family order, then HRD tumors without a translocation; Ward within each
    # group on all targets but CD19
    Xi = X.drop(index="CD19")
    cols = []
    for t in TR + [None]:
        idx = meta.index[meta["translocation"].isna()] if t is None else meta.index[meta["translocation"] == t]
        sub = Xi[idx].T
        cols += list(sub.index[sch.leaves_list(sch.linkage(sub, method="ward", metric="euclidean"))]) if len(sub) > 1 \
            else list(sub.index)
    # rows: complete-linkage clustering on the cosine distance (as sns.clustermap)
    Xc = X[cols]
    rows = list(Xc.index[sch.leaves_list(sch.linkage(Xc.values, method="complete", metric="cosine"))])
    assert rows == T.TARGETS_SORTED, rows
    assert len(cols) == 150 and meta.loc[cols, "patient_id"].nunique() == 149
    return Xc.loc[rows], meta.loc[cols]


def panel_tumor_heatmap(sf):
    M, meta = tumor_matrix()
    x0, x1 = 15.0, 115.5
    ah = axes_mm(sf, x0, Y0, x1 - x0, Y1 - Y0)
    mesh(ah, M.values, "fig2a")
    ah.set_yticks(np.arange(len(M)) + 0.5)
    ah.set_yticklabels(M.index, fontsize=SMALL)
    ah.set_xticks([])
    ah.tick_params(axis="y", length=0, pad=1.0)
    for sp in ah.spines.values():
        sp.set_visible(False)
    tracks = [("Disease stage", [P.TUMOR[s] for s in meta["stage"]]),
              ("HRD", [P.CYTO["HRD"] if h else "#FFFFFF" for h in meta["hrd"]]),
              ("Translocation", [P.CYTO[t] if t else "#FFFFFF" for t in meta["translocation"]])]
    for k, (lab, colors) in enumerate(tracks):
        y = 9.6 + k * 2.55
        at_ = axes_mm(sf, x0, y, x1 - x0, 2.2)
        track(at_, colors, "a%d" % k)
        at_mm(sf, x0 - 1.0, y + 1.1, lab, "pn-atr%d" % k, size=SMALL, ha="right")
    # color scale (right of the heatmap) and keys (top)
    cax = axes_mm(sf, x1 + 1.6, Y0 + 12.0, 1.3, Y1 - Y0 - 24.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=NORM, cmap="coolwarm"), cax, ticks=[-3, 0, 3])
    plots.plain_ticks(cb.ax.yaxis)
    rich_text(cax, 0.0, 1, [("Log", ""), ("2", "_"), (" FC", "")], size=SMALL, ha="left", va="bottom",
              transform=plots.pts(cax, 0, 3.8, cax.transAxes), gid="pn-cblab")
    for r, (title, items) in enumerate((("Disease stage", [(s_, P.TUMOR[s_]) for s_ in STAGES5]),
                                         ("Cytogenetic group", [(g, P.CYTO[g]) for g in TR + ["HRD"]]))):
        y = 2.0 + r * 3.4
        at_mm(sf, x0, y, title, "pn-kt%d" % r, size=SMALL)
        kx = x0 + rich_width_pt(sf, [(title, "")], SMALL) * PT_MM + 2.5
        plots.key(sf, kx / sf.w_mm, 1 - (y - 1.0) / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.3,
                  marker="s", ncol=len(items), row_pt=2.0 / PT_MM, gid="pn-k%d" % r)
    out = M.T.copy()
    out.insert(0, "translocation_group", meta["translocation"].fillna("HRD only").values)
    out.insert(0, "HRD", meta["hrd"].values)
    out.insert(0, "stage", meta["stage"].values)
    out.insert(0, "tumor", meta["tumor"].values)
    source("A_tumor_log2fc", out.reset_index(drop=True))


# ---- B ---------------------------------------------------------------------------------------------------------------
def panel_pseudobulk(sf):
    rd = lambda n: pd.read_csv(paths.inputs("paper", "degs", "pseudobulk", n), index_col=0)
    st = pd.read_csv(paths.result("differential_expression", "disease-tumor-matched-imwg-risk-degs.csv"), index_col=0).rename(
        columns={"BM_%s_tumor_mp" % s: s for s in STAGES5})
    cy = rd("cytogenetics-tumor-matched-degs.csv").rename(columns={
        "BM_HRD_tumor_mp": "HRD", "BM_t0414_tumor_mp": "NSD2", "BM_t0614_tumor_mp": "CCND3", "BM_t0814_tumor_mp": "MAFA",
        "BM_t1114_tumor_mp": "CCND1", "BM_t1214_tumor_mp": "CCND2", "BM_t1416_tumor_mp": "MAF", "BM_t1420_tumor_mp": "MAFB",
        "BM_Other_tumor_mp": "Other"})
    groups = TR + ["HRD"]
    A = st.loc[T.TARGETS_SORTED, STAGES5].fillna(0)
    B = cy.loc[T.TARGETS_SORTED, groups].fillna(0)
    cw, gap, x0 = 2.6, 1.6, 12.5
    blocks = [(A, x0, [P.TUMOR[s] for s in STAGES5], "stage"),
              (B, x0 + len(STAGES5) * cw + gap, [P.CYTO[g] for g in groups], "cyto")]
    for M, xs, colors, key in blocks:
        ax = axes_mm(sf, xs, Y0, M.shape[1] * cw, Y1 - Y0)
        mesh(ax, M.values, "fig2b-" + key)
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(M)) + 0.5)
        ax.set_yticklabels(M.index if key == "stage" else [], fontsize=SMALL)
        ax.tick_params(axis="y", length=0, pad=1.0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        tk = axes_mm(sf, xs, 14.6, M.shape[1] * cw, 2.2)
        track(tk, colors, "b" + key)
        for j, name in enumerate(M.columns):
            t = sf.text((xs + (j + 0.5) * cw) / sf.w_mm, 1 - 14.0 / sf.h_mm, name, rotation=90, ha="center", va="bottom",
                        fontsize=SMALL, transform=sf.transSubfigure)
            t.set_gid("pn-bcol-%s%d" % (key, j))
    source("B_pseudobulk", pd.concat([A.add_prefix("stage:"), B.add_prefix("group:")], axis=1).reset_index()
           .rename(columns={"index": "gene"}))


# ---- C-G -------------------------------------------------------------------------------------------------------------
def tumor_cells():
    z = np.load(paths.cache("tumors_cells.npz"))
    cat = lambda k: np.where(z[k + "__codes"] < 0, "nan", z[k + "__cats"].astype(str)[np.maximum(z[k + "__codes"], 0)])
    c = {k: cat(k) for k in ("sample_type", "disease_status", "tumor_id", "cytogenetics", "hrd_status")}
    patient = np.array([t.split(":")[0] for t in c["tumor_id"]])
    c["disease_status"] = np.array(ST.imwg_stages(patient, c["sample_type"], c["disease_status"]))   # IMWG risk groups
    return z["umap"], c


def panel_umap_cyto(sf, xy, c):
    bm = c["sample_type"] == "BM"
    cyto = np.array([GENE.get(x, "nan") for x in c["cytogenetics"]])
    cy = np.where((c["hrd_status"] == "Y") & (cyto != "HRD"), np.char.add(cyto, "+HRD"), cyto)
    pal = dict(P.CYTO, **P.CYTO_HRD)
    order = [g for k in TR if k != "Other" for g in (k, k + "+HRD") if g in set(cy[bm])] + ["HRD", "Other"]
    assert set(cy[bm]) <= set(order), set(cy[bm]) - set(order)
    ax = axes_mm(sf, 5.0, 4.0, 62.0, 62.0)
    col = to_rgba_array([pal[v] for v in cy[bm]])
    col[:, 3] = 0.6
    plots.umap(ax, xy[bm], col, "fig2c", s=0.05)
    ax.set_title("Malignant BM PCs by cytogenetic group", fontsize=BASE, pad=2.0)
    items = [(g, pal[g]) for g in order]
    ncol = 3
    plots.key(sf, 5.0 / sf.w_mm, 1 - 70.5 / sf.h_mm, items, transform=sf.transSubfigure, size=SMALL, ms=2.3, ncol=ncol,
              row_pt=2.1 / PT_MM, gid="pn-ccyto")


def panel_lfc(sf, gene, xy, vals):
    ax = axes_mm(sf, 3.5, 4.0, 40.0, 40.0)
    v = vals
    ok = ~np.isnan(v)
    order = np.r_[np.where(~ok)[0], np.where(ok)[0][np.argsort(v[ok], kind="stable")]]
    col = np.ones((len(v), 4))
    col[ok] = plots.matplotlib_cm("coolwarm")(NORM(v[ok]))
    plots.umap(ax, xy, col, "fig2-" + gene, s=0.03, labels=False)
    ax.set_title(gene, fontsize=BASE, pad=2.0)
    cax = axes_mm(sf, 45.0, 10.0, 1.3, 28.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=NORM, cmap="coolwarm"), cax, ticks=[-3, 0, 3])
    plots.plain_ticks(cb.ax.yaxis)
    rich_text(cax, 0.0, 1, [("Log", ""), ("2", "_"), (" FC", "")], size=SMALL, ha="left", va="bottom",
              transform=plots.pts(cax, 0, 3.8, cax.transAxes), gid="pn-cb-" + gene)


def main():
    xy, c = tumor_cells()
    genes = ["GPRC5D", "CD19", "CD48", "ITGB7"]
    vals, tumors = T.per_cell(c["sample_type"], c["disease_status"], c["tumor_id"], genes)
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_tumor_heatmap(pn["tumor_heatmap"])
        panel_pseudobulk(pn["pseudobulk"])
        panel_umap_cyto(pn["umap_cyto"], xy, c)
        for j, g in enumerate(genes):
            panel_lfc(pn["lfc_" + g], g, xy, vals[:, j])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("Fig2", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
