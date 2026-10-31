#!/usr/bin/env python3
"""Supplementary Figure 7: the sixteen signatures: GO terms, overlap with published signatures, activity across
bone marrow cell types and correlations between signatures.

Purpose:      A, 45 representative GO terms (dots where FDR <= 0.05; size, -log10 FDR clipped at 8; color, genes of the
              signature in the term), one-sided hypergeometric test with Benjamini-Hochberg FDR. B, Jaccard index
              between the 16 signatures and 42 published signatures, dots q < 0.05; average-linkage clustering. C,
              activity of the signatures across bone marrow cell types of Tirier et al., scaled to the highest cell
              type. D, Spearman correlation of the signature activities across the 181 tumors, average-linkage order on
              1 - rho.

Inputs:       results/signatures: go_enrichment_selected.tsv, published_signature_overlap_pairs.tsv,
              signature_correlations_malignant.tsv; results/external_datasets/signature_activity_bm_cell_types.tsv.

Outputs:      figures/SuppFig7.svg, SuppFig7.pdf; figures/source_data/SuppFig7_*.tsv.

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import names
import paths
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, rich_width_pt, SMALL, BASE
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("SuppFig7", [
    Slot("go_terms", "A", 0, 0, 183, 117),
    Slot("overlap", "B", 0, 119, 108, 126),
    Slot("cell_types", "C", 110, 119, 73, 62),
    Slot("correlations", "D", 110, 183, 73, 62),
], max_height=247)
PT_MM = 25.4 / 72.0
SIG_COLUMNS = ["cell_cycling_G2M", "cell_cycling_G1S", "B_cell_activation", "MHC-II_immune_checkpoint",
               "FOS-JUN_stress_response", "MCL1-BCL2_signature", "circulatory_capacity", "interferon_inducible", "HRD_signature",
               "myeloma_oncogene", "secretory_granule", "MAF*_signature", "vacuole_ER_signaling", "MHC-I_gene_module",
               "histone_chaperone_gene", "extracellular_matrix"]                     # signatures with enriched terms
TYPES = ["Malignant PCs", "Normal PCs", "B cells", "Pre-B cells", "T cells", "NK cells", "CD14+ monocytes", "CD16+ monocytes",
         "Myeloid DCs", "pDCs", "Progenitors", "Erythroid cells"]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/SuppFig7_%s.tsv" % key, sep="\t", index=False)


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center"):
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs if isinstance(runs, list) else [(runs, "")], size=size, ha=ha,
                     va=va, transform=sf.transSubfigure, gid=gid)


def sig_label(s):
    """Display name of a signature from a key or from the labels written by the comment scripts."""
    if s in names.SIGNATURES:
        return names.SIGNATURES[s]
    alias = {"HRD": "HRD signature", "MAF signature": "MAF* signature", "MAF*": "MAF* signature",
             "MCL1-BCL2": "MCL1-BCL2 signature"}
    lab = alias.get(s, s)
    assert lab in names.SIGNATURES.values(), s
    return lab


# ---- A ---------------------------------------------------------------------------------------------------------------
# three representative terms per signature (the same list as 06_signatures/19_go_enrichment.py)
ENRICHED = {
    "cell_cycling_G2M": ["cell cycle process", "mitotic spindle organization", "cell cycle G2/M phase transition"],
    "cell_cycling_G1S": ["cell cycle process", "DNA replication", "positive regulation of cell cycle G1/S phase transition"],
    "B_cell_activation": ["B cell activation", "ribonucleoside catabolic process", "DNA cytosine deamination"],
    "MHC-II_immune_checkpoint": ["MHC class II protein complex binding", "regulation of leukocyte cell-cell adhesion",
                                 "antigen receptor-mediated signaling pathway"],
    "FOS-JUN_stress_response": ["positive regulation of RNA metabolic process", "transcription factor AP-1 complex",
                                "DNA-binding transcription activator activity"],
    "MCL1-BCL2_signature": ["regulation of metabolic process", "activation of immune response",
                            "regulation of programmed cell death"],
    "circulatory_capacity": ["positive regulation of plasma membrane repair", "supramolecular fiber organization",
                             "membrane raft organization"],
    "interferon_inducible": ["defense response", "response to cytokine", "inflammatory response"],
    "HRD_signature": ["ribosomal subunit", "cytoplasmic translation", "extracellular exosome"],
    "myeloma_oncogene": ["cell migration", "cell surface receptor signaling pathway", "cell leading edge"],
    "secretory_granule": ["secretory granule membrane", "secretory vesicle", "nuclear envelope lumen"],
    "MAF*_signature": ["intracellular signaling cassette", "cell adhesion", "actin filament-based process"],
    "vacuole_ER_signaling": ["extracellular exosome", "endoplasmic reticulum", "vacuole"],
    "MHC-I_gene_module": ["MHC class I protein complex", "positive regulation of cytokine production",
                          "regulation of lymphocyte mediated immunity"],
    "histone_chaperone_gene": ["chaperone cofactor-dependent protein refolding", "response to unfolded protein",
                               "Nucleosome assembly"],
    "extracellular_matrix": ["regulation of collagen fibril organization", "cell adhesion", "extracellular matrix"],
}


def panel_go(sf):
    g = pd.read_csv(paths.result("signatures", "go_enrichment_selected.tsv"), sep="\t")
    terms = list(dict.fromkeys(t for k in SIG_COLUMNS for t in ENRICHED[k]))
    assert len(terms) == 45
    fmt = lambda x: x[0].upper() + x[1:]
    d = g[g["fdr"] <= 0.05].copy()
    d["mlog"] = np.clip(-np.log10(d["fdr"]), None, 8)
    x0, cw, rh = 62.0, 4.4, 1.96
    ax = axes_mm(sf, x0, 3.0, cw * 16, rh * 45)
    for k in range(16):
        ax.axvline(k, color="#BDBDBD", lw=0.25, ls=(0, (2, 2)), zorder=0).set_gid("bg-grid")
    for r in range(45):
        ax.axhline(r, color="#BDBDBD", lw=0.25, ls=(0, (2, 2)), zorder=0).set_gid("bg-grid")
    norm = Normalize(10, 60)
    cmap = plots.matplotlib_cm("viridis")
    smin, smax = 1.0, 3.4                                     # marker diameters (pt) for -log10 FDR 1.3 .. 8
    size = lambda m: smin + (smax - smin) * (np.clip(m, 1.3, 8) - 1.3) / (8 - 1.3)
    xs = [SIG_COLUMNS.index(s) for s in d["signature"]]
    ys = [terms.index(t) for t in d["name"]]
    sc = ax.scatter(xs, ys, s=size(d["mlog"].values) ** 2, c=cmap(norm(d["count"].clip(10, 60))), edgecolors="black",
                    linewidths=0.25, zorder=3)
    sc.set_gid("dots-go")
    ax.set_xlim(-0.6, 15.6)
    ax.set_ylim(44.6, -0.6)
    ax.set_xticks(range(16))
    ax.set_xticklabels([sig_label(s) for s in SIG_COLUMNS], rotation=90, fontsize=SMALL)
    ax.set_yticks(range(45))
    ax.set_yticklabels([fmt(t) for t in terms], fontsize=SMALL)
    ax.tick_params(length=0, pad=1.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    # keys: dot size and color
    kx = x0 + cw * 16 + 6.0
    at_mm(sf, kx, 6.0, [("−log", ""), ("10", "_"), (" ", ""), ("q", "i")], "pn-ks", size=SMALL)
    for i, m in enumerate([2, 4, 6, 8]):
        y = 9.0 + i * 2.6
        dot = plots.Line2D([(kx + 1.2) / sf.w_mm], [1 - y / sf.h_mm], ls="none", marker="o", ms=size(m), mfc="#BDBDBD",
                           mec="black", mew=0.25, transform=sf.transSubfigure, clip_on=False)
        dot.set_gid("key-size%d" % i)
        sf.add_artist(dot)
        at_mm(sf, kx + 3.4, y, "%d%s" % (m, "+" if m == 8 else ""), "pn-ks%d" % i, size=SMALL)
    at_mm(sf, kx, 22.5, "Genes in term", "pn-kc", size=SMALL)
    cax = axes_mm(sf, kx, 24.5, 1.3, 14.0)
    cb = plots.colorbar(sf, mcm.ScalarMappable(norm=norm, cmap=cmap), cax, ticks=[10, 30, 60])
    cb.ax.set_yticklabels(["≤ 10", "30", "≥ 60"])
    out = g[g["name"].isin(terms)].copy()
    out["shown"] = out["fdr"] <= 0.05
    source("A", out)


# ---- B ---------------------------------------------------------------------------------------------------------------
def published_label(r):
    """Names of the published signatures as written in the overlap table, with spaces after commas and the NF-κB spelling."""
    r = r.replace(",", ", ").replace(",  ", ", ")
    return r.replace("NFkB", "NF-κB").replace("NF-kB", "NF-κB")


def panel_overlap(sf):
    p = pd.read_csv(paths.result("signatures", "published_signature_overlap_pairs.tsv"), sep="\t")
    J = p.pivot(index="published", columns="signature", values="jaccard")
    Q = p.pivot(index="published", columns="signature", values="q")
    ro = leaves_list(linkage(J.values, "average"))
    co = leaves_list(linkage(J.values.T, "average"))
    J, Q = J.iloc[ro, co], Q.iloc[ro, co]
    assert J.shape == (42, 16)
    cw, rh = 2.55, 2.05
    x0 = 49.0
    ax = axes_mm(sf, x0, 3.0, cw * 16, rh * 42)
    vmax = max(0.3, float(J.values.max()))
    m = ax.pcolormesh(np.arange(17), np.arange(43), J.values, cmap="Purples", vmin=0, vmax=vmax, edgecolors="face",
                      linewidth=0)
    m.set_gid("mesh-overlap")
    yy, xx = np.where(Q.values < 0.05)
    dots = ax.scatter(xx + 0.5, yy + 0.5, s=1.1 ** 2, c="black", linewidths=0, zorder=3)
    dots.set_gid("dots-q")
    ax.set_xlim(0, 16)
    ax.set_ylim(42, 0)
    ax.set_xticks(np.arange(16) + 0.5)
    ax.set_xticklabels([sig_label(c) for c in J.columns], rotation=90, fontsize=SMALL)
    ax.set_yticks(np.arange(42) + 0.5)
    ax.set_yticklabels([published_label(r) for r in J.index], fontsize=SMALL)
    ax.tick_params(length=0, pad=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = axes_mm(sf, x0 + cw * 16 + 2.5, 6.0, 1.3, 20.0)
    plots.colorbar(sf, m, cax, ticks=[0, 0.1, 0.2, 0.3])
    plots.plain_ticks(cax.yaxis)
    rich_text(cax, 0, 1, [("Jaccard", "")], size=SMALL, ha="left", va="bottom", transform=plots.pts(cax, 0, 3.8, cax.transAxes),
              gid="pn-jac")
    kx = x0 + cw * 16 + 2.5
    plots.key(sf, kx / sf.w_mm, 1 - 30.0 / sf.h_mm, [([("q", "i"), (" < 0.05", "")], "black")], transform=sf.transSubfigure,
              size=SMALL, ms=1.6, gid="pn-kq")
    source("B", p[["signature", "published", "overlap", "jaccard", "q"]])


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_cell_types(sf):
    T = pd.read_csv(paths.result("external_datasets", "signature_activity_bm_cell_types.tsv"), sep="\t")
    M = T.pivot(index="signature", columns="cell_type", values="median_activity")[TYPES]
    M = M.div(M.max(axis=1), axis=0)
    M = M.loc[M["Malignant PCs"].sort_values(ascending=False).index]
    n = T.drop_duplicates("cell_type").set_index("cell_type")["n_samples"]
    cw, rh = 3.3, 2.15
    x0 = 24.0
    ax = axes_mm(sf, x0, 3.0, cw * 12, rh * 16)
    m = ax.pcolormesh(np.arange(13), np.arange(17), M.values, cmap="Purples", vmin=0, vmax=1, edgecolors="white",
                      linewidth=0.3)
    m.set_gid("mesh-celltypes")
    ax.set_xlim(0, 12)
    ax.set_ylim(16, 0)
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(["%s (n = %d)" % (t, n[t]) for t in TYPES], rotation=90, fontsize=SMALL)
    ax.set_yticks(np.arange(16) + 0.5)
    ax.set_yticklabels([sig_label(s) for s in M.index], fontsize=SMALL)
    ax.tick_params(length=0, pad=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = axes_mm(sf, x0 + cw * 12 + 2.5, 8.0, 1.3, 18.0)
    plots.colorbar(sf, m, cax, ticks=[0, 0.5, 1])
    plots.plain_ticks(cax.yaxis)
    for k, word in enumerate(("Relative", "activity")):           # relative to the cell type with the highest activity
        rich_text(cax, 0, 1, [(word, "")], size=SMALL, ha="left", va="bottom",
                  transform=plots.pts(cax, 0, 3.8 + (1 - k) * SMALL * 1.2, cax.transAxes), gid="pn-rel%d" % k)
    source("C", M.reset_index())


# ---- D ---------------------------------------------------------------------------------------------------------------
def panel_correlations(sf):
    R = pd.read_csv(paths.result("signatures", "signature_correlations_malignant.tsv"), sep="\t", index_col=0)
    order = R.index[leaves_list(linkage(squareform(1 - R.values, checks=False), "average"))]
    R = R.loc[order, order]
    cw = 2.15
    x0 = 24.0
    ax = axes_mm(sf, x0, 3.0, cw * 16, cw * 16)
    m = ax.pcolormesh(np.arange(17), np.arange(17), R.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1),
                      edgecolors="white", linewidth=0.3)
    m.set_gid("mesh-rho")
    ax.set_xlim(0, 16)
    ax.set_ylim(16, 0)
    labs = [sig_label(s) for s in order]
    ax.set_xticks(np.arange(16) + 0.5)
    ax.set_xticklabels(labs, rotation=90, fontsize=SMALL)
    ax.set_yticks(np.arange(16) + 0.5)
    ax.set_yticklabels(labs, fontsize=SMALL)
    ax.tick_params(length=0, pad=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = axes_mm(sf, x0 + cw * 16 + 2.5, 8.0, 1.3, 18.0)
    cb = plots.colorbar(sf, m, cax, ticks=[-1, 0, 1])
    plots.plain_ticks(cax.yaxis)
    for k, runs in enumerate(([("Spearman\u2019s", "")], [("ρ", "i")])):
        rich_text(cax, 0, 1, runs, size=SMALL, ha="left", va="bottom",
                  transform=plots.pts(cax, 0, 3.8 + (1 - k) * SMALL * 1.2, cax.transAxes), gid="pn-rho%d" % k)
    source("D", R.rename(index=sig_label, columns=sig_label).reset_index().rename(columns={"index": "signature"}))


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        panel_go(pn["go_terms"])
        panel_overlap(pn["overlap"])
        panel_cell_types(pn["cell_types"])
        panel_correlations(pn["correlations"])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig7", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
