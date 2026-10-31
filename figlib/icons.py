"""Simple vector icons and the schematics they build (Fig. 1A study design, Fig. 4A classifier design).

Purpose:      Every shape is a matplotlib path (no images) and every label is editable text. A schematic is drawn on one
              axes covering its panel, in millimetres from the panel's top-left corner. Box backgrounds carry gids
              starting with "bg-" (text may sit on them); icons and arrows are plot elements, so checks.py verifies that
              no text covers them. fit() raises when a label would not fit the width given to it.

Inputs:       None (the figure scripts pass the counts).

Outputs:      study_design(), classifier_schematic() and the icon helpers, imported by the figure scripts.

Dependencies: numpy, matplotlib; figstyle.py, palettes.py, layout.py and plots.py.
"""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon

import palettes as P
from figstyle import BASE, SMALL, TITLE, LW, rich_text, rich_width_pt, rng
from layout import axes_mm
from plots import matplotlib_cm

PT_MM = 25.4 / 72.0           # millimetres per point
INK = "#333333"


# ---- canvas, text, boxes, arrows ----------------------------------------------------------------------------------
def canvas(sf, key):
    ax = axes_mm(sf, 0, 0, sf.w_mm, sf.h_mm)
    ax.set_xlim(0, sf.w_mm)
    ax.set_ylim(sf.h_mm, 0)
    ax.axis("off")
    ax.set_gid("canvas-" + key)
    return ax


def runs_of(s):
    return s if isinstance(s, list) else [(s, "")]


def fit(ax, s, size, width_mm, what=""):
    w = rich_width_pt(ax.figure, runs_of(s), size) * PT_MM
    assert w <= width_mm + 1e-6, "label %r is %.1f mm wide (room %.1f mm) %s" % (s, w, width_mm, what)
    return w


def label(ax, x, y, s, key, size=BASE, ha="center", va="center", width=None, bold=False):
    """Text at (x, y) mm; `width` (mm) = room available, checked."""
    runs = runs_of(s)
    if bold:
        runs = [(t, f + "b") for t, f in runs]
    if width is not None:
        fit(ax, runs, size, width, key)
    return rich_text(ax, x, y, runs, size=size, ha=ha, va=va, transform=ax.transData, gid="pn-" + key)


def box(ax, x, y, w, h, key, fc, ec, r=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec, lw=LW, zorder=0)
    p.set_gid("bg-box-" + key)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, key, color=INK):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=2.2,head_width=1.1", mutation_scale=1.0,
                        lw=0.6, color=color, shrinkA=0, shrinkB=0, zorder=2)
    a.set_gid("arrow-" + key)
    ax.add_patch(a)
    return a


def elbow(ax, pts_, key, color=INK):
    """Polyline whose last segment ends in an arrow head (pts_ in mm)."""
    xs, ys = zip(*pts_[:-1])
    ln, = ax.plot(xs, ys, color=color, lw=0.6, solid_capstyle="butt", zorder=2)
    ln.set_gid("line-" + key)
    return arrow(ax, *pts_[-2], *pts_[-1], key, color)


# ---- icons ---------------------------------------------------------------------------------------------------------
def dots(ax, xy, colors, key, d_mm=0.55, edge=0.0):
    s = (d_mm / PT_MM) ** 2
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=colors, edgecolors="black" if edge else "none", linewidths=edge,
                    zorder=3)
    sc.set_gid("icon-" + key)
    return sc


def umap_icon(ax, cx, cy, size, key):
    """A framed embedding with five clusters of cells."""
    r = rng(key)
    fr = Rectangle((cx - size / 2, cy - size / 2), size, size, fill=False, ec="#808080", lw=0.4, zorder=1)
    fr.set_gid("icon-frame-" + key)
    ax.add_patch(fr)
    centers = np.array([[-0.22, -0.20], [0.20, -0.24], [0.24, 0.16], [-0.16, 0.22], [0.0, 0.0]]) * size
    cols = ["#56B4E9", "#C2185B", "#E69F00", "#009E73", "#8C96A6"]
    xy, cc = [], []
    for (dx, dy), c in zip(centers, cols):
        p = r.normal(0, size * 0.065, (14, 2))
        xy.append(np.c_[cx + dx + p[:, 0], cy + dy + p[:, 1]])
        cc += [c] * 14
    xy = np.vstack(xy)
    keep = (np.abs(xy[:, 0] - cx) < size / 2 - 0.6) & (np.abs(xy[:, 1] - cy) < size / 2 - 0.6)
    return dots(ax, xy[keep], np.array(cc)[keep], key)


def heatmap_icon(ax, x, y, w, h, key, n=6, cmap="YlOrRd"):
    """A small gene-gene correlation matrix with blocks on the diagonal (signatures)."""
    r = rng(key)
    M = r.uniform(0.0, 0.3, (n, n))
    for a, b in [(0, 2), (2, 4), (4, 6)]:
        M[a:b, a:b] = r.uniform(0.65, 1.0, (b - a, b - a))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    mesh = ax.pcolormesh(np.linspace(x, x + w, n + 1), np.linspace(y, y + h, n + 1), M, cmap=cmap, vmin=0, vmax=1,
                         edgecolors="white", linewidth=0.3, zorder=1)
    mesh.set_gid("icon-" + key)
    return mesh


def rank_bars_icon(ax, x, y, w, h, key, n=11, cut=6):
    """AUCell: genes ranked by expression, the signature's area under the recovery curve up to a rank cut-off."""
    xs = np.linspace(x + 0.6, x + w - 0.4, n)
    bw = (xs[1] - xs[0]) * 0.8
    hs = h * (0.18 + 0.82 * (np.arange(1, n + 1) / n) ** 1.1)
    for k, (xx, hh) in enumerate(zip(xs, hs)):
        b = Rectangle((xx - bw / 2, y + h - hh), bw, hh, fc="#1F5FA8" if k < cut else "#A6C8E8", ec="none", zorder=1)
        b.set_gid("icon-bar-%s%d" % (key, k))
        ax.add_patch(b)
    ln, = ax.plot([x, x, x + w], [y, y + h, y + h], color=INK, lw=0.4, zorder=2)
    ln.set_gid("icon-axes-" + key)
    xc = (xs[cut - 1] + xs[cut]) / 2
    cl, = ax.plot([xc, xc], [y + 0.2, y + h], color=P.BRACKET_MAL, lw=0.5, ls=(0, (1.5, 1)), zorder=2)
    cl.set_gid("icon-cut-" + key)


def chromosome_icon(ax, cx, top, w, h, key, color, arm="q", cen=0.38):
    """One chromosome (two chromatids); the highlighted arm ("p", "q" or None) is filled with `color`."""
    p_h, q_h = h * cen, h * (1 - cen)
    for side in (-1, 1):
        x0 = cx + side * w * 0.27 - w * 0.23
        for part, (y0, hh) in (("p", (top, p_h - 0.25)), ("q", (top + p_h + 0.25, q_h - 0.25))):
            fc = color if arm == part else "#D9D9D9"
            b = FancyBboxPatch((x0, y0), w * 0.46, hh, boxstyle="round,pad=0,rounding_size=%g" % (w * 0.2), fc=fc,
                               ec=INK, lw=0.4, zorder=1)
            b.set_gid("icon-chr-%s%s%d" % (key, part, side))
            ax.add_patch(b)
            for f in (0.35, 0.7):
                bd = Rectangle((x0 + 0.05, y0 + hh * f - 0.18), w * 0.46 - 0.1, 0.36, fc=INK, ec="none", alpha=0.55,
                               zorder=2)
                bd.set_gid("icon-band-%s%s%d%g" % (key, part, side, f))
                ax.add_patch(bd)


def knn_icon(ax, cx, cy, rad, key, center, neighbor, halo):
    """A cell (dark), its nearest neighbors (shaded area) and other cells (white)."""
    r = rng(key)
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False) + r.uniform(-0.25, 0.25, 6) + 0.3
    near = np.c_[cx + np.cos(ang) * rad * r.uniform(0.42, 0.55, 6), cy + np.sin(ang) * rad * r.uniform(0.42, 0.55, 6)]
    ang2 = ang[::2] + 0.45
    far = np.c_[cx + np.cos(ang2) * rad * 0.92, cy + np.sin(ang2) * rad * 0.92]
    h = Polygon(np.c_[cx + np.cos(np.linspace(0, 2 * np.pi, 40)) * rad * 0.68,
                      cy + np.sin(np.linspace(0, 2 * np.pi, 40)) * rad * 0.68], closed=True, fc=halo, ec="none", zorder=0)
    h.set_gid("icon-halo-" + key)
    ax.add_patch(h)
    segs = [[(cx, cy), tuple(p)] for p in near] + [[tuple(far[k]), tuple(near[2 * k])] for k in range(3)] + \
        [[tuple(near[k]), tuple(near[(k + 1) % 6])] for k in (0, 2, 4)]
    lc = LineCollection(segs, colors=INK, linewidths=0.4, zorder=1)
    lc.set_gid("icon-edges-" + key)
    ax.add_collection(lc)
    dots(ax, near, [neighbor] * 6, key + "-near", d_mm=1.15, edge=0.35)
    dots(ax, far, ["white"] * 3, key + "-far", d_mm=1.15, edge=0.35)
    dots(ax, np.array([[cx, cy]]), [center], key + "-cell", d_mm=1.5, edge=0.35)


def tree_icon(ax, cx, top, w, h, key, leaves):
    """A decision tree: root, two inner nodes, four leaves colored by the predicted class."""
    xs = {0: [cx], 1: [cx - w / 4, cx + w / 4], 2: [cx - 3 * w / 8, cx - w / 8, cx + w / 8, cx + 3 * w / 8]}
    ys = {0: top, 1: top + h / 2, 2: top + h}
    segs = []
    for lv in (1, 2):
        for k, x in enumerate(xs[lv]):
            segs.append([(xs[lv - 1][k // 2], ys[lv - 1]), (x, ys[lv])])
    lc = LineCollection(segs, colors=INK, linewidths=0.4, zorder=1)
    lc.set_gid("icon-tree-" + key)
    ax.add_collection(lc)
    inner = np.array([(xs[0][0], ys[0])] + [(x, ys[1]) for x in xs[1]])
    dots(ax, inner, ["white"] * 3, key + "-inner", d_mm=1.0, edge=0.35)
    dots(ax, np.array([(x, ys[2]) for x in xs[2]]), leaves, key + "-leaves", d_mm=1.0, edge=0.35)


def gradient_bar(ax, x, y, w, h, key, cmap):
    """Score color bar from 0 to 1 (vector cells), framed."""
    n = 40
    cm = matplotlib_cm(cmap)
    mesh = ax.pcolormesh(np.linspace(x, x + w, n + 1), [y, y + h], np.linspace(0, 1, n)[None, :], cmap=cm, vmin=0,
                         vmax=1, edgecolors="face", linewidth=0, zorder=1)
    mesh.set_gid("icon-bar-" + key)
    fr = Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.4, zorder=2)
    fr.set_gid("icon-barframe-" + key)
    ax.add_patch(fr)


# ---- Fig. 4A ---------------------------------------------------------------------------------------------------------
def classifier_schematic(sf):
    """Fig. 4A: features, normalization, kNN smoothing and the two random forest classifiers (feature sets from the classifier inputs: malignancy = 16 signatures (scaled mean expression) + SenMayo + 9
    translocation partner genes; MGUS-MM = 16 signatures (AUCell) + CNV-like scores of 5 driver arms)."""
    ax = canvas(sf, "classifier")
    W, H = sf.w_mm, sf.h_mm
    L1, L2 = (6.0, 21.0), (24.0, 39.0)               # the two lanes (malignancy, MGUS-MM), top and bottom in mm
    c1, c2 = sum(L1) / 2, sum(L2) / 2
    hy = 2.4                                          # column headers
    cols = {"input": (3.5, 21.0), "base": (23.0, 46.5), "norm": (50.0, 70.0), "add": (73.5, 118.5),
            "knn": (122.0, 147.0), "rf": (150.0, W)}
    mid = {k: (a + b) / 2 for k, (a, b) in cols.items()}
    room = {k: b - a for k, (a, b) in cols.items()}
    for k, s in [("input", "Input"), ("base", "Base features"), ("norm", "Normalization"),
                 ("add", "Additional features"), ("knn", "kNN smoothing"), ("rf", "Random forest")]:
        label(ax, mid[k], hy, s, "hdr-" + k, bold=True, width=room[k])

    # input: scRNA-seq of plasma cells
    umap_icon(ax, mid["input"], 19.0, 12.0, "umap")
    label(ax, mid["input"], 28.6, "scRNA-seq", "in1", size=SMALL, width=room["input"])
    label(ax, mid["input"], 31.0, "of PCs", "in2", size=SMALL, width=room["input"])
    arrow(ax, cols["input"][1] - 1.5, 19.0, cols["base"][0] - 0.2, 19.0, "in")

    # base features: the 16 signatures
    x0, x1 = cols["base"]
    box(ax, x0, L1[0], x1 - x0, L2[1] - L1[0], "base", "#EEF4FB", "#1F5FA8")
    label(ax, mid["base"], 10.0, "16 signatures", "b1", bold=True, width=room["base"] - 2)
    label(ax, mid["base"], 13.0, "1,463 genes", "b2", size=SMALL, width=room["base"] - 2)
    heatmap_icon(ax, mid["base"] - 6.0, 16.5, 12.0, 12.0, "hm")
    label(ax, mid["base"], 32.0, "Activity in", "b3", size=SMALL, width=room["base"] - 2)
    label(ax, mid["base"], 34.4, "each cell", "b4", size=SMALL, width=room["base"] - 2)
    arrow(ax, x1, c1, cols["norm"][0] - 0.3, c1, "b-l1")
    arrow(ax, x1, c2, cols["norm"][0] - 0.3, c2, "b-l2")

    # normalization: scaled mean expression (malignancy) or AUCell (MGUS-MM)
    label(ax, mid["norm"], 10.2, "Mean expression,", "n1", size=SMALL, width=room["norm"])
    label(ax, mid["norm"], 12.6, "scaled:", "n2", size=SMALL, width=room["norm"])
    label(ax, mid["norm"], 16.6, [("z", "i"), (" = (", ""), ("x", "i"), (" − ", ""), ("μ", "i"), (")/", ""),
                                  ("σ", "i")], "n3", width=room["norm"])
    label(ax, mid["norm"], 26.0, "AUCell", "n4", size=SMALL, width=room["norm"])
    rank_bars_icon(ax, mid["norm"] - 5.5, 28.0, 11.0, 7.0, "auc")
    label(ax, mid["norm"], 37.4, "Gene rank", "n5", size=SMALL, width=room["norm"])
    arrow(ax, cols["norm"][1], c1, cols["add"][0] - 0.2, c1, "n-l1")
    arrow(ax, cols["norm"][1], c2, cols["add"][0] - 0.2, c2, "n-l2")

    # additional features of each classifier
    x0, x1 = cols["add"]
    for (ya, yb), k in ((L1, "add1"), (L2, "add2")):
        box(ax, x0, ya, x1 - x0, yb - ya, k, "#FDF3E4", "#E69F00")
    inner = x1 - x0 - 3.0
    label(ax, x0 + 1.5, 9.4, [("Senescence: ", "b"), ("SenMayo gene set (125 genes)", "")], "a1", size=SMALL, ha="left",
          width=inner)
    label(ax, x0 + 1.5, 12.9, [("9 translocation partner genes:", "b")], "a2", size=SMALL, ha="left", width=inner)
    label(ax, x0 + 1.5, 15.6, "CCND1, CCND2, CCND3, FGFR3, MAF,", "a3", size=SMALL, ha="left", width=inner)
    label(ax, x0 + 1.5, 18.2, "MAFA, MAFB, MYC, NSD2", "a4", size=SMALL, ha="left", width=inner)
    chromosome_icon(ax, x0 + 5.2, 26.0, 4.2, 11.0, "drv", P.BRACKET_MAL, arm="q")
    chromosome_icon(ax, x0 + 10.4, 26.0, 4.2, 11.0, "ctl", None, arm=None)
    tx = x0 + 14.5
    label(ax, tx, 27.4, [("CNV-like scores:", "b")], "a5", size=SMALL, ha="left", width=x1 - tx - 1.0)
    label(ax, tx, 30.2, "expression on driver arms", "a6", size=SMALL, ha="left", width=x1 - tx - 1.0)
    label(ax, tx, 32.6, "1p, 1q, 13q, 14q, 17p relative", "a7", size=SMALL, ha="left", width=x1 - tx - 1.0)
    label(ax, tx, 35.0, "to control chromosomes", "a8", size=SMALL, ha="left", width=x1 - tx - 1.0)
    label(ax, tx, 37.4, "2, 4, 10, 12, 20", "a9", size=SMALL, ha="left", width=x1 - tx - 1.0)
    arrow(ax, x1, c1, cols["knn"][0] - 0.2, c1, "a-l1")
    arrow(ax, x1, c2, cols["knn"][0] - 0.2, c2, "a-l2")

    # kNN smoothing of every feature over each cell's 30 nearest neighbors
    knn_icon(ax, mid["knn"], c1 - 0.4, 5.6, "knn1", "#08519C", "#9ECAE1", "#E3EEF8")
    knn_icon(ax, mid["knn"], c2 + 0.6, 5.6, "knn2", "#54278F", "#BCBDDC", "#ECEBF4")
    label(ax, mid["knn"], 22.0, [("cell + Σ", ""), ("k", "^i"), (" NN", ""), ("i", "_i"), (" · exp(−", ""),
                                 ("D", "i"), ("i", "_i"), ("/", ""), ("k", "i"), (")", "")], "k1", size=SMALL,
          width=room["knn"] + 2)
    label(ax, mid["knn"], 24.6, [("k", "i"), (" = 30 nearest neighbors", "")], "k2", size=SMALL, width=room["knn"] + 2)
    arrow(ax, cols["knn"][1], c1, cols["rf"][0] - 0.2, c1, "k-l1")
    arrow(ax, cols["knn"][1], c2, cols["rf"][0] - 0.2, c2, "k-l2")

    # the two random forest classifiers and their scores
    xr = cols["rf"][0]
    for (ya, yb), k, lv, sub, score, cmap in (
            (L1, "rf1", [P.PC_STATUS["BM normal"], P.PC_STATUS["BM malignant"]] * 2, "Normal vs malignant",
             "Malignancy score", P.CMAP_MALIGNANCY),
            (L2, "rf2", [P.TUMOR["MGUS"], P.TUMOR["MM"]] * 2, "MGUS vs MM", "MST score", P.CMAP_MST)):
        tree_icon(ax, xr + 4.2, ya + 2.6, 6.4, 6.4, k + "a", lv)
        tree_icon(ax, xr + 11.6, ya + 2.6, 6.4, 6.4, k + "b", lv[::-1])
        label(ax, xr + 7.9, ya + 12.4, sub, k + "-sub", size=SMALL, width=17.0)
        arrow(ax, xr + 16.2, ya + 6.0, xr + 19.2, ya + 6.0, k + "-out")
        label(ax, xr + 19.8, ya + 3.6, score.split()[0], k + "-s1", ha="left", bold=True, width=W - xr - 20.0)
        label(ax, xr + 19.8, ya + 6.2, "score", k + "-s2", ha="left", bold=True, width=W - xr - 20.0)
        gradient_bar(ax, xr + 19.9, ya + 8.4, 11.5, 1.3, k, cmap)
        label(ax, xr + 19.9, ya + 11.6, "0", k + "-b0", size=SMALL)
        label(ax, xr + 31.4, ya + 11.6, "1", k + "-b1", size=SMALL)
    return ax


# ---- more icons (Fig. 1A) -------------------------------------------------------------------------------------------
def person_icon(ax, cx, base, h, key, color="#6F6F8F"):
    """A standing person: round head over a rounded body; `base` = y of the feet (mm, y downwards)."""
    from matplotlib.patches import Circle
    head = Circle((cx, base - h + h * 0.16), h * 0.16, fc=color, ec="none", zorder=2)
    head.set_gid("icon-head-" + key)
    ax.add_patch(head)
    body = FancyBboxPatch((cx - h * 0.21, base - h * 0.64), h * 0.42, h * 0.64, boxstyle="round,pad=0,rounding_size=%g"
                          % (h * 0.14), fc=color, ec="none", zorder=2)
    body.set_gid("icon-body-" + key)
    ax.add_patch(body)


def bone_icon(ax, cx, top, h, w, key):
    """A long bone (BM): shaft with two knobs at each end."""
    from matplotlib.patches import Circle
    fc, ec = "#F2E6CF", "#8C7A5B"
    shaft = Rectangle((cx - w * 0.22, top + w * 0.35), w * 0.44, h - w * 0.7, fc=fc, ec="none", zorder=2)
    shaft.set_gid("icon-shaft-" + key)
    ax.add_patch(shaft)
    for yy in (top + w * 0.3, top + h - w * 0.3):
        for dx in (-w * 0.22, w * 0.22):
            k = Circle((cx + dx, yy), w * 0.3, fc=fc, ec=ec, lw=0.4, zorder=1)
            k.set_gid("icon-knob-" + key)
            ax.add_patch(k)
    ln, = ax.plot([cx - w * 0.22, cx - w * 0.22], [top + w * 0.5, top + h - w * 0.5], color=ec, lw=0.4, zorder=3)
    ln.set_gid("icon-edge1-" + key)
    ln, = ax.plot([cx + w * 0.22, cx + w * 0.22], [top + w * 0.5, top + h - w * 0.5], color=ec, lw=0.4, zorder=3)
    ln.set_gid("icon-edge2-" + key)
    marrow = Rectangle((cx - w * 0.1, top + w * 0.55), w * 0.2, h - w * 1.1, fc="#C0392B", ec="none", alpha=0.75,
                       zorder=3)
    marrow.set_gid("icon-marrow-" + key)
    ax.add_patch(marrow)


def tube_icon(ax, cx, top, h, w, key, liquid="#B03A2E", cap="#E67E22", cells=None):
    """A sample tube with a cap; `cells` = list of colors drawn as small dots in the liquid."""
    capr = Rectangle((cx - w * 0.6, top), w * 1.2, h * 0.16, fc=cap, ec=INK, lw=0.4, zorder=3)
    capr.set_gid("icon-cap-" + key)
    ax.add_patch(capr)
    body = FancyBboxPatch((cx - w / 2, top + h * 0.16), w, h * 0.84, boxstyle="round,pad=0,rounding_size=%g" % (w * 0.45),
                          fc="white", ec=INK, lw=0.4, zorder=2)
    body.set_gid("icon-tube-" + key)
    ax.add_patch(body)
    fill = FancyBboxPatch((cx - w / 2 + 0.25, top + h * 0.38), w - 0.5, h * 0.6, boxstyle="round,pad=0,rounding_size=%g"
                          % (w * 0.4), fc=liquid, ec="none", alpha=0.9, zorder=3)
    fill.set_gid("icon-liquid-" + key)
    ax.add_patch(fill)
    if cells:
        r = rng(key)
        xy = np.c_[cx + r.uniform(-w * 0.25, w * 0.25, len(cells)), top + h * (0.5 + r.uniform(0, 0.4, len(cells)))]
        dots(ax, xy, cells, key + "-cells", d_mm=w * 0.28, edge=0.2)


def rna_icon(ax, x, y, w, key, colors=("#E69F00", "#0072B2", "#CC79A7")):
    """Three wavy strands (RNA)."""
    t = np.linspace(0, 1, 60)
    for k, c in enumerate(colors):
        ln, = ax.plot(x + t * w, y + k * 0.9 + 0.35 * np.sin(t * 4 * np.pi + k), color=c, lw=0.7, zorder=2)
        ln.set_gid("icon-rna-%s%d" % (key, k))


def antibody_icon(ax, cx, cy, s, key, color="#0072B2"):
    """A Y-shaped antibody (B cell receptor)."""
    segs = [[(cx, cy + s * 0.5), (cx, cy)], [(cx, cy), (cx - s * 0.38, cy - s * 0.42)], [(cx, cy), (cx + s * 0.38, cy - s * 0.42)]]
    lc = LineCollection(segs, colors=color, linewidths=1.2, capstyle="round", zorder=2)
    lc.set_gid("icon-ab-" + key)
    ax.add_collection(lc)


def cells_icon(ax, cx, cy, r, key, colors):
    """A small cluster of cells (circles with a dark nucleus)."""
    from matplotlib.patches import Circle
    rr = rng(key)
    n = len(colors)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rr.uniform(-0.2, 0.2, n)
    for k, c in enumerate(colors):
        x, y = cx + np.cos(ang[k]) * r * 0.55, cy + np.sin(ang[k]) * r * 0.55
        cell = Circle((x, y), r * 0.42, fc=c, ec=INK, lw=0.35, zorder=2 + k * 0.01)
        cell.set_gid("icon-cell-%s%d" % (key, k))
        ax.add_patch(cell)
        nuc = Circle((x + r * 0.08, y - r * 0.05), r * 0.17, fc=INK, ec="none", alpha=0.55, zorder=2.005 + k * 0.01)
        nuc.set_gid("icon-nuc-%s%d" % (key, k))
        ax.add_patch(nuc)


def cell_icon(ax, cx, cy, r, key, color):
    """One cell: a filled circle with a dark nucleus."""
    from matplotlib.patches import Circle
    cell = Circle((cx, cy), r, fc=color, ec=INK, lw=0.35, zorder=2)
    cell.set_gid("icon-cell-" + key)
    ax.add_patch(cell)
    nuc = Circle((cx + r * 0.2, cy - r * 0.12), r * 0.4, fc=INK, ec="none", alpha=0.55, zorder=2.01)
    nuc.set_gid("icon-nuc-" + key)
    ax.add_patch(nuc)


def chip(ax, x, cy, s, key, colors):
    """A square key mark of side s (mm) centered on cy; several colors are drawn as vertical stripes (the three SMM risk
    groups)."""
    w = s / len(colors)
    for k, c in enumerate(colors):
        st = Rectangle((x + k * w, cy - s / 2), w, s, fc=c, ec="none", zorder=2)
        st.set_gid("icon-chip-%s%d" % (key, k))
        ax.add_patch(st)
    fr = Rectangle((x, cy - s / 2), s, s, fill=False, ec="black", lw=0.25, zorder=3)
    fr.set_gid("icon-chipframe-" + key)
    ax.add_patch(fr)


def score_scale(ax, x0, x1, y, key, cmap, lo, hi, lo_color, hi_color):
    """A score from 0 to 1: its color bar between a cell of each of the two classes the classifier separates, with the
    class names under the cells."""
    r = 1.1
    cell_icon(ax, x0 + r, y, r, key + "-lo", lo_color)
    cell_icon(ax, x1 - r, y, r, key + "-hi", hi_color)
    label(ax, x0 + 2 * r + 0.7, y, "0", key + "-0", ha="left")
    label(ax, x1 - 2 * r - 0.7, y, "1", key + "-1", ha="right")
    bx0, bx1 = x0 + 2 * r + 2.9, x1 - 2 * r - 2.9
    gradient_bar(ax, bx0, y - 0.75, bx1 - bx0, 1.5, key, cmap)
    label(ax, x0, y + 3.0, lo, key + "-lo-l", ha="left")
    label(ax, x1, y + 3.0, hi, key + "-hi-l", ha="right")


def volcano_icon(ax, x, y, w, h, key):
    """Differential expression: gray points near zero, blue (down) and red (up) points in the upper arms."""
    rr = rng(key)
    fc = rr.normal(0, 1, 120)
    sig = np.abs(fc) ** 1.6 * rr.uniform(0.4, 1.0, 120)
    px = x + w / 2 + fc / 3.2 * w / 2
    py = y + h - np.clip(sig / sig.max(), 0, 1) * h
    keep = (px > x) & (px < x + w)
    col = np.where((fc > 1.0) & (sig > 1.0), "#C0392B", np.where((fc < -1.0) & (sig > 1.0), "#1F5FA8", "#BDBDBD"))
    dots(ax, np.c_[px[keep], py[keep]], col[keep], key, d_mm=0.55)
    ln, = ax.plot([x, x, x + w], [y, y + h, y + h], color=INK, lw=0.4, zorder=1)
    ln.set_gid("icon-axes-" + key)


def study_design(sf, counts, n_individuals, n_donors, n_pcs):
    """Fig. 1A: individuals -> samples and sequencing -> analyses -> single-cell scores. The
    tissues are spelled out, the disease stages carry the colors of Fig. 1B,C, and each score is drawn as the 0-1 scale
    of its classifier, between a cell of each of the two classes it was trained to separate."""
    ax = canvas(sf, "design")
    top, bot = 5.4, sf.h_mm - 0.6                  # boxes, mm from the top of the panel
    hy, mid = 2.6, (top + bot) / 2                  # box titles; arrows between the boxes
    G, A, S = (21.4, 83.0), (87.4, 135.6), (140.0, 182.6)
    # individuals
    for k, (dx, hh, c) in enumerate([(-4.0, 11.0, "#8C96A6"), (4.0, 11.0, "#6F6F8F"), (0.0, 13.0, "#4B4B6B")]):
        person_icon(ax, 9.0 + dx, 21.6, hh, "p%d" % k, c)
    label(ax, 9.0, 25.6, "%d" % n_individuals, "ind1", size=TITLE, bold=True, width=18)
    label(ax, 9.0, 28.4, "individuals", "ind2", width=18)
    label(ax, 9.0, 32.2, "%d patients" % (n_individuals - n_donors), "ind3", width=18)
    label(ax, 9.0, 34.8, "%d healthy donors" % n_donors, "ind4", width=18)
    arrow(ax, 18.2, mid, G[0] - 0.4, mid, "a0")

    # data generation: the two tissues (individuals by disease stage), then sequencing
    box(ax, G[0], top, G[1] - G[0], bot - top, "gen", "#F7F7F7", "#9E9E9E")
    label(ax, sum(G) / 2, hy, "Data generation", "h-gen", size=TITLE, bold=True, width=G[1] - G[0])
    tx, sx0 = G[0] + 9.2, 55.4                      # text of the tissues; left edge of the sequencing part
    for tissue, name, yy in (("BM", "Bone marrow", top + 2.8), ("PB", "Peripheral blood", top + 21.6)):
        if tissue == "BM":
            bone_icon(ax, G[0] + 4.4, yy - 0.8, 16.0, 4.6, "bone")
        else:
            tube_icon(ax, G[0] + 4.4, yy - 0.6, 15.0, 4.0, "pbtube", liquid="#B03A2E", cap="#E67E22")
        cnt = counts[tissue]
        label(ax, tx, yy, [("%s (%s)" % (name, tissue), "b")], "t-" + tissue, ha="left", width=sx0 - tx - 2.0)
        label(ax, tx, yy + 2.7, "%d individuals" % sum(cnt.values()), "n-" + tissue, ha="left", width=sx0 - tx - 2.0)
        for j, s in enumerate(["MM", "SMM", "MGUS", "Donor"]):
            y = yy + 5.8 + j * 2.6
            cols = [P.STAGE[g] for g in ("LRSMM", "IRSMM", "HRSMM")] if s == "SMM" else [P.STAGE[s]]
            chip(ax, tx + 0.2, y, 1.8, "%s-%s" % (tissue, s), cols)
            label(ax, tx + 2.8, y, "%s %d" % ("Donors" if s == "Donor" else s, cnt[s]), "c-%s-%s" % (tissue, s),
                  ha="left", width=sx0 - tx - 2.8)
    label(ax, (sx0 + G[1]) / 2, top + 2.8, [("Single-cell sequencing", "b")], "sc", width=G[1] - sx0 - 1.0)
    tube_icon(ax, sx0 + 3.2, top + 6.0, 14.0, 4.4, "cd138", liquid="#D6EAF8", cap="#7F8C8D",
              cells=["#C2185B", "#56B4E9", "#C2185B", "#C2185B", "#56B4E9"])
    label(ax, sx0 + 3.2, top + 22.8, "CD138+", "cd138l", width=8.0)
    label(ax, sx0 + 3.2, top + 25.4, "cells", "cd138l2", width=8.0)
    rna_icon(ax, sx0 + 8.4, top + 6.6, 5.6, "rna")                 # each assay, then what it measures
    label(ax, sx0 + 15.0, top + 7.5, "scRNA-seq", "rnal", ha="left", width=G[1] - sx0 - 15.8)
    label(ax, sx0 + 8.4, top + 10.4, "gene expression", "rnal2", ha="left", width=G[1] - sx0 - 9.2)
    for k, c in enumerate(("#0072B2", "#E69F00", "#009E73")):
        antibody_icon(ax, sx0 + 9.4 + k * 2.0, top + 14.2, 2.6, "ab%d" % k, c)
    label(ax, sx0 + 15.0, top + 14.2, "scBCR-seq", "bcrl", ha="left", width=G[1] - sx0 - 15.8)
    label(ax, sx0 + 8.4, top + 17.6, "clonotype", "bcrl2", ha="left", width=G[1] - sx0 - 9.2)
    xc = sx0 + 15.6                                 # the PCs that passed quality control
    arrow(ax, xc, top + 19.6, xc, top + 24.0, "seq")
    label(ax, xc, top + 26.8, [("{:,}".format(n_pcs), "b")], "npc1", size=TITLE, width=2 * (G[1] - xc) - 1.0)
    label(ax, xc, top + 29.6, "plasma cells (PCs)", "npc2", width=2 * (G[1] - xc) - 1.0)
    arrow(ax, G[1] + 0.4, mid, A[0] - 0.4, mid, "a1")

    # data analysis
    box(ax, A[0], top, A[1] - A[0], bot - top, "ana", "#F7F7F7", "#9E9E9E")
    label(ax, sum(A) / 2, hy, "Data analysis", "h-ana", size=TITLE, bold=True, width=A[1] - A[0])
    c1, c2 = A[0] + (A[1] - A[0]) * 0.25, A[0] + (A[1] - A[0]) * 0.75
    half = (A[1] - A[0]) / 2 - 1.0
    for k, (l1, l2, cx, yy) in enumerate([("Cell annotation,", "clustering", c1, top + 2.8),
                                          ("Heterogeneity by", "genetic alterations", c2, top + 2.8),
                                          ("Coordinated gene", "regulation", c1, top + 21.4),
                                          ("Transcriptional", "signatures", c2, top + 21.4)]):
        label(ax, cx, yy, l1, "an%da" % k, width=half)
        label(ax, cx, yy + 2.6, l2, "an%db" % k, width=half)
    umap_icon(ax, c1, top + 12.6, 11.6, "anumap")
    cells_icon(ax, c2, top + 12.6, 5.6, "ancells", ["#E69F00", "#0072B2", "#CC79A7", "#009E73", "#F0E442"])
    volcano_icon(ax, c1 - 6.0, top + 26.8, 12.0, 11.0, "volc")
    heatmap_icon(ax, c2 - 5.5, top + 26.8, 11.0, 11.0, "anhm")
    arrow(ax, A[1] + 0.4, mid, S[0] - 0.4, mid, "a2")

    # single-cell scores: the two random forest classifiers and the scale of each score
    box(ax, S[0], top, S[1] - S[0], bot - top, "sco", "#F7F7F7", "#9E9E9E")
    cs = sum(S) / 2
    label(ax, cs, hy, "Single-cell scores", "h-sco", size=TITLE, bold=True, width=S[1] - S[0])
    tree_icon(ax, cs - 6.5, top + 2.0, 7.0, 6.0, "dt1", [P.PC_STATUS["BM normal"], P.PC_STATUS["BM malignant"]] * 2)
    tree_icon(ax, cs + 6.5, top + 2.0, 7.0, 6.0, "dt2", [P.TUMOR["MGUS"], P.TUMOR["MM"]] * 2)
    label(ax, cs, top + 10.6, "Random forest classifiers", "rf", width=S[1] - S[0] - 2.0)
    x0, x1 = S[0] + 1.6, S[1] - 1.6
    label(ax, cs, top + 14.4, [("Plasma cell malignancy score", "b")], "s0", width=x1 - x0)
    label(ax, cs, top + 17.0, "Trained on normal vs malignant PCs", "s0a", width=x1 - x0)
    score_scale(ax, x0, x1, top + 19.8, "sb0", P.CMAP_MALIGNANCY, "Normal", "Malignant", P.PC_STATUS["BM normal"],
                P.PC_STATUS["BM malignant"])
    label(ax, cs, top + 26.2, [("MST score", "b")], "s1", width=x1 - x0)
    label(ax, cs, top + 28.8, "Myeloma single-cell transcription", "s1a", width=x1 - x0)
    label(ax, cs, top + 31.4, "Trained on MGUS vs MM malignant PCs", "s1b", width=x1 - x0)
    score_scale(ax, x0, x1, top + 34.2, "sb1", P.CMAP_MST, "MGUS-like", "MM-like", P.TUMOR["MGUS"], P.TUMOR["MM"])
    return ax
