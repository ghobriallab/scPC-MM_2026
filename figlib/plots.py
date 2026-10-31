"""Drawing helpers shared by the figure scripts.

Purpose:      Every helper draws into a given Axes at its final size. Annotation text carries a gid starting with "ax-"
              (must stay inside the plot area and off every drawn element) or "pn-" (inside the panel, off every drawn
              element), which checks.py verifies; only deliberate backgrounds (gid "bg-": shaded bands, color-bar
              gradients) may sit under text, and text written on purpose over cells (gid "cell-", e.g. counts in a
              confusion matrix) is not checked against them.
                box_strip    box plot without outliers + jittered points, n under every label
                brackets     significance brackets stacked without overlap, exact P or q above each
                refline      gray dashed reference line (score of 0.5)
                umap         one rasterized scatter per axes (only the dots are raster), framed, UMAP1/UMAP2
                heatmap      vector cells (pcolormesh) or, for dense matrices, one raster image; colorbar()
                km           Kaplan-Meier curves as survminer (log CI, censor ticks, dotted medians); at_risk(); logrank_p()
                forest       hazard ratios with 95% CI on a log axis
                hbar         horizontal bars (coefficients, gene scores)
                key          colored marks with black labels

Inputs:       None (the figure scripts pass the data).

Outputs:      Drawing functions imported by the figure scripts.

Dependencies: numpy, matplotlib; figstyle.py and palettes.py.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, offset_copy

import palettes as P
from figstyle import BASE, SMALL, LW, rich_text, rich_width_pt, p_runs, n_runs, text_advance_pt, rng


def root(artist_or_fig):
    f = artist_or_fig.figure if getattr(artist_or_fig, "figure", None) is not None else artist_or_fig
    while getattr(f, "figure", None) is not None and f.figure is not f:
        f = f.figure
    return f


def axes_size_pt(ax):
    if getattr(ax, "size_mm", None):                    # axes placed with layout.axes_mm()
        return ax.size_mm[0] * 72.0 / 25.4, ax.size_mm[1] * 72.0 / 25.4
    f = root(ax)
    bb = ax.get_window_extent(f.canvas.get_renderer())
    return bb.width * 72.0 / f.dpi, bb.height * 72.0 / f.dpi


def pts(ax, x=0.0, y=0.0, base=None):
    """`base` (default ax.transData) shifted by (x, y) points; independent of the output resolution."""
    return offset_copy(base if base is not None else ax.transData, fig=root(ax), x=x, y=y, units="points")


def at_mm(sf, x, y, runs, gid, size=BASE, ha="left", va="center", **kw):
    """Text (a string or rich-text runs) at (x, y) mm from the panel's top-left corner."""
    return rich_text(sf, x / sf.w_mm, 1 - y / sf.h_mm, runs if isinstance(runs, list) else [(runs, "")], size=size, ha=ha,
                     va=va, transform=sf.transSubfigure, gid=gid, **kw)


# ---- box + strip ------------------------------------------------------------------------------------------------
def box_strip(ax, data, order, colors, key, *, value="value", group="group", jitter=0.2, width=0.6, point=2.2,
              edge=0.3, points=True, highlight=None, highlight_style=None, n_labels="below", rotate=0, labels=None,
              label_size=BASE, n_size=SMALL, n_offset_pt=None, jitter_col=None, box_face=None, point_col=None,
              point_colors=None, horizontal=False):
    """data: DataFrame with one row per individual (or unit), columns `group` and `value`. order: x order; colors: dict.
    highlight: name of a boolean column drawn as dark diamonds (e.g. later progression) instead of round points.
    n_labels: "below" (n = .. under each horizontal label), "row" (a row of counts under the axis, rotated labels below
    it) or None. jitter_col: precomputed offsets (same x for a patient in two plots); box_face: one fill for all boxes;
    point_col + point_colors: color the points by another column (e.g. disease stage over white boxes); horizontal:
    boxes along y (labels and n are then left to the caller). Returns {position: highest value drawn} for brackets()."""
    r = rng(key)
    tops, ns = {}, []
    for i, g in enumerate(order):
        sub = data.loc[(data[group] == g) & data[value].notna()]
        v = sub[value].values.astype(float)
        ns.append(len(v))
        if len(v) == 0:
            continue
        c = colors[g] if isinstance(colors, dict) else colors
        bp = ax.boxplot([v], positions=[i], widths=width, patch_artist=True, showfliers=False, whis=1.5,
                        vert=not horizontal, medianprops=dict(color="black", linewidth=0.6),
                        boxprops=dict(facecolor=box_face or c, edgecolor=P.BOX_EDGE, linewidth=LW),
                        whiskerprops=dict(color=P.BOX_EDGE, linewidth=LW), capprops=dict(color=P.BOX_EDGE, linewidth=LW),
                        manage_ticks=False, zorder=2)
        for art in bp["boxes"]:
            art.set_gid("box-%s-%d" % (key, i))
        tops[i] = float(np.max(v))
        if points:
            hl = sub[highlight].astype(bool).values if highlight else np.zeros(len(sub), bool)
            x = i + (sub[jitter_col].values if jitter_col else r.uniform(-jitter, jitter, len(sub)))
            pc = np.array([point_colors[k] for k in sub[point_col]]) if point_col else np.array([c] * len(sub))
            xy = (v, x) if horizontal else (x, v)
            ax.scatter(xy[0][~hl], xy[1][~hl], s=point ** 2, c=list(pc[~hl]), edgecolors="black", linewidths=edge,
                       zorder=3)
            if hl.any():
                st = dict(marker="D", s=(point * 1.2) ** 2, c=P.PROGRESSOR, edgecolors="black", linewidths=edge)
                st.update(highlight_style or {})
                ax.scatter(xy[0][hl], xy[1][hl], zorder=4, **st)
    if horizontal:
        ax.set_ylim(len(order) - 0.4, -0.6)
        return tops
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)))
    if rotate:
        ax.set_xticklabels(labels or order, rotation=rotate, fontsize=label_size, ha="right" if rotate != 90 else "center",
                           rotation_mode="anchor" if rotate != 90 else "default")
    else:
        ax.set_xticklabels(labels or order, fontsize=label_size)
    if n_labels == "below":
        n_below(ax, ns, n_size, n_offset_pt)
    elif n_labels == "row":
        n_row(ax, ns, n_size)
    return tops


def n_below(ax, ns, size=SMALL, offset_pt=None, xs=None):
    """n = .. centered under each horizontal x label (one extra line)."""
    if offset_pt is None:
        lines = max(t.get_text().count("\n") + 1 for t in ax.get_xticklabels()) if ax.get_xticklabels() else 1
        offset_pt = ax.xaxis.get_tick_padding() + ax.xaxis.get_ticklabels()[0].get_fontsize() * 1.18 * lines + 0.5
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for i, n in zip(xs if xs is not None else range(len(ns)), ns):
        rich_text(ax, i, 0, n_runs(n), size=size, ha="center", va="top", transform=pts(ax, 0, -offset_pt, tr),
                  gid="pn-n%d" % i)


def n_row(ax, ns, size=SMALL, xs=None, rotate=False):
    """A row of counts just under the axis, with an italic n at its left; the (rotated) tick labels are pushed below.
    rotate=True writes the counts vertically (large counts in narrow columns)."""
    labels = [format(int(n), ",") for n in ns]
    row = (max(text_advance_pt(root(ax), t, size) for t in labels) + 1.5) if rotate else size * 1.2 + 1.0
    ax.tick_params(axis="x", pad=ax.xaxis.get_tick_padding() + row)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    dy = -(2.0 + 0.8)
    for i, t in zip(xs if xs is not None else range(len(ns)), labels):
        if rotate:
            ax.text(i, 0, t, rotation=90, ha="center", va="top", fontsize=size, transform=pts(ax, 0, dy, tr),
                    gid="pn-n%d" % i)
        else:
            rich_text(ax, i, 0, [(t, "")], size=size, ha="center", va="top", transform=pts(ax, 0, dy, tr), gid="pn-n%d" % i)
    tl = blended_transform_factory(ax.transAxes, ax.transAxes)
    rich_text(ax, 0, 0, [("n", "i")], size=size, ha="right", va="top", transform=pts(ax, -1.5, dy, tl), gid="pn-nsym")


def refline(ax, y=0.5, horizontal=True):
    kw = dict(color=P.REFLINE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ln = ax.axhline(y, **kw) if horizontal else ax.axvline(y, **kw)
    ln.set_gid("ref-line")
    return ln


# ---- brackets ---------------------------------------------------------------------------------------------------
def brackets(ax, comps, tops, *, ymin=None, size=SMALL, tick_pt=2.0, gap_pt=2.5, text_gap_pt=1.2, key="br",
             headroom_pt=2.0, ytop_min=None, edge_pt=1.5):
    """comps: list of dicts {i, j, p, sym ("P" or "q"), color; optional span (one fork), span_i and span_j (a fork under
    each end: a block of groups against another block)}. Brackets are stacked from the shortest span up, each above every
    point, whisker, bracket and label already in its span (and in the span of its label), at least gap_pt above the
    highest point (clear of its marker); labels stay edge_pt inside the plot frame; the y axis is extended so that
    everything stays inside the plot area. Returns the artists."""
    f = root(ax)
    y0 = ax.get_ylim()[0] if ymin is None else ymin
    data_max = max(tops.values())
    comps = sorted(comps, key=lambda c: (abs(c["j"] - c["i"]), min(c["i"], c["j"])))
    ytop = data_max + 0.04 * (data_max - y0)
    if ytop_min is not None:                 # a common top for plots sharing one y scale (brackets_shared)
        ytop = max(ytop, ytop_min)
    place = []
    for _ in range(80):                      # the scale changes when the axis grows: iterate to a fixed point
        ax.set_ylim(y0, ytop)
        w_pt, h_pt = axes_size_pt(ax)
        upp = (ytop - y0) / h_pt             # data units per point (y)
        xpp = (ax.get_xlim()[1] - ax.get_xlim()[0]) / w_pt
        sky = dict(tops)
        place = []
        for c in comps:
            i, j = sorted((c["i"], c["j"]))
            lines = c.get("lines") or [c.get("runs") or p_runs(c["p"], c.get("sym", "P"))]
            runs = lines
            half = max(rich_width_pt(f, ln, size) for ln in lines) * xpp / 2.0 + 1.0 * xpp
            mid = (i + j) / 2.0
            xa, xb = sorted(ax.get_xlim())
            m = edge_pt * xpp
            if 2 * (half + m) < xb - xa:                 # a label wider than its bracket stays inside the plot
                mid = min(max(mid, xa + half + m), xb - half - m)
            sp = c.get("span") or (i, j)
            ends = [x for s_ in (c.get("span_i"), c.get("span_j")) if s_ for x in s_]
            lo_x, hi_x = min([i, sp[0], mid - half] + ends), max([j, sp[1], mid + half] + ends)
            span = [k for k in sky if lo_x - 0.5 < k < hi_x + 0.5] + [k for k in (i, j) if k not in sky]
            base = max([sky.get(k, y0) for k in span]) + gap_pt * upp
            top_line = base + tick_pt * upp
            text_y = top_line + text_gap_pt * upp
            sup = any("^" in fmt for _, fmt in lines[0])
            line_h = size * 1.12
            text_top = text_y + (len(lines) - 1) * line_h * upp + \
                (0.38 * size + 0.905 * max(SMALL, 0.8 * size) if sup else 0.905 * size) * upp
            for k in range(int(np.floor(lo_x)), int(np.ceil(hi_x)) + 1):
                if lo_x - 0.5 < k < hi_x + 0.5:
                    sky[k] = max(sky.get(k, y0), text_top)
            place.append((c, i, j, base, top_line, text_y, runs, mid))
        need = max(sky.values()) + headroom_pt * upp
        if need <= ytop * (1 + 1e-9) + 1e-12:
            break
        ytop = need
    arts = []
    for n, (c, i, j, base, top_line, text_y, runs, mid) in enumerate(place):
        ln, = ax.plot([i, i, j, j], [base, top_line, top_line, base], color=c.get("color", "black"), lw=LW,
                      solid_capstyle="butt", zorder=4)
        ln.set_gid("br-line%d" % n)
        arts.append(ln)
        for s_ in (c.get("span"), c.get("span_i"), c.get("span_j")):
            if s_ and s_[1] > s_[0]:             # a fork across the groups compared at that end
                fk, = ax.plot([s_[0], s_[1]], [base, base], color=c.get("color", "black"), lw=LW, solid_capstyle="butt",
                              zorder=4)
                fk.set_gid("br-fork%d" % n)
                arts.append(fk)
        for k, ln_runs in enumerate(runs):
            arts += rich_text(ax, mid, text_y, ln_runs, size=size, ha="center", va="baseline",
                              transform=pts(ax, 0, (len(runs) - 1 - k) * size * 1.12, ax.transData),
                              gid="ax-%s%d_%d" % (key, n, k))
    return arts


def brackets_shared(jobs, ymin):
    """Brackets on several plots that share one y scale: `jobs` = [(ax, comps, tops, key), ...]. The common top is found
    first, then every plot gets its brackets at that scale, so the point-based gaps stay exact. Returns the artists."""
    tops_needed = []
    for ax, comps, tops, key in jobs:
        hi = max(tops.values())
        ax.set_ylim(ymin, hi + 0.04 * (hi - ymin))
        if comps:
            for a in brackets(ax, comps, tops, ymin=ymin, key=key):
                a.remove()
        tops_needed.append(ax.get_ylim()[1])
    top = max(tops_needed)
    arts = []
    for ax, comps, tops, key in jobs:
        ax.set_ylim(ymin, top)
        if comps:
            arts.append(brackets(ax, comps, tops, ymin=ymin, key=key, ytop_min=top))
        else:
            arts.append([])
    return arts


def scatter_groups(ax, x, y, groups, colors, key, s=2.4, edge=0.25, order=None):
    """Points colored by group (black edges), drawn in a fixed group order."""
    for g in (order or sorted(set(groups))):
        m = np.asarray(groups) == g
        if m.any():
            ax.scatter(np.asarray(x)[m], np.asarray(y)[m], s=s ** 2, c=[colors[g]], edgecolors="black", linewidths=edge,
                       zorder=3)


def violin(ax, values, x=0.0, width=0.8, color="#D9D9D9", points=True, key="violin", point=1.0):
    """Vertical violin (vector) with jittered points; returns the violin body."""
    vp = ax.violinplot([values], positions=[x], widths=width, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(color)
        b.set_edgecolor(P.BOX_EDGE)
        b.set_linewidth(LW)
        b.set_alpha(1.0)
        b.set_gid("violin-" + key)
    if points:
        r = rng(key)
        ax.scatter(x + r.uniform(-width / 6, width / 6, len(values)), values, s=point ** 2, c="black", linewidths=0,
                   zorder=3)
    return vp


def stage_box(ax, units, col, specs, key, *, refline_at=None, point=1.6, label_size=SMALL, ylabel=None,
              header_size=SMALL, headers=True, labels=True):
    """22-group box plot: BM normal | BM malignant | PB normal | PB malignant, one dot per patient unit, stage
    labels rotated with the number of units in a row under the axis; `specs` from stage_tests.brackets(). Stacked plots
    of the same units can show the block headers only on the top one and the labels and n only on the bottom one."""
    import stage_tests as ST
    colors = {g: (P.NORMAL[g.split(":")[0]] if g.endswith("normal") else P.TUMOR[g.split(":")[0]]) for g in ST.GROUPS}
    d = units[units[col].notna()].assign(value=units[col].astype(float))
    tops = box_strip(ax, d, ST.GROUPS, colors, key, point=point, labels=[g.split(":")[0] for g in ST.GROUPS], rotate=90,
                     label_size=label_size, n_labels="row" if labels else None, width=0.62, jitter=0.18, edge=0.25)
    if not labels:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    if refline_at is not None:
        refline(ax, refline_at)
    comps = []
    for b in specs:
        xs = b["partner_xs"]
        mid = float(np.median(xs))
        sym = [("q", "i"), (" \u2264 " if b["n"] > 1 else " = ", "")]
        q = p_runs(b["q"], "q")[1:]
        runs = sym + [(q[0][0].replace(" = ", ""), "")] + q[1:]
        comps.append(dict(i=b["anchor_x"], j=mid, span=(min(xs), max(xs)) if len(xs) > 1 else None, runs=runs,
                          span_i=b.get("span_i"), span_j=b.get("span_j"),
                          color=P.BRACKET_MAL if b["kind"] == "tvn" else P.BRACKET_STAGE))
    arts = brackets(ax, comps, tops, ymin=ax.get_ylim()[0], key=key) if comps else []
    separators(ax, (5.5, 10.5, 16.5), arts)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x, lab in ([(2.5, "BM normal"), (8, "BM malignant"), (13.5, "PB normal"), (19, "PB malignant")] if headers else []):
        rich_text(ax, x, 1, [(lab, "")], size=header_size, ha="center", va="bottom", transform=pts(ax, 0, 1.2, tr),
                  gid="pn-hdr%s" % lab.replace(" ", ""))
    if ylabel:
        ax.set_ylabel(ylabel)
    return tops


def separators(ax, xs, obstacles, gap_pt=1.0):
    """Dashed vertical lines between blocks of groups, from the x axis up to just below any bracket or bracket label
    above them (`obstacles`: the artists returned by brackets())."""
    f = root(ax)
    r = f.canvas.get_renderer()
    y0, y1 = ax.get_ylim()
    upp = (y1 - y0) / axes_size_pt(ax)[1]
    inv = ax.transData.inverted()
    spans = []
    for a in obstacles:
        if isinstance(a, Line2D):
            spans.append((min(a.get_xdata()), max(a.get_xdata()), min(a.get_ydata())))
        else:
            bb = a.get_window_extent(r)
            (xa, ya), (xb, yb) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            spans.append((min(xa, xb), max(xa, xb), min(ya, yb)))
    for x in xs:
        low = [yb for xa, xb, yb in spans if xa - 0.1 <= x <= xb + 0.1]
        top = min(low) - gap_pt * upp if low else y1
        ln, = ax.plot([x, x], [y0, top], color="black", lw=0.4, ls=(0, (2, 1.5)), zorder=1)
        ln.set_gid("ref-sep")


def plain_ticks(axis):
    """Tick labels without trailing zeros and with a true minus sign (0, 0.2, \u22120.2)."""
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v).replace("-", "\u2212") if abs(v) > 1e-12 else "0"))


def range_ticks(vmin, vmax):
    """Ticks for a color bar that spans one variable's own range: 0 (or the minimum, rounded up) and the maximum rounded
    down to two significant digits, so that the top of each scale can be read."""
    def two_digits(v, up):
        if v == 0:
            return 0.0
        e = 10.0 ** (np.floor(np.log10(abs(v))) - 1)
        return float((np.ceil(v / e) if up else np.floor(v / e)) * e)
    lo = 0.0 if vmin <= 0 <= vmax else two_digits(vmin, True)
    hi = two_digits(vmax, False)
    return [lo, hi] if hi > lo else [lo]


def nice_locator(n):
    """At most n + 1 round ticks (1, 2, 2.5 or 5 x 10^k)."""
    from matplotlib.ticker import MaxNLocator
    return MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10])


def arrow_line(ax, x0, y0, x1, y1, color, key, lw=1.0):
    """A colored arrow from (x0, y0) to (x1, y1) in data coordinates, with a dot at its start."""
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=2.6,head_width=1.3", mutation_scale=1.0, lw=lw,
                        color=color, shrinkA=0, shrinkB=0, zorder=3)
    a.set_gid("arrow-" + key)
    ax.add_patch(a)
    ax.plot([x0], [y0], marker="o", ms=2.0, color=color, ls="none", zorder=3)[0].set_gid("arrowstart-" + key)
    return a


# ---- SHAP -------------------------------------------------------------------------------------------------------
def red_blue():
    """shap's red-blue color map (low feature value blue, high red)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("shap_red_blue", ["#008bfb", "#ff0052"])


def shap_violin(ax, shap, values, names, key, max_display=20, alpha=0.4, dot=0.55):
    """shap.plots.violin (shap 0.42, plot_type "violin") re-implemented: per feature, the cells' SHAP values as dots
    colored by the feature value (raster layer) under a violin whose segments are colored by the running mean of the
    feature value (vector). Features ordered by mean absolute SHAP value; returns the displayed order."""
    from scipy.stats import gaussian_kde
    cmap = red_blue()
    order = np.argsort(-np.abs(shap).mean(0))[:max_display]
    lo_all, hi_all = np.nanpercentile(shap, 1), np.nanpercentile(shap, 99)
    r = rng(key)
    xs_all, ys_all, cs_all = [], [], []
    for pos, i in enumerate(order):
        sh, v = shap[:, i].astype(float), values[:, i].astype(float)
        y = len(order) - 1 - pos
        vmin, vmax = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        c = np.clip((np.nan_to_num(v, nan=(vmin + vmax) / 2) - vmin) / ((vmax - vmin) or 1), 0, 1)
        xs_all.append(sh)
        ys_all.append(np.full(len(sh), y) + r.uniform(-0.04, 0.04, len(sh)))
        cs_all.append(c)
        span = sh.max() - sh.min()
        xs = np.linspace(sh.min() - span * 0.2, sh.max() + span * 0.2, 100)
        if np.std(sh) < (hi_all - lo_all) / 100:
            ds = gaussian_kde(sh + r.standard_normal(len(sh)) * (hi_all - lo_all) / 100)(xs)
        else:
            ds = gaussian_kde(sh)(xs)
        ds /= np.max(ds) * 3
        srt = np.argsort(sh)
        smooth = np.zeros(len(xs) - 1)
        lead = trail = back = 0
        run = 0.0
        for j in range(len(xs) - 1):                 # running mean of the 20 nearest cells, as in shap
            while lead < len(sh) and xs[j] >= sh[srt[lead]]:
                run += v[srt[lead]]
                lead += 1
                if lead - trail > 20:
                    run -= v[srt[trail]]
                    trail += 1
            if lead - trail > 0:
                smooth[j] = run / (lead - trail)
                for k in range(back):
                    smooth[j - k - 1] = smooth[j]
                back = 0
            else:
                back += 1
        smooth = np.clip((smooth - vmin) / ((vmax - vmin) or 1), 0, 1)
        for j in range(len(xs) - 1):
            if ds[j] > 0.05 or ds[j + 1] > 0.05:
                pc = ax.fill_between([xs[j], xs[j + 1]], [y + ds[j], y + ds[j + 1]], [y - ds[j], y - ds[j + 1]],
                                     color=cmap(smooth[j]), linewidth=0, zorder=2)
                pc.set_gid("violin-%s-%d-%d" % (key, pos, j))
    sc = ax.scatter(np.concatenate(xs_all), np.concatenate(ys_all), c=np.concatenate(cs_all), cmap=cmap, vmin=0, vmax=1,
                    s=dot ** 2, alpha=alpha, linewidths=0, rasterized=True, zorder=1)
    sc.set_gid("raster-" + key)
    ax.axvline(0, color="#999999", lw=0.5, zorder=0).set_gid("ref-zero")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order][::-1])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis="y", length=0)
    for sp_ in ("top", "right", "left"):
        ax.spines[sp_].set_visible(False)
    return order


def confusion(ax, counts, labels, key, cmap, fmt, vmin=None, vmax=None, center=None):
    """2 x 2 matrix with its values written in the cells (white on dark cells); fmt: %-format or function."""
    from matplotlib.colors import TwoSlopeNorm, Normalize
    norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax) if center is not None else Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(np.arange(3), np.arange(3), counts, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.4)
    mesh.set_gid("mesh-" + key)
    cm = matplotlib_cm(cmap)
    for i in range(2):
        for j in range(2):
            rgba = cm(norm(counts[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j + 0.5, i + 0.5, fmt(counts[i, j]) if callable(fmt) else fmt % counts[i, j], ha="center",
                    va="center", fontsize=BASE,
                    color="white" if lum < 0.5 else "black", gid="cell-%s%d%d" % (key, i, j))
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.tick_params(length=0)
    return mesh, norm


def matplotlib_cm(cmap):
    import matplotlib
    return matplotlib.colormaps[cmap] if hasattr(matplotlib, "colormaps") and isinstance(cmap, str) else \
        (matplotlib.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap)


# ---- cells: violins and rank tests -----------------------------------------------------------------------------------
def mwu_log10p(x, y):
    """Two-sided Mann-Whitney test, normal approximation with tie and continuity corrections (scipy's asymptotic
    method), with P returned as log10 so that it stays exact below 10^-300. Returns (log10 P, U of x)."""
    from scipy import stats
    n1, n2 = len(x), len(y)
    r = stats.rankdata(np.concatenate([x, y]))
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    u = max(u1, n1 * n2 - u1)
    _, t = np.unique(r, return_counts=True)
    n = n1 + n2
    sd = np.sqrt(n1 * n2 / 12.0 * ((n + 1) - (t ** 3 - t).sum() / (n * (n - 1.0))))
    zv = (u - n1 * n2 / 2.0 - 0.5) / sd
    return (np.log(2.0) + stats.norm.logsf(zv)) / np.log(10.0), u1


def violin_box(ax, v, x, color, key, width=0.8, ink=P.BOX_EDGE):
    """Seaborn-style violin (Scott bandwidth, cut at the data range) with a box inside: whiskers to 1.5 IQR, thick IQR
    bar, white median dot."""
    from scipy import stats
    kde = stats.gaussian_kde(v)
    ys = np.linspace(v.min(), v.max(), 120)
    d = kde(ys)
    d = d / d.max() * width / 2.0
    body = ax.fill_betweenx(ys, x - d, x + d, facecolor=color, edgecolor=ink, linewidth=0.4, zorder=2)
    body.set_gid("violin-" + key)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    lo = v[v >= q1 - 1.5 * (q3 - q1)].min()
    hi = v[v <= q3 + 1.5 * (q3 - q1)].max()
    ax.plot([x, x], [lo, hi], color=ink, lw=0.5, zorder=3, solid_capstyle="butt")[0].set_gid("vbox-w-" + key)
    ax.plot([x, x], [q1, q3], color=ink, lw=1.6, zorder=3, solid_capstyle="butt")[0].set_gid("vbox-q-" + key)
    ax.plot([x], [med], marker="o", ms=1.2, mfc="white", mec="white", ls="none", zorder=4)[0].set_gid("vbox-m-" + key)


# ---- embeddings -------------------------------------------------------------------------------------------------
def umap(ax, xy, colors, key, s=0.15, order=None, frame=True, labels=True):
    """Only the dots are raster (figstyle.DPI); order: index order in which dots are drawn (last on top)."""
    if order is not None:
        xy = xy[order]
        colors = colors[order] if np.ndim(colors) else colors
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=s, linewidths=0, rasterized=True, zorder=1)
    sc.set_gid("raster-" + key)
    ax.set_xticks([])
    ax.set_yticks([])
    if labels:
        ax.set_xlabel("UMAP1", labelpad=1.0)
        ax.set_ylabel("UMAP2", labelpad=1.0)
    for sp in ax.spines.values():
        sp.set_visible(frame)
    ax.set_aspect("equal", adjustable="datalim")
    return sc


# ---- heatmaps ---------------------------------------------------------------------------------------------------
def heatmap(ax, M, key, cmap, vmin=None, vmax=None, raster=False):
    M = np.asarray(M, float)
    if raster:
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="none",
                       extent=(0, M.shape[1], M.shape[0], 0))
        im.set_gid("raster-" + key)
        return im
    mesh = ax.pcolormesh(np.arange(M.shape[1] + 1), np.arange(M.shape[0] + 1), M, cmap=cmap, vmin=vmin, vmax=vmax,
                         edgecolors="face", linewidth=0)
    mesh.set_gid("mesh-" + key)
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)
    return mesh


def colorbar(host, mappable, cax, label=None, ticks=None, orientation="vertical", size=SMALL):
    cb = host.colorbar(mappable, cax=cax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(LW)
    cb.ax.tick_params(width=LW, length=1.5, pad=1.0, labelsize=size)
    if getattr(cb, "solids", None) is not None:
        cb.solids.set_edgecolor("face")
        cb.solids.set_rasterized(False)
        cb.solids.set_gid("bg-colorbar")
    if label:
        cb.set_label(label, fontsize=size, labelpad=1.5)
    return cb


# ---- survival ---------------------------------------------------------------------------------------------------
def km_table(time, event):
    """Kaplan-Meier estimate with the log-transformed 95% CI of R's survfit (Greenwood variance).
    Columns: time, number at risk, events, censored, survival, lower, upper."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    n_at, s, var, out = len(time), 1.0, 0.0, []
    for t in np.unique(time):
        d = int(((time == t) & (event == 1)).sum())
        c = int(((time == t) & (event == 0)).sum())
        if d:
            s *= 1 - d / n_at
            var += d / (n_at * (n_at - d)) if n_at > d else np.inf
        se = np.sqrt(var)
        lo, hi = (s * np.exp(-1.959964 * se), min(1.0, s * np.exp(1.959964 * se))) if s > 0 else (0.0, 0.0)
        out.append((t, n_at, d, c, s, lo, hi))
        n_at -= d + c
    return np.array(out)


def km(ax, df, order, colors, *, time="time", event="event", group="group", ci=True, medians=True, censor=True):
    """Curves start at (0, 1); censor ticks '|'; dotted median lines as survminer's
    surv.median.line = "hv"."""
    tabs = {}
    for g in order:
        d = df[df[group] == g]
        T = km_table(d[time], d[event])
        tabs[g] = T
        t, s = np.r_[0, T[:, 0]], np.r_[1, T[:, 4]]
        ax.step(t, s, where="post", color=colors[g], lw=0.75, zorder=3)
        if ci:
            pc = ax.fill_between(t, np.r_[1, T[:, 5]], np.r_[1, T[:, 6]], step="post", color=colors[g], alpha=0.18,
                                 linewidth=0, zorder=1)
            pc.set_gid("km-ci")
        if censor:
            cz = T[T[:, 3] > 0]
            ax.plot(cz[:, 0], cz[:, 4], ls="none", marker="|", ms=2.0, mew=0.4, color=colors[g], zorder=4)
        if medians:
            below = np.where(T[:, 4] <= 0.5)[0]
            if len(below):
                tm = T[below[0], 0]
                ln, = ax.plot([0, tm, tm], [0.5, 0.5, 0], color="black", lw=0.4, ls=(0, (1, 1.2)), zorder=2)
                ln.set_gid("km-median")
    ax.set_ylim(0, 1.03)
    return tabs


def logrank_p(df, time="time", event="event", group="group"):
    from lifelines.statistics import multivariate_logrank_test
    return float(multivariate_logrank_test(df[time], df[group], df[event]).p_value)


def at_risk(ax, tabs, order, colors, times, size=SMALL, names=None):
    """Numbers at risk under a KM plot (ax shares the KM x axis): black numbers, a short colored line before each name."""
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    lab_tr = blended_transform_factory(ax.transAxes, ax.transData)
    w0 = max(text_advance_pt(root(ax), "%d" % int(tabs[g][0, 1]), size) for g in order)
    x0_pt = -(w0 / 2.0 + 3.0) if ax.get_xlim()[0] == min(times) else -3.0
    for r_, g in enumerate(order):
        T = tabs[g]
        for t in times:
            n = int(T[T[:, 0] >= t, 1][0]) if np.any(T[:, 0] >= t) else 0
            ax.text(t, r_, "%d" % n, ha="center", va="center", fontsize=size, gid="pn-risk%d-%g" % (r_, t))
        name = (names or {}).get(g, g)
        ax.text(0, r_, name, ha="right", va="center", fontsize=size, transform=pts(ax, x0_pt, 0, lab_tr),
                gid="pn-riskname%d" % r_)
        w = text_advance_pt(root(ax), name, size)
        ax.add_line(Line2D([0], [r_], marker="_", ms=5.0, mew=1.0, color=colors[g], ls="none",
                           transform=pts(ax, x0_pt - w - 4.0, 0, lab_tr), clip_on=False))
    return ax


# ---- forest -----------------------------------------------------------------------------------------------------
def forest(ax, rows, xlim=None, xticks=None, label="Hazard ratio (95% CI)"):
    """rows: list of dicts {hr, lo, hi} top to bottom (hr None for header/reference rows). Points and CI on a log axis;
    the dashed vertical line marks 1."""
    for y, rw in enumerate(rows):
        if rw.get("hr") is None:
            continue
        ax.plot([rw["lo"], rw["hi"]], [y, y], color="black", lw=0.6, zorder=2)
        ax.plot([rw["hr"]], [y], marker="s", ms=2.6, color="black", ls="none", zorder=3)
    ax.axvline(1.0, color=P.REFLINE, lw=0.5, ls=(0, (3, 2)), zorder=1).set_gid("ref-one")
    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels([("%g" % t) for t in xticks])
        ax.minorticks_off()
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(label)


def table_rules(ax, sf, x0, x1, top, rule, bottom, seps=(), color="#4D4D4D"):
    """Lines of a forest table (the layout of the R package forestmodel): outer frame, the rule under the
    header row and vertical column separators. x in mm across the panel, y in the forest axes' row units."""
    tr = blended_transform_factory(sf.transSubfigure, ax.transData)
    segs = [((x0, top), (x1, top)), ((x0, rule), (x1, rule)), ((x0, bottom), (x1, bottom)), ((x0, top), (x0, bottom)),
            ((x1, top), (x1, bottom))] + [((x, top), (x, bottom)) for x in seps]
    lim = ax.get_xlim(), ax.get_ylim()
    for k, ((xa, ya), (xb, yb)) in enumerate(segs):
        ln = Line2D([xa / sf.w_mm, xb / sf.w_mm], [ya, yb], transform=tr, color=color, lw=LW, clip_on=False, zorder=1,
                    solid_capstyle="projecting")
        ln.set_gid("line-table%d" % k)
        ax.add_line(ln)
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])


def forest_text(ax, rows, label_key="label", stat_key="stat", p_key="p", size=BASE, header=None):
    """Text columns of a forest plot: labels left-aligned at the panel's left edge, "HR (95% CI)" right-aligned just
    left of the plot, exact P right of it. Raises if the two left columns would touch."""
    f = root(ax)
    sf = ax.figure
    r = f.canvas.get_renderer()
    left_pt = (ax.get_window_extent(r).x0 - sf.bbox.x0) * 72.0 / f.dpi
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for y, rw in enumerate(rows):
        lab = rw[label_key] if isinstance(rw[label_key], list) else [(rw[label_key], "")]
        st = rw.get(stat_key)
        wl = rich_width_pt(f, lab, size)
        ws = rich_width_pt(f, [(st, "")], size) if st else 0.0
        assert wl + ws + 6.0 <= left_pt, "forest columns too wide (%.0f + %.0f pt > %.0f)" % (wl, ws, left_pt)
        rich_text(ax, 0, y, lab, size=size, ha="left", va="center", transform=pts(ax, -left_pt + 0.5, 0, tr),
                  gid="pn-flab%d" % y)
        if st:
            rich_text(ax, 0, y, [(st, "")], size=size, ha="right", va="center", transform=pts(ax, -3.0, 0, tr),
                      gid="pn-fstat%d" % y)
        if rw.get(p_key) is not None:
            rich_text(ax, 1, y, p_runs(rw[p_key]), size=size, ha="left", va="center", transform=pts(ax, 3.0, 0, tr),
                      gid="pn-fp%d" % y)


# ---- bars and keys ----------------------------------------------------------------------------------------------
def hbar(ax, labels, values, colors, height=0.75, zero=True):
    y = np.arange(len(labels))
    bars = ax.barh(y, values, height=height, color=colors, edgecolor="none", zorder=2)
    for b in bars:
        b.set_gid("bar")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    if zero:
        ax.axvline(0, color="black", lw=LW, zorder=3).set_gid("ref-zero")
    return bars


def key(host, x, y, items, transform=None, size=BASE, marker="o", ms=3.0, row_pt=None, ncol=1, col_pt=None,
        line=False, gid="pn-key", ls="-", gap_pt=5.0):
    """A key of colored marks with black labels; (x, y) = top-left corner in `transform` coordinates (default: the
    host's axes or panel fraction). items: list of (label or rich-text runs, color). Columns are as wide as their own
    labels plus gap_pt (even spacing), unless col_pt fixes one width for all. line=True draws short lines (linestyle ls).
    Returns the height in points."""
    from matplotlib.transforms import Affine2D, ScaledTranslation
    fig = root(host)
    if transform is None:
        transform = host.transAxes if hasattr(host, "transAxes") else host.transSubfigure
    row_pt = row_pt or size * 1.4
    nrow = int(np.ceil(len(items) / float(ncol)))
    mark_w = ms * (1.6 if line else 1.0)
    widths = [rich_width_pt(fig, it[0] if isinstance(it[0], list) else [(it[0], "")], size) for it in items]
    ncols = int(np.ceil(len(items) / float(nrow)))
    if col_pt is None:
        col_w = [max(widths[c * nrow:(c + 1) * nrow]) + mark_w + 1.8 + gap_pt for c in range(ncols)]
    else:
        col_w = [col_pt] * ncols
    starts = np.r_[0, np.cumsum(col_w)[:-1]]
    for k, (lab, col) in enumerate(items):
        c, r_ = divmod(k, nrow)
        ox, oy = starts[c], -(r_ + 0.5) * row_pt
        if line:                                  # a short segment, in points from the key's anchor
            seg = Affine2D().scale(1 / 72.0) + fig.dpi_scale_trans + ScaledTranslation(x, y, transform)
            host.add_artist(Line2D([ox, ox + mark_w], [oy, oy], color=col, lw=1.0, ls=ls, transform=seg, clip_on=False))
        else:
            tr = offset_copy(transform, fig=fig, x=ox + ms / 2.0, y=oy, units="points")
            host.add_artist(Line2D([x], [y], ls="none", transform=tr, clip_on=False, marker=marker, ms=ms, mfc=col,
                                   mec="black", mew=0.25))
        rich_text(host, x, y, lab if isinstance(lab, list) else [(lab, "")], size=size, ha="left", va="center",
                  transform=offset_copy(transform, fig=fig, x=ox + mark_w + 1.8, y=oy, units="points"),
                  gid="%s%d" % (gid, k))
    return nrow * row_pt
