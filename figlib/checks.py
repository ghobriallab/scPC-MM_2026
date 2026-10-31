"""Automatic checks run on every figure, at draw time and on the saved files.

Purpose:      Draw time (figure_checks): every drawn text is Arial, 5-7 pt (panel letters 8 pt bold) and black or
              white; it stays inside its panel (and inside its axes when its gid starts with "ax-"), never overlaps
              another text, and annotation text (gids starting with "ax-" or "pn-") never covers data (points, lines,
              boxes, bars, cells); lines are at least 0.25 pt; raster layers exist only where whitelisted (gid starting
              with "raster-") and images have at least 300 pixels per inch. Files (file_checks): SVG text is Arial and
              editable, no Inkscape namespaces; PDF fonts are TrueType (no Type 3), page = figure size, embedded images
              >= 300 dpi; no local paths or e-mail addresses in either file.

Inputs:       A drawn matplotlib Figure and its saved SVG and PDF.

Outputs:      Lists of problems, and one report per figure in figures/checks/<figure>.txt.

Dependencies: numpy, matplotlib, PyMuPDF; figstyle.py and paths.py.
"""
import io
import os
import re

import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh, LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import SubFigure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.text import Text

import paths
from figstyle import FONT, SMALL, TITLE, LETTER

TOL = 0.6                     # pixels of tolerance at the check resolution
FORBIDDEN = [r"/Users/", r"/mnt/", r"/home/", r"gs://", r"@[a-z]+\.(org|com|edu)"]


DRAWN_MARKS = []


def _drawn_texts(fig):
    """Draw once and record every text artist actually rendered (with its window extent), and every line or patch
    drawn (DRAWN_MARKS: ticks outside the view, for example, exist but are not drawn)."""
    seen = []
    orig = RendererAgg.draw_text
    orig_line, orig_patch = Line2D.draw, Patch.draw
    del DRAWN_MARKS[:]

    def spy(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if mtext is not None and s.strip():
            seen.append(mtext)
        return orig(self, gc, x, y, s, prop, angle, ismath=ismath, mtext=mtext)

    def spy_line(self, renderer):
        DRAWN_MARKS.append(self)
        return orig_line(self, renderer)

    def spy_patch(self, renderer):
        DRAWN_MARKS.append(self)
        return orig_patch(self, renderer)

    RendererAgg.draw_text = spy
    Line2D.draw, Patch.draw = spy_line, spy_patch
    try:
        fig.canvas.draw()
    finally:
        RendererAgg.draw_text = orig
        Line2D.draw, Patch.draw = orig_line, orig_patch
    r = fig.canvas.get_renderer()
    out, ids = [], set()
    for t in seen:
        if id(t) in ids:
            continue
        ids.add(id(t))
        out.append((t, t.get_window_extent(r)))
    return out, r


def _owner_bbox(t, fig):
    a = t.axes
    owner = a.figure if a is not None else t.figure
    if isinstance(owner, SubFigure):
        return owner.bbox, owner.get_gid()
    return fig.bbox, "figure"


def _gid(t):
    return t.get_gid() or ""


def _overlap(b1, b2, tol=TOL):
    return (min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > tol) and (min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > tol)


def _inside(b, box, tol=TOL):
    return b.x0 >= box.x0 - tol and b.x1 <= box.x1 + tol and b.y0 >= box.y0 - tol and b.y1 <= box.y1 + tol


def _path_hits(tp, box, filled):
    """Path-box intersection one subpath at a time: a closing code keeps a placeholder vertex at (0, 0), which a test
    on the whole path would join to the shape as a real segment."""
    for poly in tp.to_polygons(closed_only=False):
        if len(poly) and Path(poly).intersects_bbox(box, filled=filled):
            return True
    return False


def _data_hits(ax, bb, r):
    """Does the text box bb cover data drawn in ax (points, lines, patches, cells)? Data clipped to the plot area
    count only inside it."""
    from matplotlib.transforms import Bbox
    hits = []
    shrink = Bbox.from_extents(bb.x0 + TOL, bb.y0 + TOL, bb.x1 - TOL, bb.y1 - TOL)
    inner = Bbox.intersection(shrink, ax.bbox)
    for c in ax.collections:
        if not c.get_visible() or (c.get_gid() or "").startswith("bg-"):
            continue
        box = inner if c.get_clip_on() else shrink
        if box is None:
            continue
        if isinstance(c, PathCollection):
            off = c.get_offset_transform().transform(c.get_offsets())
            sizes = c.get_sizes()
            rad = np.sqrt(sizes if len(sizes) else [9.0]) / 2.0 * r.dpi / 72.0
            rad = np.broadcast_to(rad, (len(off),)) if len(off) else rad
            inside = ((off[:, 0] + rad > box.x0) & (off[:, 0] - rad < box.x1) &
                      (off[:, 1] + rad > box.y0) & (off[:, 1] - rad < box.y1)) if len(off) else []
            if np.any(inside):
                hits.append("points")
        elif isinstance(c, (PolyCollection, QuadMesh, LineCollection)):
            for p in c.get_paths():
                tp = p.transformed(c.get_transform())
                if tp.intersects_bbox(box, filled=not isinstance(c, LineCollection)):
                    hits.append(type(c).__name__)
                    break
    for ln in ax.lines:
        box = inner if ln.get_clip_on() else shrink
        if box is None or not ln.get_visible():
            continue
        if ln.get_linestyle() not in ("None", " ", "") and not (ln.get_gid() or "").startswith("bg-"):
            tp = ln.get_path().transformed(ln.get_transform())
            if tp.intersects_bbox(box, filled=False):
                hits.append("line")
        if ln.get_marker() not in (None, "None", "", " "):
            xy = ln.get_transform().transform(ln.get_xydata())
            rad = ln.get_markersize() / 2.0 * r.dpi / 72.0
            if np.any((xy[:, 0] + rad > box.x0) & (xy[:, 0] - rad < box.x1) &
                      (xy[:, 1] + rad > box.y0) & (xy[:, 1] - rad < box.y1)):
                hits.append("markers")
    for p in ax.patches:
        box = inner if p.get_clip_on() else shrink
        if box is not None and p.get_visible() and not (p.get_gid() or "").startswith("bg-"):
            tp = p.get_path().transformed(p.get_transform())
            # an arrow's path is a line plus a head: testing it as a filled polygon would close it through other text
            if _path_hits(tp, box, p.get_fill() and not isinstance(p, FancyArrowPatch)):
                hits.append("patch")
    for im in ax.images:
        if im.get_visible() and inner is not None and _overlap(im.get_window_extent(r), inner, 0):
            hits.append("image")
    return hits


def _foreign_hits(fig, t, bb, r):
    """Does a text (tick labels included) cover the frame or the data of an axes other than its own?"""
    hits = []
    for ax in fig.axes:
        if ax is t.axes or not ax.get_visible() or not _overlap(bb, ax.bbox, 0):
            continue
        if ax.axison:
            for name, sp in ax.spines.items():
                if sp.get_visible() and sp.get_linewidth() > 0:
                    tp = sp.get_path().transformed(sp.get_transform())
                    if _path_hits(tp, bb, False):
                        hits.append("frame of another plot")
                        break
        hits += ["%s of another plot" % h for h in _data_hits(ax, bb, r)]
    return hits


def _owner_panel(a, fig):
    f = a.axes.figure if getattr(a, "axes", None) is not None else a.figure
    return f.bbox if isinstance(f, SubFigure) else fig.bbox


def figure_checks(fig, lay):
    """Return a list of problems found on the drawn figure (empty list = all good)."""
    issues = []
    texts, r = _drawn_texts(fig)
    for t, bb in texts:
        g = _gid(t)
        s = t.get_text()
        size = t.get_fontproperties().get_size_in_points()
        name = t.get_fontproperties().get_name()
        if g.startswith("letter-"):
            if abs(size - LETTER) > 0.01 or t.get_fontweight() not in ("bold", 700):
                issues.append("letter %r not %g pt bold" % (s, LETTER))
        elif not (SMALL - 0.01 <= size <= TITLE + 0.01):
            issues.append("text %r is %.2f pt" % (s, size))
        if name not in (FONT, "Liberation Sans"):           # Liberation Sans: Arial's metric twin (figstyle)
            issues.append("text %r in font %s" % (s, name))
        if "$" in s:
            issues.append("mathtext in %r (exponents shrink below 5 pt): use rich_text" % s)
        c = to_rgba(t.get_color())
        if not (np.allclose(c[:3], 0, atol=0.02) or np.allclose(c[:3], 1, atol=0.02)):
            issues.append("colored text %r (%s)" % (s, t.get_color()))
        box, owner = _owner_bbox(t, fig)
        if not _inside(bb, box):
            issues.append("text %r leaves its panel %s" % (s, owner))
        if not _inside(bb, fig.bbox):
            issues.append("text %r leaves the page" % s)
        if g.startswith("ax-") and t.axes is not None and not _inside(bb, t.axes.bbox):
            issues.append("text %r leaves its plot area (%s)" % (s, owner))
        if (g.startswith("ax-") or g.startswith("pn-")) and t.axes is not None:
            hits = _data_hits(t.axes, bb, r)
            if hits:
                issues.append("text %r covers %s (%s)" % (s, ", ".join(sorted(set(hits))), owner))
        if not g.startswith("letter-"):
            hits = _foreign_hits(fig, t, bb, r)
            if hits:
                issues.append("text %r covers the %s (%s)" % (s, ", ".join(sorted(set(hits))), owner))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (t1, b1), (t2, b2) = texts[i], texts[j]
            g1, g2 = _gid(t1), _gid(t2)
            if "__" in g1 and "__" in g2 and g1.rsplit("__", 1)[0] == g2.rsplit("__", 1)[0]:
                continue                      # runs of one rich-text label
            if _overlap(b1, b2):
                issues.append("texts overlap: %r / %r" % (t1.get_text(), t2.get_text()))
    for a in list(DRAWN_MARKS):
        # unclipped marks (keys, arrows, markers placed outside a plot) must stay inside their panel
        if not a.get_visible() or (a.get_clip_on() and (a.get_clip_box() is not None or a.get_clip_path() is not None)):
            continue
        if isinstance(a, Patch) and a.figure is not None and a is getattr(a.figure, "patch", None):
            continue
        if isinstance(a, Patch) and getattr(a, "axes", None) is not None and a is a.axes.patch:
            continue
        try:
            eb = a.get_window_extent(r)
        except Exception:
            continue
        if eb.width <= 0 and eb.height <= 0:
            continue
        if not _inside(eb, _owner_panel(a, fig), tol=1.0):
            issues.append("mark leaves its panel (%s %s)" % (type(a).__name__, a.get_gid() or ""))
    for a in fig.findobj():
        g = a.get_gid() or "" if hasattr(a, "get_gid") else ""
        if isinstance(a, Line2D) and a.get_visible() and a.get_linestyle() not in ("None", " ", "") \
                and 0 < a.get_linewidth() < 0.25:
            issues.append("line thinner than 0.25 pt (%s)" % g)
        if isinstance(a, Patch) and a.get_visible() and a.get_linewidth() and 0 < a.get_linewidth() < 0.25 \
                and to_rgba(a.get_edgecolor())[3] > 0:
            issues.append("edge thinner than 0.25 pt (%s)" % g)
        rast = (hasattr(a, "get_rasterized") and a.get_rasterized()) or isinstance(a, AxesImage)
        if rast and not g.startswith("raster-"):
            issues.append("raster layer not whitelisted: %s %s" % (type(a).__name__, g))
        if isinstance(a, AxesImage):
            arr = a.get_array()
            bb = a.get_window_extent(r)
            if arr is not None and bb.width > 0:
                ppi = arr.shape[1] / (bb.width / r.dpi)
                if ppi < 300:
                    issues.append("image %s only %.0f ppi" % (g, ppi))
    if lay.width > 183.0 + 1e-6 or lay.height > lay.max_height + 1e-6:
        issues.append("figure %.1f x %.1f mm exceeds the limit" % (lay.width, lay.height))
    return issues


def _scan(text, what):
    out = []
    text = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=\s]+", "", text)
    for pat in FORBIDDEN:
        if re.search(pat, text):
            out.append("%s contains %r" % (what, pat))
    return out


def file_checks(svg, pdf, lay):
    """Checks on the saved SVG and PDF."""
    import fitz
    issues = []
    s = open(svg, encoding="utf-8").read()
    issues += _scan(s, os.path.basename(svg))
    if "inkscape" in s or "sodipodi" in s:
        issues.append("SVG has Inkscape namespaces")
    fams = set(re.findall(r"font-family:\s*'?([^;'\"]+)'?", s))
    if fams - {FONT}:
        issues.append("SVG font families %s" % sorted(fams))
    n_text = len(re.findall(r"<text\b", s))
    if n_text == 0:
        issues.append("SVG has no <text>")
    doc = fitz.open(pdf)
    w_mm, h_mm = doc[0].rect.width / 72 * 25.4, doc[0].rect.height / 72 * 25.4
    if abs(w_mm - lay.width) > 0.3 or abs(h_mm - lay.height) > 0.3:
        issues.append("PDF page %.1f x %.1f mm, layout %.1f x %.1f" % (w_mm, h_mm, lay.width, lay.height))
    for page in doc:
        for f in page.get_fonts(full=True):
            ftype, base = f[2], f[3]
            if ftype == "Type3":
                issues.append("PDF Type 3 font %s" % base)
            if "Arial" not in base:
                issues.append("PDF font %s" % base)
        for im in page.get_image_info():
            bw = (im["bbox"][2] - im["bbox"][0]) / 72.0
            if bw > 0 and im["width"] / bw < 299:
                issues.append("PDF image %.0f dpi" % (im["width"] / bw))
        issues += _scan(page.get_text(), os.path.basename(pdf))
    meta = " ".join(str(v) for v in doc.metadata.values() if v)
    issues += _scan(meta, "PDF metadata")
    doc.close()
    return issues, n_text


def report(name, issues, extra=""):
    path = os.path.join(paths.CHECKS, name + ".txt")
    with open(path, "w") as f:
        f.write("%s: %s\n" % (name, "PASS" if not issues else "%d problem(s)" % len(issues)))
        for i in issues:
            f.write("  - %s\n" % i)
        if extra:
            f.write(extra + "\n")
    return path
