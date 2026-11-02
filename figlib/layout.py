"""One matplotlib Figure per figure at its exact print size, one SubFigure per panel, panel letters, saving.

Purpose:      A figure layout is a list of Slot(key, letter, x, y, w, h) in millimetres from the top-left corner of the
              figure. Each slot becomes a SubFigure on a 0.5-mm grid; its gid is the panel key, so the SVG keeps one
              named group per panel. Axes are placed inside a panel in millimetres with axes_mm(). Panel letters exist
              only here, so a renumbering never touches the drawing code.

Inputs:       None.

Outputs:      save() writes <figure>.svg and <figure>.pdf to figures/ from the same Figure (page = figure = the panels
              only, raster layers at 600 dpi, no dates in the metadata, so reruns are byte-identical), a PNG preview to
              figures/previews and the slot geometry to figures/checks.

Dependencies: matplotlib, PyMuPDF (previews); figstyle.py and paths.py.
"""
import json
import os
from collections import namedtuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import MM, LETTER, FONT, DPI
import paths

Slot = namedtuple("Slot", "key letter x y w h")
GRID = 0.5                                   # mm per grid cell


class FigureLayout:
    def __init__(self, name, slots, width=183.0, height=None, max_height=170.0):
        self.name, self.slots = name, list(slots)
        self.width = width
        self.height = height if height is not None else max(s.y + s.h for s in self.slots)
        self.max_height = max_height
        for s in self.slots:
            assert s.x >= 0 and s.y >= 0 and s.x + s.w <= self.width + 1e-6 and s.y + s.h <= self.height + 1e-6, s
        assert self.width <= 183.0 + 1e-6, "wider than 183 mm"
        assert self.height <= self.max_height + 1e-6, "%s is %.1f mm tall (limit %.0f)" % (name, self.height, max_height)

    def slot(self, key):
        return [s for s in self.slots if s.key == key][0]


def new_figure(lay):
    fig = plt.figure(figsize=(lay.width * MM, lay.height * MM))
    nx, ny = int(round(lay.width / GRID)), int(round(lay.height / GRID))
    gs = fig.add_gridspec(ny, nx, left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    panels = {}
    for s in lay.slots:
        x0, x1 = int(round(s.x / GRID)), int(round((s.x + s.w) / GRID))
        y0, y1 = int(round(s.y / GRID)), int(round((s.y + s.h) / GRID))
        sf = fig.add_subfigure(gs[y0:y1, x0:x1])
        sf.set_facecolor("none")
        sf.set_gid(s.key)
        sf.w_mm, sf.h_mm = (x1 - x0) * GRID, (y1 - y0) * GRID
        panels[s.key] = sf
    for s in lay.slots:
        if s.letter:
            t = fig.text(s.x / lay.width, 1 - s.y / lay.height, s.letter, fontsize=LETTER, fontweight="bold",
                         fontfamily=FONT, ha="left", va="top", transform=fig.transFigure)
            t.set_gid("letter-" + s.letter)
    fig.layout_name = lay.name
    return fig, panels


def axes_mm(sf, x, y, w, h, **kw):
    """Axes at (x, y) mm from the top-left corner of the panel, w x h mm."""
    W, H = sf.w_mm, sf.h_mm
    assert x >= -1e-6 and y >= -1e-6 and x + w <= W + 1e-6 and y + h <= H + 1e-6, (sf.get_gid(), x, y, w, h, W, H)
    ax = sf.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)
    ax.size_mm = (w, h)                  # known size: layout code does not depend on when matplotlib resolves positions
    return ax


def unique_gids(fig):
    """SVG ids must be unique: a gid used more than once gets a numeric suffix (prefixes such as "ax-" are kept)."""
    seen = {}
    for a in fig.findobj():
        g = a.get_gid() if hasattr(a, "get_gid") else None
        if not g:
            continue
        if g in seen:
            seen[g] += 1
            a.set_gid("%s_%d" % (g, seen[g]))
        else:
            seen[g] = 0


AUTHOR = "No\u00e9 Perron"                                            # figure author (SVG and PDF metadata)


def save(fig, lay, stem=None, out_dir=None, preview_dpi=150):
    """Write <stem>.svg and <stem>.pdf (figures/), a PNG preview and the slot geometry (figures/previews, figures/checks)."""
    stem = stem or lay.name
    unique_gids(fig)
    out_dir = out_dir or paths.FIGURES
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    svg = os.path.join(out_dir, stem + ".svg")
    pdf = os.path.join(out_dir, stem + ".pdf")
    meta_svg = {"Date": None, "Creator": AUTHOR}                         # author in the file metadata; no dates
    meta_pdf = {"CreationDate": None, "ModDate": None, "Author": AUTHOR}
    for path, fmt, meta in ((svg, "svg", meta_svg), (pdf, "pdf", meta_pdf)):
        # written to a temporary file, then moved into place (no leading dot: some synced folders hide dot-files)
        tmp = os.path.join(out_dir, "%s.tmp.%s" % (stem, fmt))
        fig.savefig(tmp, format=fmt, dpi=DPI, metadata=meta)
        if fmt == "svg":
            svg_preserve_space(tmp)
        os.replace(tmp, path)
    geo = {"name": stem, "width_mm": lay.width, "height_mm": lay.height,
           "slots": [s._asdict() for s in lay.slots]}
    with open(os.path.join(paths.CHECKS, stem + "_layout.json"), "w") as f:
        json.dump(geo, f, indent=1)
    if preview_dpi:
        render_png(pdf, os.path.join(paths.PREVIEWS, stem + ".png"), preview_dpi)
    return svg, pdf


def svg_preserve_space(path):
    """Text runs such as " = 0.031" start with a space: SVG editors drop leading and trailing spaces unless the document
    asks to keep them (xml:space="preserve" on the root element)."""
    with open(path, encoding="utf-8") as f:
        s = f.read()
    i = s.index("<svg ")
    if 'xml:space="preserve"' not in s[i:s.index(">", i)]:
        s = s[:i] + '<svg xml:space="preserve" ' + s[i + 5:]
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)


def render_png(pdf, png, dpi=150, page=0):
    import fitz
    doc = fitz.open(pdf)
    pix = doc[page].get_pixmap(dpi=dpi)
    pix.save(png)
    doc.close()
    return png
