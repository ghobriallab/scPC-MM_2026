"""Figure style shared by every figure script (Nature format).

Purpose:      Arial text of 5-7 pt at final size, 8-pt bold panel letters, lines of 0.25-1 pt, and text kept editable
              (SVG <text>, PDF TrueType fonts). Matplotlib mathtext is not used: rich_text() writes italic symbols and
              superscripts as separate text runs, so that exponents stay at 5 pt or more.

Inputs:       None.

Outputs:      style() (rcParams context), size constants, rich_text(), p_runs(), n_runs() and rng(), used by every
              figure script.

Dependencies: Python 3.8+, numpy, matplotlib 3.5+.
"""
import contextlib
import math
import os
import zlib

import matplotlib
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import offset_copy

MM = 1 / 25.4            # inches per millimetre
PT_PER_MM = 72 / 25.4    # points per millimetre
FONT = "Arial"
BASE = 6.0               # labels, keys, statistics
SMALL = 5.0              # dense tick labels and the smallest keys (Nature minimum)
TITLE = 7.0              # panel titles
LETTER = 8.0             # panel letters
DPI = 600                # resolution of the only raster layers (UMAP and SHAP dots, dense matrices)
LW = 0.5                 # axes, boxes, brackets
MINUS = "−"
TIMES = "×"

RC = {
    "font.family": [FONT],
    "font.size": BASE,
    "axes.titlesize": TITLE,
    "axes.labelsize": BASE,
    "xtick.labelsize": BASE,
    "ytick.labelsize": BASE,
    "legend.fontsize": BASE,
    "legend.title_fontsize": BASE,
    "figure.titlesize": TITLE,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.linewidth": LW,
    "axes.titlepad": 2.5,
    "axes.labelpad": 1.5,
    "axes.unicode_minus": True,
    "axes.facecolor": "none",
    "figure.facecolor": "white",
    "xtick.major.width": LW,
    "ytick.major.width": LW,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.2,
    "ytick.major.pad": 1.2,
    "lines.linewidth": 0.75,
    "lines.markersize": 2.5,
    "patch.linewidth": LW,
    "hatch.linewidth": LW,
    "legend.frameon": False,
    "legend.handlelength": 1.0,
    "legend.handleheight": 0.7,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.8,
    "legend.borderpad": 0.2,
    "legend.borderaxespad": 0.3,
    "image.interpolation": "none",
    "svg.fonttype": "none",            # SVG text stays <text>
    "svg.hashsalt": "figures",         # deterministic SVG ids
    "pdf.fonttype": 42,                # TrueType fonts in the PDF
    "ps.fonttype": 42,
    "savefig.dpi": DPI,
    "savefig.facecolor": "white",
}


def _arial_or_metric_twin():
    """On a machine without Arial (e.g. a Linux server), Liberation Sans, whose glyph widths equal Arial's, is registered
    under the name Arial: the layout is the same and the SVG still names Arial (the PDF is then made from the SVG on a
    machine with Arial)."""
    import dataclasses
    from matplotlib import font_manager as fm
    if any(f.name == FONT for f in fm.fontManager.ttflist):
        return
    for path in fm.findSystemFonts():
        if os.path.basename(path).startswith("LiberationSans-"):
            fm.fontManager.addfont(path)
    twins = [f for f in fm.fontManager.ttflist if f.name == "Liberation Sans"]
    assert twins, "neither Arial nor Liberation Sans is installed"
    fm.fontManager.ttflist.extend(dataclasses.replace(f, name=FONT) for f in twins)
    fm.FontManager._findfont_cached.cache_clear()
    _FONTS.clear()


@contextlib.contextmanager
def style():
    """Every figure is drawn inside this context, so nothing else changes matplotlib's global settings."""
    _arial_or_metric_twin()
    with matplotlib.rc_context(RC):
        yield


def rng(key):
    """Random generator seeded from a panel key: jitter is the same at every run and independent of the panel order."""
    return np.random.default_rng(zlib.crc32(key.encode()))


# ---- measuring text ------------------------------------------------------------------------------------------------
# Arial metrics as fractions of the font size (hhea table): used to align text runs on a common baseline.
ASCENT, DESCENT, CAP = 0.905, 0.212, 0.716


_FONTS = {}


def _ft(style_, weight):
    """FreeType face of Arial in the given style (cached)."""
    from matplotlib import font_manager, ft2font
    key = (style_, weight)
    if key not in _FONTS:
        path = font_manager.findfont(font_manager.FontProperties(family=FONT, style=style_, weight=weight))
        _FONTS[key] = ft2font.FT2Font(path)
    return _FONTS[key]


def text_advance_pt(fig, s, size, style_="normal", weight="normal"):
    """Advance width of a string in points: glyph advances plus kerning, as FreeType lays out the text (trailing
    spaces included). `fig` is unused (kept for the call signature)."""
    from matplotlib import ft2font
    if s == "":
        return 0.0
    f = _ft(style_, weight)
    f.set_size(size, 72)                                     # at 72 dpi one pixel is one point
    no_hint = getattr(ft2font, "LOAD_NO_HINTING", None)
    if no_hint is None:
        no_hint = ft2font.LoadFlags.NO_HINTING
    kern_mode = getattr(ft2font, "KERNING_DEFAULT", None)
    if kern_mode is None:
        kern_mode = ft2font.Kerning.DEFAULT
    total, prev = 0.0, None
    for ch in s:
        gi = f.get_char_index(ord(ch))
        if prev is not None:
            total += f.get_kerning(prev, gi, kern_mode) / 64.0
        g = f.load_char(ord(ch), flags=no_hint)
        total += g.linearHoriAdvance / 65536.0
        prev = gi
    return total


def rich_text(host, x, y, runs, size=BASE, ha="left", va="baseline", transform=None, gid=None, zorder=5, **kw):
    """Write `runs` = [(text, fmt), ...] as adjacent text artists; fmt is a string of flags: "i" italic, "b" bold, "^"
    superscript (smaller and raised, never below 5 pt), "_" subscript. `host` is an Axes, a SubFigure or the Figure;
    (x, y) are in `transform` coordinates (default: the host's axes or figure fraction). Returns the list of Text
    artists; their gids are `gid`__k (one label for the checks)."""
    fig = host.figure if hasattr(host, "figure") and host.figure is not None else host
    while hasattr(fig, "figure") and fig.figure is not None and fig.figure is not fig:
        fig = fig.figure
    if transform is None:
        transform = host.transAxes if hasattr(host, "transAxes") else host.transSubfigure \
            if hasattr(host, "transSubfigure") else host.transFigure
    parts = []
    for s, fmt in runs:
        sub = "^" in fmt or "_" in fmt
        sz = max(SMALL, round(0.8 * size, 2)) if sub else size
        st = "italic" if "i" in fmt else "normal"
        wt = "bold" if "b" in fmt else "normal"
        dy = 0.38 * size if "^" in fmt else (-0.12 * size if "_" in fmt else 0.0)
        parts.append((s, sz, st, wt, dy, text_advance_pt(fig, s, sz, st, wt)))
    total = sum(p[-1] for p in parts)
    x0 = {"left": 0.0, "center": -total / 2.0, "right": -total}[ha]
    y0 = {"baseline": 0.0, "bottom": DESCENT * size, "top": -ASCENT * size,
          "center": -(CAP * size) / 2.0}[va]
    gid = gid or "rt%08x" % zlib.crc32(("%r %r %r" % (runs, x, y)).encode())
    out = []
    dx = x0
    for k, (s, sz, st, wt, dy, w) in enumerate(parts):
        tr = offset_copy(transform, fig=fig, x=dx, y=y0 + dy, units="points")
        t = host.text(x, y, s, transform=tr, fontsize=sz, fontstyle=st, fontweight=wt, ha="left", va="baseline",
                      fontfamily=FONT, zorder=zorder, **kw)
        t.set_gid("%s__%d" % (gid, k))
        out.append(t)
        dx += w
    return out


def rich_width_pt(fig, runs, size=BASE):
    """Total advance of rich-text runs in points (for layout decisions before drawing)."""
    total = 0.0
    for s, fmt in runs:
        sub = "^" in fmt or "_" in fmt
        sz = max(SMALL, round(0.8 * size, 2)) if sub else size
        total += text_advance_pt(fig, s, sz, "italic" if "i" in fmt else "normal", "bold" if "b" in fmt else "normal")
    return total


# ---- numbers -------------------------------------------------------------------------------------------------------
def fmt_num(v, sig=2):
    """Two significant digits, keeping trailing zeros (0.50, 0.031, 0.0055, 1.2, 12)."""
    if v == 0 or not np.isfinite(v):
        return "0" if v == 0 else str(v)
    dec = max(0, sig - 1 - int(math.floor(math.log10(abs(v)))))
    return ("%." + str(dec) + "f") % v


def p_runs(p, sym="P", sig=2):
    """Exact P (or q) value as rich-text runs: italic symbol, decimals from 0.001 up, otherwise m x 10^-k."""
    if p >= 0.001:
        return [(sym, "i"), (" = " + fmt_num(p, sig), "")]
    if p <= 0:
        return [(sym, "i"), (" < 10", ""), (MINUS + "300", "^")]
    m, e = ("%.1e" % p).split("e")
    return [(sym, "i"), (" = %s %s 10" % (m, TIMES), ""), (MINUS + str(abs(int(e))), "^")]


def p_runs_log10(l10, sym="P"):
    """Exact P from its base-10 logarithm, for P values below the floating-point range (two significant digits)."""
    if l10 >= -3:
        return p_runs(10 ** l10, sym)
    e = int(math.floor(l10))
    m = 10 ** (l10 - e)
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return [(sym, "i"), (" = %.1f %s 10" % (m, TIMES), ""), (MINUS + str(abs(e)), "^")]


def n_runs(n, eq=" = "):
    return [("n", "i"), ("%s%s" % (eq, format(int(n), ",")), "")]


def rho_runs(rho, sym="ρ", sig=2):
    s = fmt_num(abs(rho), sig)
    return [(sym, "i"), (" = " + (MINUS if rho < 0 else "") + s, "")]


def join_runs(*groups, sep=", "):
    """Concatenate rich-text run lists with a separator, e.g. join_runs(rho_runs(r), p_runs(p))."""
    out = []
    for k, g in enumerate(groups):
        if k:
            out.append((sep, ""))
        out.extend(g)
    return out
