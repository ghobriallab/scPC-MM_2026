"""Every color used in the figures, chosen to stay distinguishable with color-vision deficiencies.

Purpose:      Ordinal groups (disease stages, risk groups) use ramps with monotonic lightness; categorical groups use
              the Okabe-Ito and Paul Tol color-blind-safe sets; no red/green pairs. Text is never colored.

Inputs:       None.

Outputs:      Palette constants imported by the figure scripts. `python3 palettes.py --check` simulates protanopia,
              deuteranopia and tritanopia (Machado et al., 2009, severity 1), reports the smallest CIEDE2000
              difference within each categorical palette and the lightness order of each ordinal palette, and draws a
              palette sheet (figures/checks/palettes.pdf).

Dependencies: numpy; matplotlib for the check.
"""
import os
import sys

import numpy as np

# ---- disease stages ----------------------------------------------------------------------------------------------
STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
DONOR = "#8C96A6"                       # neutral blue-gray
# malignant PCs and stage-colored embeddings: light -> dark warm ramp (seaborn "rocket" family)
TUMOR = {"MGUS": "#F6C28B", "LRSMM": "#F08350", "IRSMM": "#DA3B46", "HRSMM": "#A11A5B", "MM": "#5C1A4F"}
# normal PCs: light -> dark blue ramp
NORMAL = {"Donor": DONOR, "MGUS": "#C3DCEE", "LRSMM": "#93C2E2", "IRSMM": "#5BA0D0", "HRSMM": "#2F74B5",
          "MM": "#0E4A8A"}
STAGE = dict(TUMOR, Donor=DONOR)        # one color per stage (embeddings, per-patient plots of malignant PCs)

# ---- plasma-cell groups (embeddings and keys) --------------------------------------------------------------------
PC_STATUS = {"BM normal": "#56B4E9", "PB normal": "#B5DDF5", "BM malignant": "#C2185B", "PB malignant": "#F48FB1"}
TISSUE = {"BM": "#4D4D4D", "PB": "#D55E00"}

# ---- cytogenetic groups ------------------------------------------------------------------------------------------
CYTO_ORDER = ["CCND1", "CCND2", "CCND3", "NSD2", "MAF", "MAFA", "MAFB", "HRD", "Other"]
CYTO = {"CCND1": "#E69F00", "CCND2": "#EEDD88", "CCND3": "#FFAABB", "NSD2": "#228833", "MAF": "#77AADD",
        "MAFA": "#99DDFF", "MAFB": "#882255", "HRD": "#332288", "Other": "#8C8C8C"}   # greedy max-min CIEDE2000 under
# normal vision and the three simulations, with the darker "+HRD" twins included (minimum difference 6.2)


def darker(hex_color, f=0.6):
    """Same hue, darker (used for "<translocation>+HRD" next to "<translocation>")."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float) * f
    return "#%02X%02X%02X" % tuple(np.clip(np.round(c), 0, 255).astype(int))


CYTO_HRD = {"%s+HRD" % k: darker(v) for k, v in CYTO.items() if k not in ("HRD", "Other")}

# ---- the 16 transcriptional signatures (heatmap order of Fig. 3A, Supplementary Figs. 3 and 4 order) ---------------------
# 16 of 46 color-blind-safe colors (Tol, Okabe-Ito) by greedy max-min CIEDE2000 over normal vision and the three
# simulations (minimum 8.2), ordered so that neighbors in the Fig. 3A color strip differ by at least 29.7
SIGNATURE = {"cell_cycling_G2M": "#009E73", "cell_cycling_G1S": "#332288", "B_cell_activation": "#99DDFF",
             "MHC-II_immune_checkpoint": "#994455", "FOS-JUN_stress_response": "#56B4E9",
             "MCL1-BCL2_signature": "#EE6677", "circulatory_capacity": "#4B4B8F", "interferon_inducible": "#999933",
             "HRD_signature": "#5C1A4F", "myeloma_oncogene": "#EEDD88", "secretory_granule": "#CC3311",
             "MAF*_signature": "#0077BB", "MHC-I_gene_module": "#BBCC33", "vacuole_ER_signaling": "#CC79A7",
             "histone_chaperone_gene": "#228833", "extracellular_matrix": "#FFAABB"}

# ---- the longitudinally sampled patients (Supplementary Fig. 5A and Extended Data Fig. 8C) ---------------------------------------------------------
PATIENT3 = {"Pt-28": "#0072B2", "Pt-122": "#D55E00", "Pt-160": "#009E73"}

# ---- isotypes ----------------------------------------------------------------------------------------------------
ISOTYPE = {"IgG": "#0072B2", "IgA": "#E69F00", "IgD": "#009E73", "IgE": "#CC79A7", "IgM": "#D55E00",
           "IgM/IgD": "#56B4E9", "Light chain only": "#666666"}

# ---- three ordered groups (KM curves by thirds or by MST group) --------------------------------------------------
LOW_MED_HIGH = {"Low": "#0072B2", "Medium": "#E69F00", "High": "#A11A5B"}

# ---- statistics marks --------------------------------------------------------------------------------------------
BRACKET_MAL = "#C0392B"      # malignant vs normal PCs (legends: "red")
BRACKET_STAGE = "#1F5FA8"    # between disease stages (legends: "blue")
REFLINE = "#808080"          # score of 0.5 (gray dashed line)
PROGRESSOR = "#1A1A1A"       # patients who later progressed (diamonds)
BOX_EDGE = "#333333"

# ---- continuous maps (names of matplotlib colormaps) -------------------------------------------------------------
CMAP_MST = "Purples"
CMAP_MALIGNANCY = "Blues"
CMAP_DIVERGING = "RdBu_r"
CMAP_EXPR = "viridis"

CATEGORICAL = {"stage (embeddings)": STAGE, "PC status": PC_STATUS, "cytogenetic group": CYTO,
               "cytogenetic group with +HRD": dict(CYTO, **CYTO_HRD), "isotype": ISOTYPE,
               "low/medium/high": LOW_MED_HIGH, "tissue": TISSUE, "signature": SIGNATURE, "patients (Extended Data Fig. 8C)": PATIENT3,
               "brackets": {"malignant vs normal": BRACKET_MAL, "between stages": BRACKET_STAGE}}
ORDINAL = {"malignant PCs by stage": [TUMOR[s] for s in STAGES[1:]],
           "normal PCs by stage": [NORMAL[s] for s in STAGES[1:]]}

# ---- color-vision simulation (Machado, Oliveira & Fernandes 2009, severity 1.0) ---------------------------------
MACHADO = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb(hex_colors):
    return np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in hex_colors])


def simulate(hex_colors, kind):
    lin = _srgb_to_linear(rgb(hex_colors))
    return _linear_to_srgb(lin @ MACHADO[kind].T)


def lab(srgb):
    from skimage.color import rgb2lab
    return rgb2lab(srgb.reshape(1, -1, 3)).reshape(-1, 3)


def min_delta(hex_colors, kind=None):
    from skimage.color import deltaE_ciede2000
    c = rgb(hex_colors) if kind is None else simulate(hex_colors, kind)
    L = lab(c)
    best = (np.inf, None)
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            d = float(deltaE_ciede2000(L[i], L[j]))
            if d < best[0]:
                best = (d, (i, j))
    return best


def check(out_pdf=None):
    rows = []
    for name, pal in CATEGORICAL.items():
        keys, cols = list(pal), list(pal.values())
        line = [name]
        for kind in (None, "protanopia", "deuteranopia", "tritanopia"):
            d, (i, j) = min_delta(cols, kind)
            line.append("%s %.1f (%s/%s)" % (kind or "normal", d, keys[i], keys[j]))
        rows.append(" | ".join(line))
    for name, cols in ORDINAL.items():
        line = [name]
        for kind in (None, "protanopia", "deuteranopia", "tritanopia"):
            c = rgb(cols) if kind is None else simulate(cols, kind)
            L = lab(c)[:, 0]
            line.append("%s L* %s %s" % (kind or "normal", " ".join("%.0f" % v for v in L),
                                        "monotonic" if np.all(np.diff(L) < 0) else "NOT MONOTONIC"))
        rows.append(" | ".join(line))
    if out_pdf:
        _sheet(out_pdf)
    return rows


def _sheet(path):
    """One row per palette, under normal vision and the three simulations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import style, MM
    pals = dict(CATEGORICAL)
    pals.update({k: dict(zip(STAGES[1:], v)) for k, v in ORDINAL.items()})
    with style():
        fig, axs = plt.subplots(len(pals), 4, figsize=(183 * MM, 9 * len(pals) * MM))
        for r, (name, pal) in enumerate(pals.items()):
            cols = list(pal.values())
            for c, kind in enumerate((None, "protanopia", "deuteranopia", "tritanopia")):
                ax = axs[r, c]
                sim = rgb(cols) if kind is None else simulate(cols, kind)
                for k, col in enumerate(sim):
                    ax.add_patch(plt.Rectangle((k, 0), 0.9, 1, color=col))
                ax.set_xlim(0, max(len(cols), 9))
                ax.set_ylim(0, 1)
                ax.axis("off")
                if r == 0:
                    ax.set_title(kind or "normal vision", fontsize=6)
                if c == 0:
                    ax.text(-0.2, 0.5, name, ha="right", va="center", fontsize=5, transform=ax.transData)
        fig.subplots_adjust(left=0.2, right=0.99, top=0.95, bottom=0.02, hspace=0.6, wspace=0.05)
        fig.savefig(path)
        plt.close(fig)


if __name__ == "__main__":
    if "--check" in sys.argv:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path[:0] = [here, os.path.dirname(here)]
        import paths
        out = os.path.join(paths.CHECKS, "palettes.pdf")
        for line in check(out):
            print(line)
