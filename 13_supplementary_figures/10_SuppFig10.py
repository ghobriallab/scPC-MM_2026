#!/usr/bin/env python3
"""Supplementary Figure 10: progression-free survival of CoMMpass patients by thirds of four signatures.

Purpose:      The survival panel of Fig. 3D,E (cut-offs at the 33rd and 66th percentiles, log-rank P) for B cell
              activation, cell cycling G1/S, histone/chaperone and cell cycling G2/M.

Inputs:       data/commpass/MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv; panel function of 12_main_figures/03_Fig3.py.

Outputs:      figures/SuppFig10.svg, SuppFig10.pdf (no source data: CoMMpass patient-level data).

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import importlib
from checks import figure_checks, file_checks, report
from figstyle import style
from layout import FigureLayout, Slot, new_figure, save

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "12_main_figures"))
fig3 = importlib.import_module("03_Fig3")        # survival panel of Fig. 3D,E

SIGS = ["B_cell_activation", "cell_cycling_G1S", "histone_chaperone_gene", "cell_cycling_G2M"]
LAYOUT = FigureLayout("SuppFig10", [Slot("km_" + s, "", k * 46.0, 0, 45, 47) for k, s in enumerate(SIGS)], max_height=247)


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        for s in SIGS:
            fig3.panel_km(pn["km_" + s], s)
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("SuppFig10", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
