"""Where the figure scripts read and write.

Purpose:      Figure-specific locations derived from config.py: figure files in figures/, the plotted values of each
              panel in figures/source_data, checks and previews in figures/checks and figures/previews; figure inputs
              written by 11_figure_inputs in results/figure_inputs; tables of the analysis steps in results/<step>.

Inputs:       config.py (repository root).

Outputs:      FIGURES, SOURCE_DATA, CHECKS, PREVIEWS and the helpers cache(), result() and inputs().

Dependencies: Python 3.8+ (standard library only).
"""
import os

import config

FIGURES = str(config.FIGURES_DIR)
SOURCE_DATA = os.path.join(FIGURES, "source_data")      # plotted values per panel
CHECKS = os.path.join(FIGURES, "checks")
PREVIEWS = os.path.join(FIGURES, "previews")
for _d in (FIGURES, SOURCE_DATA, CHECKS, PREVIEWS):
    os.makedirs(_d, exist_ok=True)


def cache(name):
    """A figure input written by 11_figure_inputs, e.g. cache("pcs_cells.npz")."""
    return str(config.result("figure_inputs", name))


def result(step, name):
    """A table written by an analysis step, e.g. result("stage_and_progression", "smm_progression_results.tsv")."""
    return str(config.result(step, name))


def inputs(*parts):
    """An input file under data/, e.g. inputs("paper", "cohort-clinical-metadata.csv")."""
    p = config.data(*parts)
    assert p.exists(), "missing input: data/" + "/".join(parts)
    return str(p)
