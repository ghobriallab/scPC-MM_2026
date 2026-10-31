"""Shared paths for every Python script.

Purpose:      One place for the input, result and figure directories, so that no script holds a machine-specific
              path. Scripts import the helpers below after adding the repository root to sys.path.

Inputs:       Optional environment variables SCPCMM_DATA, SCPCMM_RESULTS and SCPCMM_FIGURES; the defaults are data/,
              results/ and figures/ next to this file. data/ follows the layout described in the root README.

Outputs:      DATA_DIR, RESULTS_DIR, FIGURES_DIR, SEED and the helpers data(), result() and result_dir().

Dependencies: Python 3.8+ (standard library only).
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SCPCMM_DATA", REPO_DIR / "data"))
RESULTS_DIR = Path(os.environ.get("SCPCMM_RESULTS", REPO_DIR / "results"))
FIGURES_DIR = Path(os.environ.get("SCPCMM_FIGURES", REPO_DIR / "figures"))

SEED = 42  # random_state of every model, split and resampling


def data(*parts):
    """An input file under DATA_DIR, e.g. data("paper", "cohort-clinical-metadata.csv")."""
    return DATA_DIR.joinpath(*parts)


def result_dir(step):
    """The output folder of one analysis step, e.g. result_dir("stage_and_progression"); created if needed."""
    d = RESULTS_DIR / step
    d.mkdir(parents=True, exist_ok=True)
    return d


def result(step, name):
    """A table written by an earlier step; fails early if that step has not been run."""
    p = RESULTS_DIR / step / name
    if not p.exists():
        raise FileNotFoundError("run the step that writes %s first" % p.relative_to(RESULTS_DIR))
    return p
