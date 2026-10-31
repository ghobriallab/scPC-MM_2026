# Main figures

| Figure | Script | Content |
|--------|--------|---------|
| 1 | `01_Fig1.py` | Study design, cohort overview and the plasma cell atlas |
| 2 | `02_Fig2.py` | Sixteen therapeutic targets by cytogenetic group |
| 3 | `03_Fig3.py` | Sixteen transcriptional signatures, their stage associations and CoMMpass survival |
| 4 | `04_Fig4.py` | The malignancy and MST random forest classifiers |
| 5 | `05_Fig5.py` | Clinical relevance, external validation and genomic correlates of the MST score |
| 6 | `06_Fig6.py` | MGUS-like and MM-like transcriptional states within tumors across disease stages and CNV clones |

Each script draws one figure at its print size and writes `figures/<figure>.svg` (editable text) and `.pdf`, the
plotted values of each panel to `figures/source_data/`, and the result of automatic layout checks (fonts, sizes,
overlaps) to `figures/checks/`; it ends with `ALL CHECKS PASS`. Every printed statistic is read from a table of the
analysis steps, or recomputed and checked against it.

Inputs are the tables in `results/` (analysis steps and `11_figure_inputs`) and the processed data under `data/`
(controlled access, dbGaP phs003855; see the root README).

## Run

```bash
for f in 12_main_figures/*.py; do PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=0 python3 "$f"; done
```

Environment: `requirements_figures.txt` (Python 3.13); font Arial, or Liberation Sans (same metrics).
