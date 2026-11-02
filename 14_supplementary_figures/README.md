# Supplementary figures

| Figure | Script | Content |
|--------|--------|---------|
| S1 | `01_SuppFig1.py` | Sixteen therapeutic targets on the UMAP of Fig. 1C |
| S2 | `02_SuppFig2.py` | Genes stably up- or downregulated in malignant PCs across disease stages |
| S3, S4 | `03_SuppFig3_4.py` | Summary of the sixteen signatures (eight per figure) |
| S5 | `04_SuppFig5.py` | The circulatory capacity signature |
| S6 | `05_SuppFig6.py` | CoMMpass progression-free survival by four signatures |
| S7 | `06_SuppFig7.py` | Additional features of the malignancy and MST classifiers |

Outputs as for the main figures: `figures/<figure>.svg` and `.pdf`, `figures/source_data/` and `figures/checks/`.

Supplementary Fig. 5 uses CoMMpass patient-level tables (MMRF, controlled access) and runs in the analysis environment
(`requirements.txt`); its PDF is made from the SVG with Inkscape on a machine with Arial
(`inkscape figures/SuppFig5.svg --export-type=pdf`).

Inputs are the tables in `results/` (analysis steps and `11_figure_inputs`) and the processed data under `data/`
(controlled access, dbGaP phs003855; see the root README).

## Run

```bash
for f in 14_supplementary_figures/*.py; do PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=0 python3 "$f"; done
```
