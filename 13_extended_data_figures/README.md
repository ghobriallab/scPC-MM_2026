# Extended Data figures

| Figure | Script | Content |
|--------|--------|---------|
| ED1 | `01_EDFig1.py` | Cohort structure and malignant PCs across the cohort |
| ED2 | `02_EDFig2.py` | Malignant PCs without detectable CNV |
| ED3 | `03_EDFig3.py` | Batch structure of the PC UMAP; paired BM and PB samples |
| ED4 | `04_EDFig4.py` | Cytogenetic classification at the single-cell level |
| ED5 | `05_EDFig5.py` | GO terms, published signatures, bone marrow cell types and correlations of the signatures |
| ED6 | `06_EDFig6.py` | Feature selection and parameter tuning of the classifiers |
| ED7 | `07_EDFig7.py` | MGUS malignant PCs called MM; malignancy scores in published datasets |
| ED8 | `08_EDFig8.py` | Progression in SMM, serially sampled patients, cytogenetic groups and published datasets |
| ED9 | `09_EDFig9.py` | Survival models of the MST score in CoMMpass; mutation- versus transcription-based scores |
| ED10 | `10_EDFig10.py` | Stage associations with and without adjustment for genetic alterations; histone/chaperone activity and tumor ploidy |

Outputs as for the main figures: `figures/<figure>.svg` and `.pdf`, `figures/source_data/` and `figures/checks/`.

Inputs are the tables in `results/` (analysis steps and `11_figure_inputs`) and the processed data under `data/`
(controlled access, dbGaP phs003855; see the root README).

## Run

```bash
for f in 13_extended_data_figures/*.py; do PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=0 python3 "$f"; done
```
