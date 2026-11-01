# Supplementary figures

| Figure | Script | Content |
|--------|--------|---------|
| S1 | `01_SuppFig1.py` | Cohort structure and malignant PCs across the cohort |
| S2 | `02_SuppFig2.py` | Malignant PCs without detectable CNV |
| S3 | `03_SuppFig3.py` | Batch structure of the PC UMAP; paired BM and PB samples |
| S4 | `04_SuppFig4.py` | Cytogenetic classification at the single-cell level |
| S5 | `05_SuppFig5.py` | Sixteen therapeutic targets on the UMAP of Fig. 1C |
| S6 | `06_SuppFig6.py` | Genes stably up- or downregulated in malignant PCs across disease stages |
| S7 | `07_SuppFig7.py` | GO terms, published signatures, bone marrow cell types and correlations of the signatures |
| S8 | `08_SuppFig8.py` | Summary of the sixteen signatures (two pages) |
| S9 | `09_SuppFig9.py` | The circulatory capacity signature |
| S10 | `10_SuppFig10.py` | CoMMpass progression-free survival by four signatures |
| S11 | `11_SuppFig11.py` | Additional features of the malignancy and MST classifiers |
| S12 | `12_SuppFig12.py` | Feature selection and parameter tuning of the classifiers |
| S13 | `13_SuppFig13.py` | MGUS malignant PCs called MM; malignancy scores in external datasets |
| S14 | `14_SuppFig14.py` | Progression in SMM, serially sampled patients, cytogenetic groups and external datasets |
| S15 | `15_SuppFig15.py` | Survival models of the MST score in CoMMpass; mutation- versus transcription-based scores |
| S16 | `16_SuppFig16.py` | Stage associations with and without adjustment for genetic alterations; histone/chaperone activity and tumor ploidy |

Outputs as for the main figures: `figures/<figure>.svg` and `.pdf` (Supplementary Fig. 8: `SuppFig8_p1.svg`,
`SuppFig8_p2.svg` and a two-page PDF), `figures/source_data/` and `figures/checks/`.

Supplementary Fig. 9 uses CoMMpass patient-level tables (MMRF, controlled access) and runs in the analysis environment
(`requirements.txt`); its PDF is made from the SVG with Inkscape on a machine with Arial
(`inkscape figures/SuppFig9.svg --export-type=pdf`).

Inputs are the tables in `results/` (analysis steps and `11_figure_inputs`) and the processed data under `data/`
(controlled access, dbGaP phs003855; see the root README).

## Run

```bash
for f in 13_supplementary_figures/*.py; do PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=0 python3 "$f"; done
```
