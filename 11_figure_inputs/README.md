# Figure inputs

De-identified, plot-ready tables and arrays for the figures, extracted from the processed objects.

| Script | Computes |
|--------|----------|
| `01_extract_tables.py` | Plot-ready, de-identified tables for the figures, from the processed objects (small inputs). |
| `02_extract_cells.py` | Cell-level, de-identified arrays for the UMAP panels, from the processed objects (large inputs). |
| `03_wgs_residuals_by_stage.py` | Association of the residuals of the two WGS regressions with disease stage (Spearman; Supplementary Fig. 15C,D). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 11_figure_inputs/01_extract_tables.py
python3 11_figure_inputs/02_extract_cells.py
python3 11_figure_inputs/03_wgs_residuals_by_stage.py
```
