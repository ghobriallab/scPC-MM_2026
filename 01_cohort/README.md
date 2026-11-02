# Cohort

Cohort structure and the sample, cell, BCR and isotype counts of the cohort.

| Script | Computes |
|--------|----------|
| `01_cohort_structure.py` | Individuals, samples and PCs by disease stage and tissue, sampling designs and the individuals or tumors kept by each analysis (Extended Data Fig. 1A; Supplementary Table 2; Methods, Cohort structure and repeat samples). |
| `02_cell_bcr_isotype_counts.py` | Descriptive counts of the cohort reported in the Results and Methods: PCs per individual and stage, BCR coverage, clonal isotypes, immunofixation concordance, healthy-donor samples, cytogenetic assays and the cells used for signature detection. |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 01_cohort/01_cohort_structure.py
python3 01_cohort/02_cell_bcr_isotype_counts.py
```
