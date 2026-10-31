# Malignant cells

B cell receptor sequences, clonotypes and isotypes of the plasma cells.

| Script | Computes |
|--------|----------|
| `01_dandelion_preprocess.sh` | Runs the dandelion preprocessing pipeline (dandelion-preprocess) on the scBCR-seq contigs of every individual, one individual at a time (Methods, scBCR-seq data preprocessing and BCR filtering). |
| `02_dandelion_clonotypes.py` | BCR clonotypes of the PCs of one individual with dandelion, added to the gene expression object of that individual (Methods, scBCR-seq data preprocessing and BCR filtering). |
| `03_cohort_bcr_tables.py` | Cohort BCR tables: combines the dandelion objects of all individuals and filters cells and gene calls (Methods, scBCR-seq data preprocessing and BCR filtering). |
| `04_isotype_immunofixation.py` | Isotype and light chain of the malignant BM PCs of each patient from scBCR-seq, and their comparison with the clinical serum immunofixation (isotype table and immunofixation concordance table). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
bash 03_malignant_cells/01_dandelion_preprocess.sh
python3 03_malignant_cells/02_dandelion_clonotypes.py
python3 03_malignant_cells/03_cohort_bcr_tables.py
python3 03_malignant_cells/04_isotype_immunofixation.py
```
