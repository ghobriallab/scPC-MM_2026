# External datasets

Public single-cell datasets: processing, classifier features and signature activity in bone marrow cell types.

| Script | Computes |
|--------|----------|
| `01_boiarsky_natcommun2022.py` | AnnData objects of the PCs of Boiarsky et al., Nat Commun 2022 (GSE193531): counts, log-normalized counts and embedding, for the external validation of the malignancy and MST scores (Fig. 5C; Supplementary Figs. 13C, 14E). |
| `02_ledergor_natmedicine2018.py` | AnnData objects of the PCs of Ledergor et al., Nat Med 2018 (GSE117156): counts, log-normalized counts and embedding, for the external validation of the malignancy and MST scores (Fig. 5C; Supplementary Figs. 13C, 14E). |
| `03_dang_cancercell2023.py` | AnnData objects of the PCs of Dang et al., Cancer Cell 2023 (EGAS00001006694): counts, log-normalized counts and embedding, for the external validation of the malignancy and MST scores (Fig. 5C; Supplementary Figs. 13C, 14E). |
| `04_zavidij_bm_immune_cells.py` | AnnData objects of the bone marrow immune cells of Zavidij et al., Nat Cancer 2020 (GSE124310), for the signature activity in bone marrow immune cells across disease stages (Methods, signature activity in other bone marrow cell types). |
| `05_combined_ledergor_boiarsky_dang.py` | Combined AnnData objects of the PCs of the three external validation datasets (Ledergor et al. 2018, Boiarsky et al. 2022, Dang et al. 2023), for the stage-free signature test and the compact MGUS-MM classifier (Results and Methods, MGUS-MM classifier). |
| `06_signature_activity_bm_cell_types.py` | Activity of the 16 signatures across bone marrow cell types, and its association with disease stage in bone marrow immune cells, in two public scRNA-seq datasets (Supplementary Fig. 7C; Results; Methods, signature activity in other bone marrow cell types). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 07_external_datasets/01_boiarsky_natcommun2022.py
python3 07_external_datasets/02_ledergor_natmedicine2018.py
python3 07_external_datasets/03_dang_cancercell2023.py
python3 07_external_datasets/04_zavidij_bm_immune_cells.py
python3 07_external_datasets/05_combined_ledergor_boiarsky_dang.py
python3 07_external_datasets/06_signature_activity_bm_cell_types.py
```
