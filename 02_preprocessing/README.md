# Preprocessing

Quality control, doublet removal, normalization, clustering and embedding of the plasma cells.

| Script | Computes |
|--------|----------|
| `01_cellranger_metrics.py` | One row of the Cell Ranger metrics of one library, for the library QC table. |
| `02_doublet_scores.py` | Doublet scores and calls of every cell of one library with scds, scDblFinder and Scrublet (cell QC). |
| `03_preprocess_umap.py` | Normalization, highly variable genes, PCA, Harmony integration, neighbor graphs and UMAPs of one aggregated plasma cell object (input of the Leiden clustering). |
| `04_leiden_clusters.py` | Leiden clusters of one preprocessed plasma cell object at one resolution, on the PCA graph and on the two Harmony graphs (cell-type annotation). |
| `05_cluster_markers.py` | Marker genes of the Leiden clusters of one plasma cell object (cell-type annotation). |
| `06_format_anndata.py` | Aggregated plasma cell objects: counts of the cohort, annotated (malignant and normal) PCs, log-normalized expression, UMAP of all PCs and UMAP coordinate tables (Methods, scRNA-seq data normalization and clustering). |
| `07_per_patient_objects.py` | Per-patient count objects of all PCs and of the annotated (malignant and normal) PCs. |
| `08_batch_neighbor_composition.py` | Origin of the 30 nearest neighbors of each PC in the graph of the Fig. 1C UMAP, same-day and tissue mixing tests, and per-patient UMAPs of same-visit BM and PB samples (Results; Methods, scRNA-seq data normalization and clustering; Extended Data Fig. 3). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 02_preprocessing/01_cellranger_metrics.py
python3 02_preprocessing/02_doublet_scores.py
python3 02_preprocessing/03_preprocess_umap.py
python3 02_preprocessing/04_leiden_clusters.py
python3 02_preprocessing/05_cluster_markers.py
python3 02_preprocessing/06_format_anndata.py
python3 02_preprocessing/07_per_patient_objects.py
python3 02_preprocessing/08_batch_neighbor_composition.py
```
