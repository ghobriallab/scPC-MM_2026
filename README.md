# scPC-MM: single-cell transcription of plasma cells from myeloma precursors to myeloma

Code to reproduce the analyses and every figure panel in **"A Myeloma Single-Cell Transcription (MST) Score Tracks
Disease Stage in Multiple Myeloma and Its Precursors"**.

The study profiles 1,649,290 plasma cells (PCs) from the bone marrow (BM) and peripheral blood (PB) of 226 individuals
(41 with multiple myeloma (MM), 120 with smoldering MM (SMM), 36 with monoclonal gammopathy of undetermined significance
(MGUS) and 29 healthy donors) by single-cell RNA and B cell receptor sequencing. It defines 16 transcriptional signatures
and two random forest classifiers: a malignancy score, and the MST score, which places malignant PCs between MGUS-like and
MM-like. The scores are validated in three public single-cell datasets and in the CoMMpass bulk RNA-seq cohort.

## Conventions

Folders and scripts carry numeric prefixes in the order they run; each script states its purpose, inputs, outputs and
dependencies in its header. Statistical tests are two-sided unless stated, multiple testing is corrected with the
Benjamini-Hochberg method (q values), and every random step is seeded. Processing steps write the processed data
under `data/`, where the next steps read them; analysis steps write to `results/<step>/` and figure scripts to
`figures/`, both created on first use and not tracked. SMM risk groups follow the IMWG 2/20/20 model (bone marrow
plasma cells > 20%, serum M-protein > 2 g/dL, involved/uninvolved free light chain ratio > 20) and are assigned in `risk.py`.

## Analysis steps

| Folder | Content |
|--------|---------|
| `01_cohort` | Cohort structure, sample, cell, BCR and isotype counts |
| `02_preprocessing` | Quality control, doublets, normalization, clustering and embedding |
| `03_malignant_cells` | BCR sequences, clonotypes and isotypes |
| `04_genomics` | Copy number variants, clones and cytogenetic classification |
| `05_differential_expression` | Differential expression and therapeutic targets |
| `06_signatures` | The 16 transcriptional signatures |
| `07_external_datasets` | Public single-cell datasets |
| `08_classifiers` | Malignancy and MST classifiers and their scores |
| `09_stage_and_progression` | Stage associations, progression in SMM, MM-like cells |
| `10_commpass` | CoMMpass expression, scores and survival models |
| `11_figure_inputs` | De-identified, plot-ready inputs of the figures |
| `12_main_figures` | Figures 1-6 |
| `13_extended_data_figures` | Extended Data Figures 1-10 |
| `14_supplementary_figures` | Supplementary Figures 1-7 |

Each folder has its own README with its scripts.

## Figures

| Figure | Script |
|--------|--------|
| Figure 1 | `12_main_figures/01_Fig1.py` |
| Figure 2 | `12_main_figures/02_Fig2.py` |
| Figure 3 | `12_main_figures/03_Fig3.py` |
| Figure 4 | `12_main_figures/04_Fig4.py` |
| Figure 5 | `12_main_figures/05_Fig5.py` |
| Figure 6 | `12_main_figures/06_Fig6.py` |
| Extended Data Figure 1 | `13_extended_data_figures/01_EDFig1.py` |
| Extended Data Figure 2 | `13_extended_data_figures/02_EDFig2.py` |
| Extended Data Figure 3 | `13_extended_data_figures/03_EDFig3.py` |
| Extended Data Figure 4 | `13_extended_data_figures/04_EDFig4.py` |
| Extended Data Figure 5 | `13_extended_data_figures/05_EDFig5.py` |
| Extended Data Figure 6 | `13_extended_data_figures/06_EDFig6.py` |
| Extended Data Figure 7 | `13_extended_data_figures/07_EDFig7.py` |
| Extended Data Figure 8 | `13_extended_data_figures/08_EDFig8.py` |
| Extended Data Figure 9 | `13_extended_data_figures/09_EDFig9.py` |
| Extended Data Figure 10 | `13_extended_data_figures/10_EDFig10.py` |
| Supplementary Figure 1 | `14_supplementary_figures/01_SuppFig1.py` |
| Supplementary Figure 2 | `14_supplementary_figures/02_SuppFig2.py` |
| Supplementary Figures 3 and 4 | `14_supplementary_figures/03_SuppFig3_4.py` |
| Supplementary Figure 5 | `14_supplementary_figures/04_SuppFig5.py` |
| Supplementary Figure 6 | `14_supplementary_figures/05_SuppFig6.py` |
| Supplementary Figure 7 | `14_supplementary_figures/06_SuppFig7.py` |

## Data

### Single-cell data

Single-cell RNA and BCR sequencing data, raw (FASTQ) and processed, are shared under controlled access through dbGaP,
in accordance with the informed consent of the participants: accession
[phs003855.v1.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs003855.v1.p1) (samples also
included in a study of circulating tumor cells) and phsXXXXXX (other samples). Access is requested through the dbGaP
Authorized Access website.

### Other datasets

| Source | Reference | Used in |
|--------|-----------|---------|
| [GSE117156](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117156) | Ledergor et al., *Nat Med* 2018 | Fig. 5C; Extended Data Figs. 7C and 8E |
| [GSE193531](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193531) | Boiarsky et al., *Nat Commun* 2022 | Fig. 5C; Extended Data Figs. 7C and 8E |
| [EGAS00001006694](https://ega-archive.org/studies/EGAS00001006694) | Dang et al., *Cancer Cell* 2023 | Fig. 5C; Extended Data Figs. 7C and 8E |
| [GSE161801](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161801) | Tirier et al., *Nat Commun* 2021 | Extended Data Fig. 5C |
| [GSE124310](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124310) | Zavidij et al., *Nat Cancer* 2020 | Signature activity in bone marrow cell types (Methods) |
| CoMMpass IA22 and IA24 ([MMRF Research Gateway](https://research.themmrf.org/)) | Skerget et al., *Nat Genet* 2024 | Figs. 3D-F, 5D; Supplementary Figs. 5B,C and 6 and Extended Data Fig. 9A,B |
| [g:Profiler](https://biit.cs.ut.ee/gprofiler/) e112 archive (GO and Reactome GMT files) | Kolberg et al., *Nucleic Acids Res* 2023 | Extended Data Fig. 5A |
| 10x Genomics reference GRCh38-2020-A | 10x Genomics | Gene IDs for Extended Data Fig. 5A |

### Local layout

The scripts read the processed data from `data/` (or the folder named by `SCPCMM_DATA`):

```
data/
├── metadata/    cohort metadata and cytogenetic calls
├── annotation/  per-cell annotation (malignant or normal PC, clone)
├── h5/          per-library objects (filtered/), doublet scores (doublets/), cohort objects and per-cell tables (agg/),
│                per-individual objects (pat/), public datasets (external/<study>/)
├── vdj/         BCR contigs and clonotypes, isotypes
├── amp1q/       dandelion objects (bcr/), Numbat outputs (cnv/)
├── cnv/         CNV segments, clones and drivers, cell-level and cytoband CNV calls, Numbat input table
├── pseudobulk/  pseudobulk matrices and DESeq2 results
├── scrna_de/    within-patient differential expression of malignant and normal PCs
├── barcode/     cells used for signature detection
├── signature/   gene lists, Hotspot inputs and modules, per-cell scores, features of the public datasets
├── classifier/  training and test sets, kNN-smoothed features, models, SHAP values, tuning tables, scores
├── paper/       clinical and cytogenetic tables, AnnData objects, scores, differential expression, Numbat clones,
│                colors and WGS tables used by the figures
├── pathway/     published signature gene lists, g:Profiler GMT files
├── reference/   refdata-gex-GRCh38-2020-A, gene names, chromosome lengths
└── commpass/    CoMMpass IA22 tables and scores, IA24 tables (ia24/)
```

## Setup

```bash
conda create -n scpcmm python=3.8.10
conda activate scpcmm
pip install -r requirements.txt               # analyses and 11_figure_inputs

conda create -n scpcmm-bcr python=3.8
conda activate scpcmm-bcr
pip install -r requirements_bcr.txt           # dandelion steps of 03_malignant_cells

conda create -n scpcmm-figures python=3.13
conda activate scpcmm-figures
pip install -r requirements_figures.txt       # figure scripts; pinned
```

```r
source("environment.R")                       # R packages
```

Command-line tools: bedtools (04_genomics) and Singularity with the sc-dandelion container (03_malignant_cells).

The random forest results depend on the version of scikit-learn (1.0.2): newer versions grow different trees from the
same seed. Paths are set in `config.py` and `config.R`, or with the environment variables `SCPCMM_DATA`, `SCPCMM_RESULTS`
and `SCPCMM_FIGURES`.

## Running

Run the folders in numeric order and, within a folder, the scripts in numeric order. Python scripts run from any
directory; R scripts run from their own folder. Scripts that process one library, tumor or individual at a time take
its name as an argument, and their header gives the loop over the cohort.

```bash
python3 01_cohort/01_cohort_structure.py
(cd 05_differential_expression && Rscript 03_deseq2_stage_comparisons.R)
for f in 12_main_figures/*.py 13_extended_data_figures/*.py 14_supplementary_figures/*.py; do PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=0 python3 "$f"; done
```

`11_figure_inputs/02_extract_cells.py` loads per-cell tables of 1.65 million cells and needs about 50 GB of memory.

## Reproducibility check

`.github/workflows/run-reproducibility-check.yml` runs `tools/check_reproducibility.py` weekly and appends the score to
`reproducibility.md`. It scores repository structure, not scientific correctness. Locally:

```bash
python3 tools/check_reproducibility.py .
```
