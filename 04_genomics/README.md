# Genomics

Copy number variants inferred from scRNA-seq, clones and cytogenetic classification.

| Script | Computes |
|--------|----------|
| `01_combine_library_counts.py` | Merge the count matrices of the libraries of one tumor into one 10x-format HDF5 file, the count input of Numbat (Methods, Copy number variant detection). |
| `02_combine_library_alleles.py` | Merge the allele counts of the libraries of one tumor into one table with harmonized phasing, the allele input of Numbat (Methods, Copy number variant detection). |
| `03_run_numbat.R` | Numbat on one tumor (Methods, Copy number variant detection): reads the merged count matrix (Read10X_h5), keeps the cells listed in --barcodes when that file exists, reads the merged allele table and runs run_numbat (genome hg38; reference expression: Numbat's ref_hca, or the RDS file given with --refexpr; plot TRUE). |
| `04_clone_mutation_graphs.R` | Node and edge tables of the Numbat clone trees (mutation graph of iteration 2), one pair per tumor (Methods, Copy number variant detection). |
| `05_cnv_segments.py` | Consensus CNV segments of the tumors with CNVs, with the genes, cytobands and chromosome arms they cover (Methods, Copy number variant detection; segment table read by every later CNV step). |
| `06_cnv_cell_matrix_tumor.py` | Cell-by-cytoband CNV table of one tumor from its Numbat cell posteriors (input of the CNV matrix of all cells, 08_cnv_cytoband_matrix.py; Methods, Copy number variant detection). |
| `07_run_cnv_cell_matrix.sh` | Runs 06_cnv_cell_matrix_tumor.py once for every tumor with CNV segments, that is every distinct tumor_name of data/cnv/cnv_segments_152_tumors.tsv (153 tumors of 152 patients). |
| `08_cnv_cytoband_matrix.py` | CNV state and probability of every cell with a Numbat CNV call, per cytoband and chromosome arm, for all tumors (Methods, Copy number variant detection). |
| `09_cell_level_cnv.py` | CNV probabilities per malignant PC and segment, and the CNV clone of every malignant PC (Methods, Copy number variant detection). |
| `10_clone_driver_arms.py` | Driver arm CNVs of every malignant PC with a CNV clone, and of every CNV segment (Methods, Copy number variant detection). |
| `11_clone_phylogeny.py` | Phylogeny of the CNV clones: tree position, genotype and driver arm CNVs of every clone in every sample group (Methods, Copy number variant detection). |
| `12_genome_cytoband_bins.py` | Genome-wide CNV maps of the CNV clones on 5-kb genome bins and on cytobands (Methods, Copy number variant detection). |
| `13_cnv_gene_expression_scores.py` | Copy-number-like expression scores of the driver arms 1p, 1q, 13q, 14q and 17p in every cell, used as classifier features, and the expression of the translocation partner genes (Methods, Development of malignancy and MST scores). |
| `14_cnv_negative_cells.py` | Malignant PCs without detectable CNV: tumors without any CNV by cytogenetic group, stage and cell number, CNV-negative cells within tumors with CNVs and their kNN neighborhoods, and CNV clones by stage (Results; Methods, Copy number variant detection; Extended Data Figs. 1C and 2). |
| `15_two_clone_tumors.py` | Patients with two malignant PC populations: BCR, constant genes and partner genes (Results; Methods, Classification of cytogenetic alterations). |
| `16_cytogenetic_call_robustness.py` | Scale and support of the scRNA-seq cytogenetic calls: population sizes, partner-gene detection and HRD calls against FISH and WGS (Methods, Classification of cytogenetic alterations). |
| `17_translocation_classifier.R` | Translocation classifier: multinomial logistic regression fitted leaving out one tumor at a time, final model as the median of the coefficients, calls per population and tissue, then the reviewed calls by patient (Methods, Classification of cytogenetic alterations). |
| `18_translocation_classifier_accuracy.py` | Accuracy of the classifier: leave-one-out sensitivity and specificity, accuracy with few malignant PCs, calls on the normal PCs of donors and patients by tissue, PB and BM agreement, reviewed calls (Results; Methods, Classification of cytogenetic alterations; Extended Data Fig. 4). |

`translocation_classifier_coefficients.tsv` holds the coefficients of the final model (one row per translocation class, HRD as reference; columns: intercept and the 14 features). `Rscript 17_translocation_classifier.R --score-only` applies them without refitting.

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 04_genomics/01_combine_library_counts.py
python3 04_genomics/02_combine_library_alleles.py
(cd 04_genomics && Rscript 03_run_numbat.R)
(cd 04_genomics && Rscript 04_clone_mutation_graphs.R)
python3 04_genomics/05_cnv_segments.py
python3 04_genomics/06_cnv_cell_matrix_tumor.py
bash 04_genomics/07_run_cnv_cell_matrix.sh
python3 04_genomics/08_cnv_cytoband_matrix.py
python3 04_genomics/09_cell_level_cnv.py
python3 04_genomics/10_clone_driver_arms.py
python3 04_genomics/11_clone_phylogeny.py
python3 04_genomics/12_genome_cytoband_bins.py
python3 04_genomics/13_cnv_gene_expression_scores.py
python3 04_genomics/14_cnv_negative_cells.py
python3 04_genomics/15_two_clone_tumors.py
python3 04_genomics/16_cytogenetic_call_robustness.py
(cd 04_genomics && Rscript 17_translocation_classifier.R)
python3 04_genomics/18_translocation_classifier_accuracy.py
```
