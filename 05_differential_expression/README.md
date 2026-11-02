# Differential expression

Differential expression between malignant and normal plasma cells, disease stages and cytogenetic groups.

| Script | Computes |
|--------|----------|
| `01_pseudobulk_matrices.py` | Pseudobulk count matrices and sample tables of the pseudobulk differential expression analyses (Methods, Differential expression analysis, pseudobulk level). |
| `02_deseq2_functions.R` | Functions of the pseudobulk DESeq2 comparisons (Methods, Differential expression analysis, pseudobulk level), sourced by 03-07. |
| `03_deseq2_stage_comparisons.R` | Pseudobulk DESeq2 comparisons by disease stage (Methods, Differential expression analysis, pseudobulk level), with the functions of 02_deseq2_functions.R. |
| `04_deseq2_stage_tumor_vs_all_normal.R` | Pseudobulk DESeq2 comparisons of the malignant BM PCs of each disease stage (MGUS, SMM, MM; stages from disease_status) with the normal BM PCs of all patients and healthy donors together (Methods, Differential expression analysis, pseudobulk level), with the functions of 02_deseq2_functions.R. |
| `05_deseq2_cytogenetic_groups.R` | Pseudobulk DESeq2 comparisons of BM PCs within each cytogenetic group of patients (Methods, Differential expression analysis, pseudobulk level), with the functions of 02_deseq2_functions.R. |
| `06_deseq2_bm_vs_pb_tumor_pairs.R` | Pseudobulk DESeq2 comparison of malignant PCs between BM and PB (circulating tumor cells) in the patients sampled in both tissues at the same stage (BM_PB_set Pair), with the functions of 02_deseq2_functions.R (Methods, Differential expression analysis, pseudobulk level). |
| `07_deseq2_bm_vs_pb_normal.R` | Pseudobulk DESeq2 comparison of normal PCs between BM and PB, patients and healthy donors together (Methods, Differential expression analysis, pseudobulk level), with the functions of 02_deseq2_functions.R: all pseudobulks, and the 5' (GEX5) pseudobulks. |
| `08_deseq2_normalized_counts.R` | DESeq2 size-factor normalized counts and variance-stabilized expression (vst, blind = FALSE) of all pseudobulks (Methods, Differential expression analysis, pseudobulk level). |
| `09_tumor_pseudobulk_gene_lists.py` | Malignant-PC gene lists from the pseudobulk DESeq2 comparisons by disease stage, including the candidate genes of signature detection (Methods, Detection of single-cell transcriptional signatures; Differential expression analysis). |
| `10_normal_pseudobulk_gene_lists.py` | Normal-PC gene lists from the pseudobulk DESeq2 comparisons with healthy donors, between stages and between BM and PB, including candidate genes of signature detection (Methods, Detection of single-cell transcriptional signatures). |
| `11_ctc_cytogenetics_gene_lists.py` | Pseudobulk DEG tables of malignant BM PCs vs circulating tumor cells (paired BM and PB) and of malignant vs normal PCs within each cytogenetic group (Methods, Detection of single-cell transcriptional signatures; Differential expression analysis). |
| `12_stage_dysregulated_genes.py` | Genes dysregulated in malignant BM PCs from MGUS on (early), from SMM on, or in MM only (late), from pseudobulk comparisons of the malignant PCs of each disease stage with the normal PCs of every stage and of healthy donors (Methods, Differential expression analysis, pseudobulk level). |
| `13_scrna_de_tumor_vs_normal.py` | Within-patient differential expression between malignant and normal BM PCs, per patient, disease stage and malignant population (Methods, Differential expression analysis, single-cell level; Fig. 2A). |
| `14_scrna_de_matrices.py` | Matrices of adjusted P values and log2 fold changes of the within-patient single-cell differential expression results, one column per comparison (Methods, Differential expression analysis, single-cell level; Fig. 2A). |
| `15_tumor_de_imwg_tables.py` | Within-patient differential expression tables of malignant vs normal BM PCs labelled with the IMWG 2/20/20 risk groups (Fig. 2A,D-G; Supplementary Figs. 1 and 2; Results). |
| `16_gprc5d_frequency.py` | Frequency of GPRC5D up- and downregulation in malignant versus normal PCs of the same patient, overall, by disease stage and in t(11;14) versus other tumors, and of CD19 downregulation (Results; Discussion; Methods, Differential expression analysis, single-cell level). |
| `17_six_genes_nsd2.py` | Genes consistently upregulated in malignant PCs at every disease stage, and their upregulation in NSD2 [t(4;14)] versus other tumors (Results; Methods, Differential expression analysis, single-cell level; Supplementary Fig. 2). |
| `18_target_heatmap_cohort.py` | Tumors included in and left out of the within-patient comparisons of malignant and normal PCs (Fig. 2A), and whether the included tumors represent all tumors (Methods, Cohort structure and repeat samples). |
| `19_deseq2_age_sex.R` | Refits the pseudobulk DESeq2 comparisons of BM PCs that select the genes for signature detection (Methods, Detection of single-cell transcriptional signatures, groups i-iv) with age and sex as covariates, and the malignant versus normal comparisons also with patient as a covariate, on the same pseudobulk samples (Results; Methods, Pseudobulk level). |
| `20_deseq2_imwg_risk_groups.R` | Pseudobulk DESeq2 comparisons of malignant vs normal BM PCs at each disease stage with SMM split by the IMWG 2/20/20 risk group (Fig. 2B), with the functions of 02_deseq2_functions.R. |
| `21_imwg_risk_pseudobulk_table.py` | Log2 fold changes of malignant vs normal BM PCs by disease stage, SMM split by the IMWG 2/20/20 risk group (Fig. 2B). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 05_differential_expression/01_pseudobulk_matrices.py
(cd 05_differential_expression && Rscript 02_deseq2_functions.R)
(cd 05_differential_expression && Rscript 03_deseq2_stage_comparisons.R)
(cd 05_differential_expression && Rscript 04_deseq2_stage_tumor_vs_all_normal.R)
(cd 05_differential_expression && Rscript 05_deseq2_cytogenetic_groups.R)
(cd 05_differential_expression && Rscript 06_deseq2_bm_vs_pb_tumor_pairs.R)
(cd 05_differential_expression && Rscript 07_deseq2_bm_vs_pb_normal.R)
(cd 05_differential_expression && Rscript 08_deseq2_normalized_counts.R)
python3 05_differential_expression/09_tumor_pseudobulk_gene_lists.py
python3 05_differential_expression/10_normal_pseudobulk_gene_lists.py
python3 05_differential_expression/11_ctc_cytogenetics_gene_lists.py
python3 05_differential_expression/12_stage_dysregulated_genes.py
python3 05_differential_expression/13_scrna_de_tumor_vs_normal.py
python3 05_differential_expression/14_scrna_de_matrices.py
python3 05_differential_expression/15_tumor_de_imwg_tables.py
python3 05_differential_expression/16_gprc5d_frequency.py
python3 05_differential_expression/17_six_genes_nsd2.py
python3 05_differential_expression/18_target_heatmap_cohort.py
(cd 05_differential_expression && Rscript 19_deseq2_age_sex.R)
(cd 05_differential_expression && Rscript 20_deseq2_imwg_risk_groups.R)
PYTHONHASHSEED=0 python3 05_differential_expression/21_imwg_risk_pseudobulk_table.py
```
