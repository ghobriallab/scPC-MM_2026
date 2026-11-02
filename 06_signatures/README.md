# Signatures

The 16 transcriptional signatures: detection, scoring, annotation, overlap with published signatures and relations between signatures.

| Script | Computes |
|--------|----------|
| `01_downsample_cells.py` | PCs used for signature discovery: at most 1,000 malignant and 1,000 normal PCs per individual and tissue, with their raw UMI tables and expression objects (Methods, detection of single-cell transcriptional signatures). |
| `02_hotspot_inputs.py` | The two Hotspot input objects of signature discovery, restricted to the 4,980-gene pool: the downsampled BM and PB PCs, and the downsampled BM PCs (Methods, detection of single-cell transcriptional signatures). |
| `03_modules_bm_pb.py` | Annotation of the Hotspot gene modules of the BM and PB run: flags for modules driven by one gene or one individual, Kruskal-Wallis and Dunn tests across disease stage, tissue, cell type and cytogenetic group, and module names (Methods, detection of single-cell transcriptional signatures). |
| `04_modules_bm.py` | Annotation of the Hotspot gene modules of the BM run: flags for modules driven by one gene or one individual, Kruskal-Wallis and Dunn tests across disease stage, cell type and cytogenetic group, and module names (Methods, detection of single-cell transcriptional signatures). |
| `05_consensus_signatures.py` | The 16 consensus signatures from the BM and the BM and PB Hotspot runs: genes of each signature with their Hotspot statistics, and the gene-gene local-correlation Z matrix ordered by signature (Fig. 3A,B; Methods, detection of single-cell transcriptional signatures). |
| `06_signature_scores.py` | Per-cell activity of the 16 signatures and of six stage-specific DEG lists, one individual at a time: mean and sum of the log-normalized expression of their genes, and AUCell scores (Methods, signature activity). |
| `07_aggregate_scores_and_trim_genes.py` | Per-cell table of signature gene expression and signature scores for all PCs, and the trimmed signatures (top 5, 10, 15 and 20 genes of each signature) used as classifier feature sets (Methods, signature activity and classifiers). |
| `08_trimmed_signature_scores.py` | Per-cell activity of the trimmed signatures (top 5, 10, 15 and 20 genes of each of the 16 signatures), one individual at a time: mean log-normalized expression of their genes and AUCell scores (classifier feature sets). |
| `09_aggregate_trimmed_scores.py` | Per-cell table of the trimmed signature scores for all PCs, and the feature-column lists of the trimmed signatures used by the classifiers (Methods, classifiers). |
| `10_multi_stage_samples.py` | Raw counts of the PCs of the individuals sampled at two disease stages, the input of the longitudinal UMAP (Supplementary Fig. 5A and Extended Data Fig. 8C). |
| `11_published_gene_lists.py` | Gene lists of published myeloma signatures matched to the gene names of the expression data (CoMMpass RNA subtypes, Zhan et al. subgroups, Boiarsky et al. programs, GEP70 and GEP5 risk scores, RAS signatures) and the table of the six stage-specific DEG lists (Methods, scores of published signatures). |
| `12_published_signature_scores.py` | Per-cell activity of the published gene sets used by the classifiers, one individual at a time: mean and sum of the log-normalized expression of their genes, and AUCell scores (Methods, Development of malignancy and MST scores). |
| `13_aggregate_published_scores.py` | Per-cell tables of the published gene-set scores used by the classifiers for all PCs, with the expression of the SenMayo and NF-kB index genes (Methods, Development of malignancy and MST scores). |
| `14_cell_annotation.py` | Per-cell annotation table of the 1,649,290 PCs, read by the later steps (cell type, cytogenetic group, disease stage and analysis units of every PC). |
| `15_subtype_gene_lists.py` | Official gene symbols and Ensembl IDs of the genes of the Zhan et al. subgroups and the CoMMpass RNA subtypes (Methods, comparison with published signatures). |
| `16_top_variable_genes.py` | The 5,000 most highly variable genes of the 1,649,290 PCs, for the test of their overlap with the gene pool of signature discovery (Methods, detection of single-cell transcriptional signatures). |
| `17_published_signature_overlap.py` | Gene overlap between the 16 signatures and 42 published myeloma and plasma cell signatures (Extended Data Fig. 5B; Results; Methods, comparison with published signatures). |
| `18_gene_ids.py` | Ensembl gene IDs of the gene symbols of the expression matrices, for the GO enrichment (Extended Data Fig. 5A). |
| `19_go_enrichment.py` | Gene Ontology and Reactome terms enriched among the genes of each of the 16 signatures (Extended Data Fig. 5A). |
| `20_deg_hvg_overlap.py` | Overlap of the 4,980 genes used for signature discovery with the 5,000 most highly variable genes, and the number of genes tested in the pseudobulk comparisons (Methods, detection of single-cell transcriptional signatures). |
| `21_signature_correlations.py` | Genes per signature, genes shared between signatures and Spearman correlations between the activities of the 16 signatures across the BM tumors (Extended Data Fig. 5D; Results; Methods, detection of single-cell transcriptional signatures). |
| `22_b_cell_activation_normal_pcs.py` | B cell activation signature in malignant and normal BM PCs: same-patient comparison, change across disease stages in malignant PCs, and healthy donors versus patients in normal PCs (Results; Discussion). |
| `23_hotspot_cell_cap.py` | Composition of the PCs used for signature discovery (at most 1,000 PCs per individual, tissue and cell type) compared with all PCs, by disease stage, cytogenetic group, hyperdiploidy and age (Methods, detection of single-cell transcriptional signatures). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 06_signatures/01_downsample_cells.py
python3 06_signatures/02_hotspot_inputs.py
python3 06_signatures/03_modules_bm_pb.py
python3 06_signatures/04_modules_bm.py
python3 06_signatures/05_consensus_signatures.py
python3 06_signatures/06_signature_scores.py
python3 06_signatures/07_aggregate_scores_and_trim_genes.py
python3 06_signatures/08_trimmed_signature_scores.py
python3 06_signatures/09_aggregate_trimmed_scores.py
python3 06_signatures/10_multi_stage_samples.py
python3 06_signatures/11_published_gene_lists.py
python3 06_signatures/12_published_signature_scores.py
python3 06_signatures/13_aggregate_published_scores.py
python3 06_signatures/14_cell_annotation.py
python3 06_signatures/15_subtype_gene_lists.py
python3 06_signatures/16_top_variable_genes.py
python3 06_signatures/17_published_signature_overlap.py
python3 06_signatures/18_gene_ids.py
python3 06_signatures/19_go_enrichment.py
python3 06_signatures/20_deg_hvg_overlap.py
python3 06_signatures/21_signature_correlations.py
python3 06_signatures/22_b_cell_activation_normal_pcs.py
python3 06_signatures/23_hotspot_cell_cap.py
```
