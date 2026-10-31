# CoMMpass

Processing, scores and survival models in the CoMMpass bulk RNA-seq cohort (MMRF).

| Script | Computes |
|--------|----------|
| `01_commpass_expression_and_scores.py` | CoMMpass IA22 bulk RNA-seq of CD138-selected samples: gene-level tables, baseline BM metadata and survival, BM-PB pairs, and signature, CNV-arm and partner-gene scores of the baseline BM samples (inputs of Figs. 3D-F, 5D; Supplementary Figs. 9B,C, 10, 15A,B). |
| `02_commpass_mst_scores.py` | Malignancy and MST scores of the CoMMpass baseline BM samples from the final random forest classifiers (Fig. 5D; Supplementary Fig. 15A,B). |
| `03_imwg2025_risk.py` | IMS/IMWG 2025 high-risk classification of CoMMpass patients for the Cox models (Results; Supplementary Fig. 15A,B; Methods). |
| `04_mst_cox_risk_models.R` | Tests whether the MST score adds prognostic information to clinical and cytogenetic risk factors in newly diagnosed myeloma (CoMMpass, IA22 endpoints; Results, Supplementary Fig. |
| `05_signature_cox_age_sex.R` | Refits the multivariable Cox model of the progression-associated signatures in CoMMpass (Fig. |
| `06_cox_race.R` | Sensitivity of the CoMMpass survival associations to self-reported race (Methods). |
| `07_circulatory_signature.py` | Circulatory capacity signature in CoMMpass and in paired PB and BM samples (Results; Supplementary Fig. 9B-D; Methods). |
| `08_cell_cycle_signature_cox.R` | G1/S and G2/M cell-cycle signatures in the Cox model of Fig. 3F: their correlation, each signature alone and both together (Results; Methods). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 10_commpass/01_commpass_expression_and_scores.py
python3 10_commpass/02_commpass_mst_scores.py
python3 10_commpass/03_imwg2025_risk.py
(cd 10_commpass && Rscript 04_mst_cox_risk_models.R)
(cd 10_commpass && Rscript 05_signature_cox_age_sex.R)
(cd 10_commpass && Rscript 06_cox_race.R)
python3 10_commpass/07_circulatory_signature.py
(cd 10_commpass && Rscript 08_cell_cycle_signature_cox.R)
```
