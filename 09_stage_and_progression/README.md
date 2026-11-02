# Disease stage and progression

Stage associations of the signatures and scores, progression in smoldering myeloma and MM-like cells.

| Script | Computes |
|--------|----------|
| `01_stage_comparisons.py` | Stage comparisons of tumor fraction, signature activity and malignancy and MST scores per patient (Figs. 3C and 4D,E; Extended Data Fig. 1B and Supplementary Figs. 3 and 4). |
| `02_stage_effects_genetics.py` | Stage associations of the MST score and of the five progression-associated signatures with and without adjustment for genetic alterations, and malignancy and MST scores by cytogenetic group (Extended Data Figs. 8D and 10A; Results). |
| `03_stage_effects_demographics.py` | Stage associations of the MST score and of the five progression-associated signatures adjusted for age and sex, alone and together with the genetic alterations (Results). |
| `04_demographics_by_stage.py` | Age, sex and self-reported race by disease stage, their comparisons between stages, and stage associations adjusted for race (Supplementary Table 3; Methods). |
| `05_minimum_cell_threshold.py` | Stage correlations of the MST score and of the five progression-associated signatures after removing units with few malignant PCs (Methods, Cohort structure and repeat samples). |
| `06_longitudinal_changes.py` | Changes in signature activity and MST score of malignant PCs between the first and last samples of individuals sampled at several visits, within all malignant PCs and within the largest shared CNV clone (Results, Extended Data Fig. 8C paragraph; Methods, Longitudinal samples). |
| `07_smm_progression.py` | MST scores of SMM patients who later progressed to MM versus those who did not, and comparison with the IMWG 2/20/20 risk group (Fig. 5A; Extended Data Fig. 8A,B; Results). |
| `08_mm_like_cells.py` | Percentage of MM-like malignant PCs (MST score above 0.5) in each BM tumor and in each of its Numbat CNV clones (Figs. 5A and 6; Results). |
| `09_mm_like_cells_statistics.py` | MM-like cells in SMM tumors by IMWG 2/20/20 risk group and progression to MM, with the AUROC of their percentage for progression, and the same statistics for the mean MST score (Fig. 5A; Extended Data Fig. 8B; Results). |
| `10_histone_dna_content.py` | Histone/chaperone signature activity and tumor ploidy from whole-genome sequencing: ploidy by disease stage, its correlation with the signature, and the stage association adjusted for ploidy or HRD (Extended Data Fig. 10B-D; Results). |
| `11_mm_like_cells_out_of_sample.py` | MM-like cells in every BM tumor scored out of sample: percentage by disease stage, across the Numbat CNV clones of each tumor, and in MGUS tumors (Fig. 6; Results). |
| `12_gprc5d_t1114_1q_gain.py` | GPRC5D upregulation in t(11;14) versus other tumors adjusted for 1q gain (Results). |
| `13_signature_smm_progression.py` | Activity of each of the 16 signatures in SMM patients who later progressed to MM versus those who did not (Results; Methods). |
| `14_smm_progression_followup.py` | Progression analyses of the MST score and of MM-like cells restricted to non-progressors with at least 3 or 4 years of follow-up (Results; Methods). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 09_stage_and_progression/01_stage_comparisons.py
python3 09_stage_and_progression/02_stage_effects_genetics.py
python3 09_stage_and_progression/03_stage_effects_demographics.py
python3 09_stage_and_progression/04_demographics_by_stage.py
python3 09_stage_and_progression/05_minimum_cell_threshold.py
python3 09_stage_and_progression/06_longitudinal_changes.py
python3 09_stage_and_progression/07_smm_progression.py
python3 09_stage_and_progression/08_mm_like_cells.py
python3 09_stage_and_progression/09_mm_like_cells_statistics.py
python3 09_stage_and_progression/10_histone_dna_content.py
python3 09_stage_and_progression/11_mm_like_cells_out_of_sample.py
python3 09_stage_and_progression/12_gprc5d_t1114_1q_gain.py
python3 09_stage_and_progression/13_signature_smm_progression.py
python3 09_stage_and_progression/14_smm_progression_followup.py
```
