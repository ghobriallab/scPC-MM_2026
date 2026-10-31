# Classifiers and scores

The malignancy and MST random forest classifiers: training, tuning, scores, external datasets, feature contributions, compact and extended models.

| Script | Computes |
|--------|----------|
| `01_training_test_split.py` | Training and test sets of the malignancy and MGUS-MM classifiers: 75/25 split of the eligible individuals within each disease stage (Methods, classifier development). |
| `02_knn_smoothing_train_test.py` | kNN smoothing of the per-cell classifier features of the training and test sets, for k = 10 to 100 neighbors (Methods, classifier development; Supplementary Fig. 12C). |
| `03_knn_smoothing_all_cells.py` | kNN smoothing of the classifier features of all PCs, k = 30 and 50: the input of the cell-level malignancy and MST scores (Methods, classifier development). |
| `04_tumor_normal_feature_sets.py` | Cross-validated accuracy of malignant versus normal PC classifiers for 32 feature sets, three versions of the training data and two model types (Supplementary Fig. 12A; Methods, classifier development). |
| `05_two_tier_feature_sets.py` | Cross-validated accuracy of the two-tier classifier (malignant vs normal, then MGUS vs MM) for every combination of the feature sets of the two tiers (Supplementary Fig. 12B; Methods, classifier development). |
| `06_feature_set_ranking.py` | Ranking of the feature sets of the malignancy and MGUS-MM tiers by cross-validated accuracy, the basis of the feature choice of the final classifiers (Methods, classifier development; Supplementary Fig. 12A,B). |
| `07_grid_search_full_models.py` | Grid search of the random forest parameters for five candidate models: two one-step three-class models and three two-tier models (Methods, classifier development). |
| `08_grid_search_tiers.py` | Grid search of the random forest parameters of each tier on its own: the MGUS-MM tier and three versions of the malignancy tier (Methods, classifier development; Supplementary Fig. 12D). |
| `09_tuned_two_tier_performance.py` | Cross-validated accuracy of the two-tier classifier with the tuned parameters of each tier, for the three candidate malignancy tiers (Methods, classifier development). |
| `10_knn_neighbors_accuracy.py` | Cross-validated performance of the final two-tier classifier for k = 0 (no smoothing) to 100 neighbors of the kNN smoothing (Supplementary Fig. 12C; Methods, classifier development). |
| `11_shap_training_folds.py` | Training and held-out cells of the five cross-validation folds of the two-tier classifier, for the SHAP feature ranking (the ranking used by the feature ablation, Supplementary Fig. 12E). |
| `12_shap_values_malignancy.py` | SHAP values of the malignancy random forest for one cross-validation fold (feature ranking used by the feature ablation, Supplementary Fig. 12E). |
| `13_shap_values_mst.py` | SHAP values of the MGUS-MM random forest (the MST tier) for one cross-validation fold (feature ranking used by the feature ablation, Supplementary Fig. 12E). |
| `14_shap_feature_importance.py` | Mean absolute SHAP value of each feature of the two classifier tiers over the cross-validation folds, and cross-validated performance of feature subsets chosen from it (feature ranking used by the feature ablation, Supplementary Fig. 12E). |
| `15_final_models_and_cell_scores.py` | Final malignancy and MGUS-MM random forests trained on the training and test cells, and the malignancy and MST scores of every PC with their means per sample group (Methods, classifier development; Fig. 4D,E). |
| `16_patient_and_tissue_mean_scores.py` | Malignancy and MST scores of every PC and their means per patient and per tissue sample, by disease stage, tissue and PC status, with the comparisons between groups (Fig. 4D,E; Fig. 5; Supplementary Fig. 14). |
| `17_test_set_performance.py` | Performance of the two-tier classifier trained on the training set and applied to the test set, with kNN smoothing over 30 and 50 neighbors (Fig. 4B,C; Results and Methods, classifier performance); also writes the unsmoothed per-cell feature table of all PCs. |
| `18_external_signature_scores.py` | Signature features of the cells of the public single-cell datasets: mean expression and AUCell scores, inputs of the malignancy and MST scores (Methods, calculation of malignancy and MST scores on external single-cell data). |
| `19_external_cnv_arm_scores.py` | CNV-arm features of the cells of the public single-cell datasets, inputs of the MST score (Methods, calculation of malignancy and MST scores on external single-cell data). |
| `20_external_partner_genes.py` | Expression of the nine translocation partner genes in the cells of the public single-cell datasets, inputs of the malignancy score (Methods, calculation of malignancy and MST scores on external single-cell data). |
| `21_external_knn_features.py` | kNN-smoothed features of the cells of the public single-cell datasets, the inputs of their malignancy and MST scores (Fig. 5C; Supplementary Figs. 13C, 14E; Methods, calculation of malignancy and MST scores on external single-cell data). |
| `22_external_mst_scores.py` | MST scores of the PCs of three external single-cell studies and the separation of disease stages, with effect sizes and 95% CIs (Fig. 5C; Results, external validation). |
| `23_external_malignancy_scores.py` | Malignancy scores of the PCs of three external single-cell studies compared with the malignant and normal PC labels of the original authors (Supplementary Fig. 13C; Results, external validation). |
| `24_nfkb_index_model.py` | MGUS-MM classifier refit with and without the published NF-kB index as an extra feature (Results and Methods, MGUS-MM classifier). |
| `25_stage_free_signature_test.py` | MGUS-MM classification with signatures restricted to genes selected without comparing disease stages (Results and Methods, MGUS-MM classifier). |
| `26_feature_ablation.py` | Feature ablation of the malignancy and MGUS-MM random forest classifiers (Supplementary Fig. 12E; Results and Methods). |
| `27_compact_model.py` | A compact MGUS-MM classifier that keeps only the top-ranked genes of each signature (Results and Methods, MGUS-MM classifier). |
| `28_misclassified_mgus_cells.py` | Malignant MGUS PCs that the MGUS-MM classifier calls MM: test-set recall, out-of-sample scores and within-patient comparisons (Fig. 4B,C; Supplementary Fig. 13A,B; Results and Methods). |

Inputs are the processed data (controlled access, dbGaP phs003855; see the root README for the `data/` layout) and the tables written by earlier steps in `results/`. Each script lists its inputs and outputs in its header.

## Run

```bash
python3 08_classifiers/01_training_test_split.py
python3 08_classifiers/02_knn_smoothing_train_test.py
python3 08_classifiers/03_knn_smoothing_all_cells.py
python3 08_classifiers/04_tumor_normal_feature_sets.py
python3 08_classifiers/05_two_tier_feature_sets.py
python3 08_classifiers/06_feature_set_ranking.py
python3 08_classifiers/07_grid_search_full_models.py
python3 08_classifiers/08_grid_search_tiers.py
python3 08_classifiers/09_tuned_two_tier_performance.py
python3 08_classifiers/10_knn_neighbors_accuracy.py
python3 08_classifiers/11_shap_training_folds.py
python3 08_classifiers/12_shap_values_malignancy.py
python3 08_classifiers/13_shap_values_mst.py
python3 08_classifiers/14_shap_feature_importance.py
python3 08_classifiers/15_final_models_and_cell_scores.py
python3 08_classifiers/16_patient_and_tissue_mean_scores.py
python3 08_classifiers/17_test_set_performance.py
python3 08_classifiers/18_external_signature_scores.py
python3 08_classifiers/19_external_cnv_arm_scores.py
python3 08_classifiers/20_external_partner_genes.py
python3 08_classifiers/21_external_knn_features.py
python3 08_classifiers/22_external_mst_scores.py
python3 08_classifiers/23_external_malignancy_scores.py
python3 08_classifiers/24_nfkb_index_model.py
python3 08_classifiers/25_stage_free_signature_test.py
python3 08_classifiers/26_feature_ablation.py
python3 08_classifiers/27_compact_model.py
python3 08_classifiers/28_misclassified_mgus_cells.py
```
