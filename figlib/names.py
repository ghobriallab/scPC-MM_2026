"""Display names used in the figures, matching the wording of the manuscript legends.

Purpose:      Map the internal names of the 16 signatures and of the classifier features to their display names.

Inputs:       None.

Outputs:      SIGNATURES, SIGNATURE_ORDER, FEATURE_NAMES and feature(), imported by the figure scripts.

Dependencies: Python 3.8+ (standard library only).
"""

# the 16 transcriptional signatures (internal name -> display name; "signature" added where the short name is also a
# gene or a cytogenetic group, as the legends should read)
SIGNATURES = {
    "cell_cycling_G2M": "Cell cycling G2/M",
    "cell_cycling_G1S": "Cell cycling G1/S",
    "B_cell_activation": "B cell activation",
    "MHC-II_immune_checkpoint": "MHC-II immune checkpoint",
    "FOS-JUN_stress_response": "FOS-JUN stress response",
    "MCL1-BCL2_signature": "MCL1-BCL2 signature",
    "circulatory_capacity": "Circulatory capacity",
    "interferon_inducible": "Interferon inducible",
    "HRD_signature": "HRD signature",
    "secretory_granule": "Secretory granule",
    "myeloma_oncogene": "Myeloma oncogene",
    "MAF*_signature": "MAF* signature",
    "MHC-I_gene_module": "MHC-I gene module",
    "vacuole_ER_signaling": "Vacuole/ER signaling",
    "histone_chaperone_gene": "Histone/chaperone",
    "extracellular_matrix": "Extracellular matrix",
}
SIGNATURE_ORDER = list(SIGNATURES)          # order of Supplementary Fig. 8 (A-P)
# as classifier features, signatures whose short name is also a gene or a cytogenetic group carry "signature"
FEATURE_NAMES = {"HRD_signature": "HRD signature", "MAF*_signature": "MAF* signature",
                 "MCL1-BCL2_signature": "MCL1-BCL2 signature"}


def feature(name):
    """Classifier feature (e.g. 'auc:histone_chaperone_gene', 'mean_expr:14q_full_log_fc') -> display name."""
    base = name.split(":")[-1]
    if base in SIGNATURES:
        return FEATURE_NAMES.get(base, SIGNATURES[base])
    if base.endswith("_full_log_fc"):
        return base.replace("_full_log_fc", "") + " CNV-like score"
    if base == "SenMayo":
        return "Senescence (SenMayo)"
    return base.replace("_", " ")
