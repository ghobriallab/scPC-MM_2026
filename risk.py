"""SMM risk groups of the IMWG 2/20/20 model, for every table keyed by patient, tissue and disease stage.

Purpose:      Gives each SMM sample its IMWG 2/20/20 risk group from the cohort table (SMM_stage_strict: one point each
              for bone marrow PCs > 20% on biopsy, serum M-protein > 2 g/dL and involved/uninvolved serum free light
              chain ratio > 20; 0 points low risk, 1 point intermediate risk, 2 or 3 points high risk). The cell tables
              and the per-unit tables carry the stage of the cell annotation, in which three PB samples of two patients
              have another risk group than their clinical records (CLINICAL_RISK); stages() applies that correction,
              then the IMWG group. MGUS, MM and donor labels are returned unchanged. Units that end up sharing
              patient, stage, tissue and PC status are pooled by the calling scripts (cell-count weighted means).

Inputs:       data/metadata/tumor-cohort-metadata-final-240325.csv: patient_id, sample_type, disease_status,
              SMM_stage_strict (progression_reached(): also gex_id, timepoint).
              data/metadata/updated_progression.csv: gex_id, latest_smm_diagnosis_category (progression_reached()).
              data/paper/cohort-clinical-metadata.csv: patient_id, progression_stage (progression_reached()).

              progression_reached() gives the most advanced stage reached after sampling by each patient whose disease
              progressed (Fig. 1B): MM for the patients recorded as progressed to MM, otherwise the highest IMWG group
              among the patient's later samples and the latest clinical 2/20/20 risk category (scored with the same
              cut-offs), when it is above the group of the first visit.

Outputs:      SMM, CLINICAL_RISK, risk_map(), stages(), pool_units() and progression_reached(), imported by the analysis
              and figure scripts.

Dependencies: Python 3.8+, pandas; config.py.
"""
import pandas as pd

from config import data

SMM = ("LRSMM", "IRSMM", "HRSMM")
CLINICAL_RISK = {("Pt-90", "PB", "LRSMM"): "IRSMM", ("Pt-95", "PB", "LRSMM"): "HRSMM"}   # cell-table group -> clinical record
_MAP = {}


def risk_map():
    """(patient_id, sample_type, recorded SMM stage) -> IMWG 2/20/20 risk group; one group per key."""
    if not _MAP:
        m = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"),
                        usecols=["patient_id", "sample_type", "disease_status", "SMM_stage_strict"])
        m = m[m["disease_status"].isin(SMM)]
        g = m.groupby(["patient_id", "sample_type", "disease_status"])["SMM_stage_strict"].unique()
        assert all(len(v) == 1 and v[0] in SMM for v in g), "one IMWG risk group per patient, tissue and stage"
        _MAP.update({k: v[0] for k, v in g.items()})
    return _MAP


def stages(patients, tissues, statuses):
    """Stage of each row: clinical-record correction of the three PB samples, then the IMWG risk group for SMM."""
    m = risk_map()
    keys = list(zip(*(pd.Series(list(x)).astype(str) for x in (patients, tissues, statuses))))
    look = {}
    for p, t, s in set(keys):
        s2 = CLINICAL_RISK.get((p, t, s), s)
        look[(p, t, s)] = m[(p, t, s2)] if s2 in SMM else s2
    return [look[k] for k in keys]


def pool_units(df, keys, weight, values):
    """Rows of a per-unit table that share the keys (after stages()) become one row: means of the value columns
    weighted by the weight column (cells per unit), weights summed; other columns keep the first row's value."""
    out = []
    for _, g in df.groupby(keys, sort=False, observed=True):
        r = g.iloc[0].copy()
        if len(g) > 1:
            w = g[weight].astype(float)
            for c in values:
                v = g[c].astype(float)
                ok = v.notna()
                r[c] = (v[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else float("nan")
            r[weight] = w.sum()
        out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)


def progression_reached():
    """patient_id -> most advanced stage reached after sampling, for the patients whose disease progressed."""
    rank = {"MGUS": 0, "LRSMM": 1, "IRSMM": 2, "HRSMM": 3}
    m = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"),
                    usecols=["gex_id", "patient_id", "sample_type", "disease_status", "timepoint"])
    latest = pd.read_csv(data("metadata", "updated_progression.csv"), usecols=["gex_id", "latest_smm_diagnosis_category"])
    m = m.merge(latest, on="gex_id", how="left")
    m["stage"] = stages(m["patient_id"], m["sample_type"], m["disease_status"])
    pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"), usecols=["patient_id", "progression_stage"])
    out = {str(p): "MM" for p in pub.loc[pub["progression_stage"] == "MM", "patient_id"].unique()}
    for p, g in m.groupby("patient_id"):
        p = str(p)
        first = g[g["timepoint"] == g["timepoint"].min()]["stage"].unique()
        assert len(first) == 1, "one stage at the first visit"
        if p in out or first[0] not in rank:
            continue
        later = list(g.loc[g["timepoint"] > g["timepoint"].min(), "stage"]) + list(g["latest_smm_diagnosis_category"].dropna())
        later = [s for s in later if s in rank]
        if later and max(rank[s] for s in later) > rank[first[0]]:
            out[p] = max(later, key=rank.get)
    return out
