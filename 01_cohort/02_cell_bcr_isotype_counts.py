#!/usr/bin/env python3
"""Descriptive counts of the cohort reported in the Results and Methods: PCs per individual and stage, BCR coverage,
clonal isotypes, immunofixation concordance, healthy-donor samples, cytogenetic assays and the cells used for signature
detection.

Purpose:      From the cell table (one row per PC passing QC): PCs per individual (median and range, all samples
              pooled, donors included); malignant PCs from MGUS and SMM; PCs with a BCR clonotype and individuals
              without scBCR-seq data. Clonal isotype of each patient with malignant BM PCs covered by scBCR-seq: the
              heavy-chain class (constant gene) carried by most of these PCs, or none when most lack a heavy-chain call.
              Immunofixation concordance: for each patient with serum immunofixation and malignant PCs covered by
              scBCR-seq (BM and PB; oligosecretory results excluded), the majority light chain must be among the light
              chains of the immunofixation result, and the majority heavy-chain class (when called) among its heavy
              chains. Healthy-donor samples by source. Patients with FISH and with WGS results (Supplementary Table 4).
              Cells used for signature detection: at most 1,000 malignant and 1,000 normal PCs per individual and
              tissue.

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (cell table): redcap_id, sample_type, annotation
              (tumor = malignant, normal), disease_status, has_vdj, heavy_c_gene, light_c_gene.
              data/metadata/tumor-cohort-metadata-final-240325.csv: redcap_id, patient_id, disease_status, sample_type,
              tissue_id, NBM_Source.
              data/paper/cohort-clinical-metadata.csv (publication table): patient_id, IFX (immunofixation result,
              components separated by '.', e.g. IgG_Kappa.FF_Kappa).
              data/paper/cytogenetics-status.csv (Supplementary Table 4): patient_id, FISH, WGS.

Outputs:      results/cohort/cell_bcr_isotype_counts.tsv: quantity, value, definition.
              results/cohort/cell_bcr_isotype_counts_log.txt: the same counts as printed.

Dependencies: Python 3.8 with requirements.txt (pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import pandas as pd

OUT = result_dir("cohort")
FH = open(OUT / "cell_bcr_isotype_counts_log.txt", "w")
rows = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


def report(quantity, value, definition):
    rows.append({"quantity": quantity, "value": value, "definition": definition})
    log("%-58s %s" % (quantity, value))


SMM = ["LRSMM", "IRSMM", "HRSMM"]

# ---- inputs -------------------------------------------------------------------------------------
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
cy = pd.read_csv(data("paper", "cytogenetics-status.csv"))
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells[["redcap_id", "sample_type", "annotation", "disease_status", "has_vdj", "heavy_c_gene",
               "light_c_gene"]].copy()
for c in ["redcap_id", "sample_type", "annotation", "disease_status"]:
    cells[c] = cells[c].astype(str)
pid = meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"]
cells["patient_id"] = cells["redcap_id"].map(pid)
assert cells["patient_id"].notna().all()
mal = cells[cells["annotation"] == "tumor"]

# ---- 1. PCs ---------------------------------------------------------------------------------------
pp = cells.groupby("patient_id").size()
report("PCs passing QC", len(cells), "rows of the cell table")
report("malignant / normal PCs", "%d / %d" % (len(mal), (cells["annotation"] == "normal").sum()), "annotation")
report("PCs per individual, median", pp.median(), "all samples of an individual pooled, donors included")
report("PCs per individual, range", "%d-%d" % (pp.min(), pp.max()), "same")
pre = mal["disease_status"].isin(["MGUS"] + SMM).sum()
report("malignant PCs from MGUS and SMM", "%d (%.1f%% of malignant PCs)" % (pre, 100 * pre / len(mal)),
       "disease stage of the sample")

# ---- 2. BCR coverage ------------------------------------------------------------------------------
vdj = cells["has_vdj"].astype(str) == "True"
per_ind = vdj.groupby(cells["patient_id"]).sum()
none = per_ind[per_ind == 0].index
report("PCs with a BCR clonotype", "%d (%.2f%%)" % (vdj.sum(), 100 * vdj.mean()), "has_vdj")
report("individuals with BCR clonotypes", "%d of %d" % ((per_ind > 0).sum(), len(per_ind)), "has_vdj")
report("individuals without scBCR-seq data, and their PCs", "%d (%d PCs)" % (
    len(none), cells["patient_id"].isin(none).sum()), "no PC with a BCR clonotype")
report("PCs without a BCR clonotype outside these individuals", int((~vdj & ~cells["patient_id"].isin(none)).sum()),
       "has_vdj")

# ---- 3. clonal isotype of malignant BM PCs --------------------------------------------------------
HEAVY = {"IGHG": "IgG", "IGHA": "IgA", "IGHD": "IgD", "IGHM": "IgM", "IGHE": "IgE"}
LIGHT = {"IGKC": "Kappa", "IGLC": "Lambda"}


def clonal_isotype(x):
    """Per patient: heavy-chain class of most PCs ('none' if most lack a heavy-chain call) and majority light chain."""
    x = x[x["has_vdj"].astype(str) == "True"].assign(
        h=lambda d: d["heavy_c_gene"].astype(str).str[:4].map(HEAVY),
        l=lambda d: d["light_c_gene"].astype(str).str[:4].map(LIGHT))
    g = x.groupby("patient_id")
    heavy = g["h"].agg(lambda s: s.dropna().value_counts().index[0] if s.notna().mean() >= 0.5 else "none")
    light = g["l"].agg(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else "none")
    return heavy, light


hbm, _ = clonal_isotype(mal[mal["sample_type"] == "BM"])
vc = hbm.value_counts()
report("patients with malignant BM PCs covered by scBCR-seq", len(hbm), "isotype denominator")
for k in ["IgG", "IgA", "none", "IgD", "IgM"]:
    report("clonal isotype %s" % k, "%d (%.1f%%)" % (vc.get(k, 0), 100 * vc.get(k, 0) / len(hbm)),
           "heavy-chain class of most malignant BM PCs (none: most lack a heavy-chain call)")

# ---- 4. immunofixation concordance -------------------------------------------------------------------
ifx = pub.dropna(subset=["IFX"]).drop_duplicates("patient_id").set_index("patient_id")["IFX"].astype(str)
heavy, light = clonal_isotype(mal)
n = n_light = n_hcall = n_heavy = 0
for p in sorted(set(ifx.index) & set(heavy.index)):
    if ifx[p] == "Oligosecretory":
        continue
    comps = ifx[p].split(".")                           # M-protein(s) and free light chains, e.g. IgG_Kappa.FF_Kappa
    lights = {k.split("_")[-1] for k in comps if not k.startswith("FF_Ig")}
    heavies = {k.split("_")[0] for k in comps if k.startswith("Ig")}
    n += 1
    n_light += light[p] in lights
    if heavy[p] != "none":
        n_hcall += 1
        n_heavy += heavy[p] in heavies
report("immunofixation: light chain concordant", "%d of %d" % (n_light, n),
       "patients with immunofixation and malignant PCs covered by scBCR-seq, oligosecretory excluded")
report("immunofixation: heavy-chain class concordant", "%d of %d" % (n_heavy, n_hcall), "patients with a heavy-chain call")

# ---- 5. healthy donors --------------------------------------------------------------------------------
don = meta[meta["disease_status"] == "Donor"].drop_duplicates("tissue_id")
for (source, tissue), k in don.groupby(["NBM_Source", "sample_type"]).size().items():
    report("healthy-donor %s samples from %s" % (tissue, source), int(k), "NBM_Source of the cohort metadata")

# ---- 6. cytogenetic assays ------------------------------------------------------------------------------
report("patients with FISH results", cy.loc[cy["FISH"].notna() & (cy["FISH"] != "Unavailable"), "patient_id"].nunique(),
       "Supplementary Table 4, FISH not 'Unavailable'")
report("patients with WGS", cy.loc[cy["WGS"].notna() & (cy["WGS"] != "Unavailable"), "patient_id"].nunique(),
       "Supplementary Table 4, WGS not 'Unavailable'")

# ---- 7. cells used for signature detection ------------------------------------------------------------------
capped = cells.groupby(["patient_id", "sample_type", "annotation"]).size().clip(upper=1000)
hs = capped.groupby(["sample_type", "annotation"]).sum()
for t in ["BM", "PB"]:
    report("signature detection: %s malignant / normal PCs" % t, "%d / %d" % (hs[(t, "tumor")], hs[(t, "normal")]),
           "at most 1,000 per individual, tissue and cell type")
report("signature detection: PCs in total", int(hs.sum()), "same")

pd.DataFrame(rows).to_csv(OUT / "cell_bcr_isotype_counts.tsv", sep="\t", index=False)
FH.close()
