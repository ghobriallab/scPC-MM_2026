#!/usr/bin/env python3
"""IMS/IMWG 2025 high-risk classification of CoMMpass patients for the Cox models (Results; Supplementary Fig. 15A,B; Methods).

Purpose:      High risk if any of (Avet-Loiseau et al., J Clin Oncol 2025;43(24):2739-2751): C1 del(17p) at >20% cancer
              clonal fraction and/or TP53 mutation; C2 t(4;14), t(14;16) or t(14;20) together with 1q+ and/or del(1p32);
              C3 monoallelic del(1p32) with 1q+, or biallelic del(1p32); C4 beta2-microglobulin >= 5.5 mg/L with
              creatinine < 1.2 mg/dL (non-genomic). 1q gain and amplification both count as 1q+. IMWG2025 uses all four
              criteria; C1-C3 alone reproduce MMRF's IMWG_2024_HR of the Immune Atlas metadata (IMWG 2024 draft, no
              beta2-microglobulin criterion). Operationalization (100% concordance with MMRF's per-criterion flags in
              per_aliquot_metadata_MMRF_IA.csv, 298/298 definitive calls on baseline visits): del(17p), 1q+ and
              del(1p32) = SeqWGS_Cp_17p13_20percent, SeqWGS_Cp_1q21_20percent and SeqWGS_Cp_CDKN2C_20percent == 1
              (sequencing FISH-equivalent, >= 20% of cells; CDKN2C at 1p32.3); biallelic = del(1p32) and CDKN2C log2 <
              log2(0.4) = -1.3219 (< 0.8 copies on average; MMRF's definition in
              per_aliquot_metadata_MMRF_IA_dictionary.csv); TP53 mutation = any TP53 row of the combined canonical
              nonsynonymous variant table; t(4;14), t(14;16), t(14;20) = pairoscope NSD2, MAF and MAFB calls with an IgH
              partner (light-chain partners excluded). Exome seqFISH is used when WGS seqFISH is missing, and curated
              clinical FISH fills translocations when pairoscope is missing. One row per enrolled patient (1,143).
              Genomic calls come from the diagnosis bone marrow sample only (visit 1, or a bone marrow sample collected
              before first-line treatment started); patients without one are classified from laboratory values. Each
              abnormality is True, False or missing (not tested, never negative); IMWG2025 is high_risk, standard_risk,
              not_calculable or no_data.

Inputs:       data/commpass/ia24: subject_deid.csv (public_id); sample_deid.csv (public_id, sample_visit_id, specimen_type,
              days_to_collection); MMRF_CoMMpass_IA24_genome_gatk_cna_seqFISH.tsv and
              MMRF_CoMMpass_IA24_exome_gatk_cna_seqFISH.tsv (SAMPLE, <prefix>_17p13_20percent, _1q21_20percent,
              _CDKN2C_20percent, _CDKN2C); MMRF_CoMMpass_IA24_genome_tumor_only_mm_igtx_pairoscope.tsv (SAMPLE,
              NSD2/MAF/MAFB _CALL and _IGSOURCE); MMRF_CoMMpass_IA24_CuratedClinicalFISH.tsv (Visit_ID,
              FISH_Tx_414_Status, FISH_Tx_1416_Status, FISH_Tx_1420_Status);
              MMRF_CoMMpass_IA24_combined_vcfmerger2_All_Canonical_NS_Variants.tsv (SAMPLE, ANN[*].GENE); labs_deid.csv
              (public_id, baseline_lab, lab_test, lab_test_result, lab_test_result_unit).

Outputs:      results/commpass/imwg2025_risk.tsv: one row per patient: diagnosis visit, assays, each abnormality, the four
              criteria (C1-C4) and IMWG2025.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import math

import numpy as np
import pandas as pd

BIALLELIC_LOG2 = math.log2(0.4)   # -1.3219, MMRF definition of biallelic CDKN2C loss
B2M_CUT, CREAT_CUT = 5.5, 1.2

out = result_dir("commpass") / "imwg2025_risk.tsv"
P = lambda f: data("commpass", "ia24", f)


def visit_of(s):
    return s.str.extract(r"^(MMRF_\d+_\d+)", expand=False)


# ---- cohort and diagnosis bone marrow sample per patient --------------------------------------
# days_to_collection counts from the first day of first-line treatment (day 1 for every patient in
# administered_line_deid.csv). Diagnosis samples fall at day -105 to +111; the next-earliest
# samples of patients lacking one are at +115 or later (on treatment or at relapse), and Skerget et
# al. used none of them. Two patients with a blood-only visit 1 have a visit-2 bone marrow sample
# from before treatment, which counts.
TREATMENT_START_DAY = 1
cohort = pd.DataFrame({"patient": pd.read_csv(P("subject_deid.csv")).public_id.str.upper()})
s = pd.read_csv(P("sample_deid.csv"))
bm = s[s.specimen_type == "bone_marrow"].copy()
bm["visit"] = bm.sample_visit_id.str.upper()
bm["patient"] = bm.public_id.str.upper()
bm = bm[bm.visit.str.endswith("_1") | (bm.days_to_collection <= TREATMENT_START_DAY)]
base = (bm.sort_values("days_to_collection")
          .groupby("patient", as_index=False).head(1)[["patient", "visit", "days_to_collection"]])
base = cohort.merge(base, on="patient", how="left")

# ---- copy number: WGS seqFISH preferred, exome fallback ----------------------------------------
def seqfish(fname, pre, assay):
    t = pd.read_csv(P(fname), sep="\t")
    t = t[t.SAMPLE.str.contains("_BM_")].copy()
    t["visit"] = visit_of(t.SAMPLE)
    return pd.DataFrame({
        "visit": t.visit,
        "cn_sample": t.SAMPLE,
        "assay": assay,
        "del17p": t[f"{pre}_17p13_20percent"] == 1,
        "gain1q": t[f"{pre}_1q21_20percent"] == 1,
        "del1p32": t[f"{pre}_CDKN2C_20percent"] == 1,
        "cdkn2c_log2": t[f"{pre}_CDKN2C"],
    }).drop_duplicates("visit")

cw = seqfish("MMRF_CoMMpass_IA24_genome_gatk_cna_seqFISH.tsv", "SeqWGS_Cp", "WGS")
ce = seqfish("MMRF_CoMMpass_IA24_exome_gatk_cna_seqFISH.tsv", "SeqExome_Cp", "WXS")
cn = pd.concat([cw, ce[~ce.visit.isin(cw.visit)]], ignore_index=True)

# ---- translocations: pairoscope preferred, curated clinical FISH fallback ----------------------
po = pd.read_csv(P("MMRF_CoMMpass_IA24_genome_tumor_only_mm_igtx_pairoscope.tsv"), sep="\t")
po = po[po.SAMPLE.str.contains("_BM_")].copy()
po["visit"] = visit_of(po.SAMPLE)
# *_IGSOURCE is the immunoglobulin partner: 1=IgH (chr14), 2=IgK (chr2), 3=IgL (chr22).
# t(4;14), t(14;16), t(14;20) are IgH translocations by definition, so light-chain partners are
# excluded. This reproduces MMRF's curated nsd2_call/maf_call/mafb_call (376/376 each); counting
# any partner would disagree with them in 5 samples.
igh = lambda g: (po[f"{g}_CALL"] == 1) & (po[f"{g}_IGSOURCE"] == 1)
tx = pd.DataFrame({"visit": po.visit, "translocation_source": "pairoscope",
                   "t414": igh("NSD2"), "t1416": igh("MAF"),
                   "t1420": igh("MAFB")}).drop_duplicates("visit")
cf = pd.read_csv(P("MMRF_CoMMpass_IA24_CuratedClinicalFISH.tsv"), sep="\t")

def fish(col):
    v = cf[col].astype(str).str.strip().str.lower()
    return v.map({"positive": True, "negative": False, "1": True, "0": False})

cfx = pd.DataFrame({"visit": cf.Visit_ID.str.upper(), "translocation_source": "clinical_FISH",
                    "t414": fish("FISH_Tx_414_Status"), "t1416": fish("FISH_Tx_1416_Status"),
                    "t1420": fish("FISH_Tx_1420_Status")})
cfx = cfx[cfx[["t414", "t1416", "t1420"]].notna().any(axis=1)].drop_duplicates("visit")
tx = pd.concat([tx, cfx[~cfx.visit.isin(tx.visit)]], ignore_index=True)

# ---- TP53 mutation ------------------------------------------------------------------------------
mu = pd.read_csv(P("MMRF_CoMMpass_IA24_combined_vcfmerger2_All_Canonical_NS_Variants.tsv"),
                 sep="\t", low_memory=False, usecols=["SAMPLE", "ANN[*].GENE"])
mu = mu[mu.SAMPLE.str.contains("_BM_")]
mut_visits = set(visit_of(mu.SAMPLE).dropna())
tp53_visits = set(visit_of(mu.loc[mu["ANN[*].GENE"] == "TP53", "SAMPLE"]).dropna())

# ---- beta2M and creatinine at baseline ----------------------------------------------------------
lb = pd.read_csv(P("labs_deid.csv"), low_memory=False)
lb = lb[lb.baseline_lab == "yes"].copy()
lb["patient"] = lb.public_id.str.upper()
lb["v"] = pd.to_numeric(lb.lab_test_result, errors="coerce")

b = lb[lb.lab_test == "b2m"].copy()
b2m_factor = {"mg/l": 1, "mcg/ml": 1, "ng/ml": 1e-3, "mcg/l": 1e-3,
              "mg/dl": 10, "g/dl": 10}      # g/dl values (median 0.30) are read as mg/dL
b["b2m"] = b.v * b.lab_test_result_unit.map(b2m_factor)
b.loc[(b.lab_test_result_unit == "mg/l") & (b.v > 200), "b2m"] = b.v / 1000  # "mg/l" values above 200 read as ng/mL
b = b[b.b2m.between(0.3, 100)].groupby("patient", as_index=False).b2m.median()

c = lb[lb.lab_test == "serum_creatinine"].copy()
c["creat"] = np.where(c.lab_test_result_unit == "umol/l", c.v / 88.4,
                      np.where(c.lab_test_result_unit == "mg/dl", c.v, np.nan))
c = c[c.creat.between(0.1, 20)].groupby("patient", as_index=False).creat.median()

# ---- assemble -----------------------------------------------------------------------------------
# Patients without a diagnosis sample have visit = NaN; pandas would match NaN keys, so the
# genomic tables must not contain any.
assert cn.visit.notna().all() and tx.visit.notna().all()
x = (base.merge(cn, on="visit", how="left").merge(tx, on="visit", how="left")
         .merge(b, on="patient", how="left").merge(c, on="patient", how="left"))
x["has_cn"] = x.assay.notna()
x["has_tx"] = x.translocation_source.notna()
x["has_mut"] = x.visit.isin(mut_visits)

# Three-valued abnormality calls: True / False / <NA>. <NA> means "not tested", never "negative".
# pandas' nullable "boolean" dtype applies Kleene logic, so a criterion is only False when it is
# definitively negative, and a known negative component can still settle a criterion on its own
# (e.g. no 1q gain and no del(1p32) makes C2 False whatever the translocation status).
def tri(values, available):
    return values.astype("boolean").where(available, pd.NA)

x["del17p"] = tri(x.del17p, x.has_cn)
x["gain1q"] = tri(x.gain1q, x.has_cn)
x["del1p32"] = tri(x.del1p32, x.has_cn)
x["biallelic_1p32"] = tri(x.del1p32.fillna(False) & (x.cdkn2c_log2 < BIALLELIC_LOG2), x.has_cn)
x["tp53_mut"] = tri(x.visit.isin(tp53_visits), x.has_mut)
for col in ["t414", "t1416", "t1420"]:
    x[col] = tri(x[col], x.has_tx & x[col].notna())

x["C1_17p_TP53"] = x.del17p | x.tp53_mut
x["C2_IgHtx_chr1"] = (x.t414 | x.t1416 | x.t1420) & (x.gain1q | x.del1p32)
x["C3_1p32"] = (x.del1p32 & x.gain1q) | x.biallelic_1p32
# Each lab half is three-valued on its own: a known B2M < 5.5, or a known creatinine >= 1.2, rules
# C4 out even when the other value is missing.
x["C4_B2M"] = tri(x.b2m >= B2M_CUT, x.b2m.notna()) & tri(x.creat < CREAT_CUT, x.creat.notna())


def classify(pos, anydata):
    return np.where(pos.fillna(False), "high_risk",
           np.where((pos == False).fillna(False), "standard_risk",
           np.where(anydata, "not_calculable", "no_data")))

anydata = x.has_cn | x.has_tx | x.has_mut | x.b2m.notna() | x.creat.notna()
x["IMWG2025"] = classify(x.C1_17p_TP53 | x.C2_IgHtx_chr1 | x.C3_1p32 | x.C4_B2M, anydata)

cols = ["patient", "visit", "days_to_collection", "assay", "translocation_source", "has_mut",
        "del17p", "tp53_mut", "gain1q", "del1p32", "cdkn2c_log2", "biallelic_1p32",
        "t414", "t1416", "t1420", "b2m", "creat",
        "C1_17p_TP53", "C2_IgHtx_chr1", "C3_1p32", "C4_B2M",
        "IMWG2025"]
x[cols].to_csv(out, sep="\t", index=False)
print(f"wrote {out}: {len(x)} patients")
