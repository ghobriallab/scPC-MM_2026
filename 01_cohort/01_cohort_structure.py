#!/usr/bin/env python3
"""Individuals, samples and PCs by disease stage and tissue, sampling designs and the individuals or tumors kept by each
analysis (Extended Data Fig. 1A; Supplementary Table 2; Methods, Cohort structure and repeat samples).

Purpose:      Counts individuals, samples and plasma cells (PCs) passing QC by disease stage and tissue (an individual
              sampled at two stages is counted at both), classifies the sampling design of each individual (one BM
              sample, BM and PB at one time point, PB only, several time points), counts the per-patient analysis units
              (individual x stage x tissue, repeat samples pooled) under the stage labels of the publication table and of
              the cell table, and counts the individuals or tumors kept at each step of each analysis (flow diagram).
              Classifier eligibility (BM units with >= 500 malignant PCs; donors with >= 500 normal PCs) is checked
              against the saved training and test sample lists.
              Definitions: individual = patient or healthy donor (patient_id); sample = one BM aspirate or one PB draw
              (tissue_id), sequenced as 1-3 libraries (rna_sample_id) whose cells are combined; time point = sampling
              visit of the cohort metadata (timepoint), BM and PB samples of one time point are paired; stage = disease
              stage at sampling (disease_status; SMM risk groups of the IMWG 2/20/20 model, risk.py, in both tables).

Inputs:       data/paper/cohort-clinical-metadata.csv (publication table): patient_id, sample_type, tissue_id,
              rna_sample_id, disease_status, sample_date.
              data/metadata/tumor-cohort-metadata-final-240325.csv: rna_sample_id, catissue_id, timepoint.
              data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (cell table, one row per PC passing QC): catissue_id,
              redcap_id, sample_type, annotation (tumor = malignant, normal), disease_status.
              data/classifier/datasets/sample_info/train_samples.csv, test_samples.csv: redcap_id, disease_status,
              tumor, normal.

Outputs:      results/cohort/cohort_by_stage.tsv: per stage, individuals, samples and PCs by tissue.
              results/cohort/sampling_design.tsv: individuals per sampling design, by stage group.
              results/cohort/individuals.tsv: one row per individual (Pt-/HD- label only): samples, time points, design.
              results/cohort/analysis_units.tsv: per-patient analysis units (individual x stage x tissue) by tissue.
              results/cohort/analysis_use.tsv: tissue, unit, repeat samples and cell thresholds of each analysis.
              results/cohort/analysis_flow.tsv: individuals or tumors kept by each analysis (Extended Data Fig. 1A).
              results/cohort/supplementary_table1_cohort.xlsx: cohort by stage, sampling design and use in analyses.
              results/cohort/cohort_structure_log.txt: counts printed by the script.

Dependencies: Python 3.8 with requirements.txt (pandas, openpyxl); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages as imwg_stages

import pandas as pd

OUT = result_dir("cohort")
FH = open(OUT / "cohort_structure_log.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
SMM = ["LRSMM", "IRSMM", "HRSMM"]

# ---- inputs -------------------------------------------------------------------------------------
pub = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
cells = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
cells = cells[["gex_id", "catissue_id", "redcap_id", "sample_type", "annotation", "disease_status"]].astype(str)
pub["disease_status"] = imwg_stages(pub["patient_id"], pub["sample_type"], pub["disease_status"])
cells["patient_id"] = cells["redcap_id"].map(meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"])
assert cells["patient_id"].notna().all()
cells["disease_status"] = imwg_stages(cells["patient_id"], cells["sample_type"], cells["disease_status"])

# one row per library: publication table + time point and catissue_id (the cell table's sample key)
lib = pub[["patient_id", "sample_type", "tissue_id", "rna_sample_id", "disease_status", "sample_date"]].merge(
    meta[["rna_sample_id", "catissue_id", "timepoint"]], on="rna_sample_id", how="left", validate="1:1")
assert lib["catissue_id"].notna().all()

# PCs per sample (malignant / normal) from the cell table; a sample is catissue_id + tissue (two
# patients have their BM and PB samples under one catissue_id)
cells["sample_key"] = cells["catissue_id"] + ":" + cells["sample_type"]
lib["sample_key"] = lib["catissue_id"] + ":" + lib["sample_type"]
pcs = cells.groupby(["sample_key", "annotation"]).size().unstack(fill_value=0).rename(
    columns={"tumor": "malignant_PCs", "normal": "normal_PCs"})
smp = lib.groupby("tissue_id").agg(patient_id=("patient_id", "first"), sample_type=("sample_type", "first"),
                                   stage=("disease_status", "first"), timepoint=("timepoint", "first"),
                                   date=("sample_date", lambda d: pd.to_datetime(d, format="%m/%d/%y").min()),
                                   sample_key=("sample_key", "first"), libraries=("rna_sample_id", "nunique"))
assert (lib.groupby("tissue_id")[["patient_id", "sample_type", "disease_status", "timepoint", "sample_key"]].nunique() == 1).all().all()
assert smp["sample_key"].is_unique and set(pcs.index) <= set(smp["sample_key"])
smp = smp.join(pcs, on="sample_key").fillna({"malignant_PCs": 0, "normal_PCs": 0})
smp["PCs"] = smp["malignant_PCs"] + smp["normal_PCs"]
log("publication table: %d libraries, %d samples, %d individuals" % (len(lib), len(smp), smp["patient_id"].nunique()))
nocell = smp[smp["PCs"] == 0]
log("samples without any PC passing QC: %d (%s); individuals left without PCs: %d" % (
    len(nocell), ", ".join(nocell["sample_type"] + " " + nocell["stage"]),
    smp["patient_id"].nunique() - smp.loc[smp["PCs"] > 0, "patient_id"].nunique()))
smp = smp[smp["PCs"] > 0].copy()
log("with PCs: %d samples (BM %d, PB %d), %d libraries in the publication table, %d individuals, %d PCs (%d malignant, %d normal)" % (
    len(smp), (smp["sample_type"] == "BM").sum(), (smp["sample_type"] == "PB").sum(), smp["libraries"].sum(),
    smp["patient_id"].nunique(), smp["PCs"].sum(), smp["malignant_PCs"].sum(), smp["normal_PCs"].sum()))
log("libraries per sample:", smp["libraries"].value_counts().sort_index().to_dict())

# ---- stage labels of the cell table (used by the per-patient analyses; analysis units below) -------
cst = cells.groupby("sample_key")["disease_status"].agg(lambda x: "/".join(sorted(set(x))))
smp["stage_cell_table"] = smp["sample_key"].map(cst)

# ---- counts by stage (an individual sampled at two stages is counted at both) ----------------------
rows = []
for name, st in [(s, [s]) for s in STAGES[:2]] + [(s, [s]) for s in SMM] + [("SMM (any risk)", SMM), ("MM", ["MM"]), ("All", STAGES)]:
    x = smp[smp["stage"].isin(st)]
    bm, pb = x[x["sample_type"] == "BM"], x[x["sample_type"] == "PB"]
    rows.append({"stage": name, "individuals": x["patient_id"].nunique(),
                 "individuals_with_BM": bm["patient_id"].nunique(), "individuals_with_PB": pb["patient_id"].nunique(),
                 "individuals_with_BM_and_PB": len(set(bm["patient_id"]) & set(pb["patient_id"])),
                 "samples": len(x), "BM_samples": len(bm), "PB_samples": len(pb),
                 "malignant_PCs": int(x["malignant_PCs"].sum()), "normal_PCs": int(x["normal_PCs"].sum())})
bystage = pd.DataFrame(rows)
bystage.to_csv(OUT / "cohort_by_stage.tsv", sep="\t", index=False)
log("\nby stage:\n" + bystage.to_string(index=False))
log("sum of individuals over the six stages: %d (individuals sampled at two stages: %d)" % (
    bystage.set_index("stage").loc[STAGES, "individuals"].sum(), (smp.groupby("patient_id")["stage"].nunique() > 1).sum()))

# ---- sampling design per individual ----------------------------------------------------------------
ind = []
for pid, x in smp.groupby("patient_id"):
    tps = sorted(x["timepoint"].unique())
    per_tp = x.groupby("timepoint")["sample_type"].agg(lambda s: set(s))
    repeat = x.groupby(["timepoint", "sample_type"]).size().max() > 1          # >1 sample of one tissue at one visit
    paired = any(v == {"BM", "PB"} for v in per_tp)
    if len(tps) > 1:
        design = "several time points"
    elif paired:
        design = "BM and PB, one time point"
    else:
        design = "BM only, one time point" if set(x["sample_type"]) == {"BM"} else "PB only, one time point"
    stages = x.sort_values("timepoint")["stage"].drop_duplicates().tolist()
    visit = x.groupby("timepoint")["date"].min()                               # date of each visit
    ind.append({"patient_id": pid, "group": "Donor" if stages[0] == "Donor" else "Patient", "stages": "/".join(stages),
                "first_stage": stages[0], "samples": len(x), "BM_samples": int((x["sample_type"] == "BM").sum()),
                "PB_samples": int((x["sample_type"] == "PB").sum()), "time_points": len(tps),
                "paired_time_points": int(sum(v == {"BM", "PB"} for v in per_tp)), "repeat_samples_same_visit": bool(repeat),
                "months_first_to_last_visit": round((visit.max() - visit.min()).days / 30.44, 1),
                "design": design, "PCs": int(x["PCs"].sum())})
ind = pd.DataFrame(ind)
ind.to_csv(OUT / "individuals.tsv", sep="\t", index=False)
log("\nPCs per individual (all samples pooled): median %.0f, range %d-%d" % (ind["PCs"].median(), ind["PCs"].min(), ind["PCs"].max()))

grp = ind["first_stage"].where(~ind["first_stage"].isin(SMM), "SMM")
design = pd.crosstab(ind["design"], grp).reindex(columns=["Donor", "MGUS", "SMM", "MM"], fill_value=0)
design["All"] = design.sum(axis=1)
design.loc["All"] = design.sum()
design.to_csv(OUT / "sampling_design.tsv", sep="\t")
log("\nsampling design (individuals; SMM = first SMM stage):\n" + design.to_string())
log("individuals with a same-visit BM + PB pair: %d (%d pairs)" % ((ind["paired_time_points"] > 0).sum(), ind["paired_time_points"].sum()))
log("individuals with >1 sample of one tissue at one visit: %d" % ind["repeat_samples_same_visit"].sum())
lt = ind[ind["time_points"] > 1]
log("individuals with several time points: %d (time points: %s); stage changed in %d: %s" % (
    len(lt), lt["time_points"].value_counts().sort_index().to_dict(), (lt["stages"].str.contains("/")).sum(),
    ", ".join(lt.loc[lt["stages"].str.contains("/"), "patient_id"] + " " + lt.loc[lt["stages"].str.contains("/"), "stages"])))
log("months between first and last visit: median %.1f, range %.1f-%.1f" % (
    lt["months_first_to_last_visit"].median(), lt["months_first_to_last_visit"].min(), lt["months_first_to_last_visit"].max()))

# ---- units of the per-patient analyses: cells pooled per individual x stage x tissue ---------------
units = []
for lab, col in [("publication table", "stage"), ("cell table (as analyzed)", "stage_cell_table")]:
    u = smp.drop_duplicates(["patient_id", col, "sample_type"])
    for t in ["BM", "PB"]:
        v = u[u["sample_type"] == t]
        units.append({"stage_labels": lab, "tissue": t, "units": len(v), "individuals": v["patient_id"].nunique(),
                      **{s: int((v[col] == s).sum()) for s in STAGES}})
units = pd.DataFrame(units)
units.to_csv(OUT / "analysis_units.tsv", sep="\t", index=False)
log("\nper-patient analysis units (individual x stage x tissue; repeat samples pooled):\n" + units.to_string(index=False))

# ---- other analysis sets ---------------------------------------------------------------------------
# patients with malignant PCs (CNV calling and cytogenetic classification pool BM and PB per patient)
mal = smp[smp["malignant_PCs"] > 0]
log("\npatients with malignant PCs (any tissue): %d; with malignant PCs in BM: %d; in PB: %d" % (
    mal["patient_id"].nunique(), mal.loc[mal["sample_type"] == "BM", "patient_id"].nunique(),
    mal.loc[mal["sample_type"] == "PB", "patient_id"].nunique()))
# classifier training and test sets (saved sample lists; SMM listed but not used for training)
n_cls = {}
for name in ["train", "test"]:
    f = data("classifier", "datasets", "sample_info", "%s_samples.csv" % name)
    s = pd.read_csv(f)
    s = s[s["disease_status"].isin(["Donor", "MGUS", "MM"])]
    n_cls[name] = s["redcap_id"].nunique()
    log("classifier %s set: %d individuals %s, %d cells listed" % (
        name, s["redcap_id"].nunique(), s.groupby("disease_status")["redcap_id"].nunique().to_dict(),
        int(s[["tumor", "normal"]].sum().sum())))

# ---- how samples enter each analysis --------------------------------------------------------------
# Tissue, unit of analysis, handling of repeat samples and cell thresholds of each analysis (Methods); counts from
# this script.
u = units.set_index(["stage_labels", "tissue"])["units"]
lt3 = lt[lt["stages"].str.contains("/")]
use = pd.DataFrame([
    ["Cohort overview, tumor fraction", "Fig. 1B; Extended Data Fig. 1B", "BM and PB",
     "Individual (Fig. 1B); individual, stage and tissue (tumor fraction)", "All samples shown; cells pooled per tissue", "None",
     "%d individuals" % len(ind)],
    ["CNV inference and cytogenetic classification", "Extended Data Figs. 1C and 2", "BM and PB pooled",
     "Patient (main malignant clone)", "Malignant PCs of all samples of a patient pooled", "Clones of at least 100 cells (CNV)",
     "%d patients with malignant PCs" % mal["patient_id"].nunique()],
    ["Within-patient malignant vs normal PCs (therapeutic targets, stable DEGs)", "Fig. 2A; Extended Data Figs. 1 and 2", "BM",
     "Patient and stage", "Samples of the same stage pooled", "At least 30 malignant and 30 normal PCs", ""],
    ["Gene universe for signature discovery (pseudobulk DE)", "Methods", "BM and PB (BM vs PB only within patients with both)",
     "Patient, stage and tissue", "Samples of the same stage and tissue pooled", "At least 30 cells per pseudobulk", ""],
    ["Signature discovery (Hotspot)", "Fig. 3A,B", "BM + PB run and BM-only run",
     "Cells", "Pooled across samples, stages and visits", "At most 1,000 malignant and 1,000 normal PCs per individual and tissue", ""],
    ["Signature activity, malignancy and MST scores across stages", "Fig. 3C; Fig. 4D,E; Extended Data Figs. 3 and 4; Extended Data Fig. 8D",
     "BM and PB analyzed separately", "Individual, stage, tissue and cell type (mean of all PCs)",
     "Samples of the same stage and tissue pooled", "None",
     "%d BM and %d PB units" % (u[("cell table (as analyzed)", "BM")], u[("cell table (as analyzed)", "PB")])],
    ["Malignancy and MGUS-MM classifiers", "Fig. 4A-C; Extended Data Fig. 6", "BM",
     "Individual (each in the training or the test set only)", "One stage per individual; cells downsampled",
     "Patients with at least 500 malignant PCs; donors with at least 500 normal PCs", "%d training, %d test individuals" % (n_cls["train"], n_cls["test"])],
    ["Longitudinal examples", "Extended Data Fig. 8C; Supplementary Fig. 1A", "BM and PB",
     "Visit", "Each visit shown separately", "None", "%d patients sampled at two SMM risk levels" % len(lt3)],
    ["MST and genomic progression scores (WGS)", "Fig. 5E,F; Extended Data Fig. 9C,D", "BM",
     "Sample", "The BM sample analyzed by WGS", "None", "84 samples"],          # WGS-matched BM samples (Results)
], columns=["analysis", "figures", "tissue", "unit_of_analysis", "repeat_samples", "cell_thresholds", "n"])
use.to_csv(OUT / "analysis_use.tsv", sep="\t", index=False)

# ---- analysis flow: how many individuals or tumors each analysis kept (Extended Data Fig. 1A) ------
flow = []


def add(key, n, note=""):
    flow.append({"key": key, "n": int(n), "note": note})


# tissues per individual and stage (an individual counts once per stage; BM only + both + PB only = individuals)
for _, r in bystage.iterrows():
    both = r["individuals_with_BM_and_PB"]
    add("tissue_BM_only:" + r["stage"], r["individuals_with_BM"] - both)
    add("tissue_BM_and_PB:" + r["stage"], both)
    add("tissue_PB_only:" + r["stage"], r["individuals_with_PB"] - both)
add("SMM_at_two_risk_levels", bystage.set_index("stage").loc[SMM, "individuals"].sum()
    - bystage.set_index("stage").loc["SMM (any risk)", "individuals"])

# CNV inference and cytogenetics: malignant PCs of all samples of a patient pooled (Numbat needs >= 30)
pat = smp[smp["patient_id"].str.startswith("Pt-")]
mal_pt = pat.groupby("patient_id")["malignant_PCs"].sum()
add("patients", len(mal_pt))
add("patients_with_malignant_PCs", (mal_pt > 0).sum(), "min %d malignant PCs" % mal_pt[mal_pt > 0].min())
add("patients_with_at_least_30_malignant_PCs", (mal_pt >= 30).sum())
add("patients_without_malignant_PCs", (mal_pt == 0).sum(),
    str(ind.set_index("patient_id").loc[mal_pt[mal_pt == 0].index, "first_stage"].value_counts().to_dict()))

# within-patient malignant vs normal PCs (Fig. 2A): BM tumors = patient x stage units with malignant BM PCs
bmu = smp[smp["sample_type"] == "BM"].groupby(["patient_id", "stage"])[["malignant_PCs", "normal_PCs"]].sum().reset_index()
tum = bmu[bmu["malignant_PCs"] > 0]
both30 = tum[(tum["malignant_PCs"] >= 30) & (tum["normal_PCs"] >= 30)]
add("BM_tumors", len(tum), "%d patients" % tum["patient_id"].nunique())
add("BM_tumors_at_least_30_malignant", (tum["malignant_PCs"] >= 30).sum())
add("BM_tumors_at_least_30_malignant_and_normal", len(both30), "%d patients" % both30["patient_id"].nunique())
add("BM_tumors_fewer_than_30_normal", (tum["normal_PCs"] < 30).sum())

# classifiers: BM units of MGUS and MM patients with >= 500 malignant PCs and of donors with >= 500 normal PCs,
# then 75% training and 25% test (saved sample lists, checked against this rule)
lists = pd.concat([pd.read_csv(data("classifier", "datasets", "sample_info", "%s_samples.csv" % n)).assign(set=n)
                   for n in ["train", "test"]])
lists = lists[lists["disease_status"].isin(["Donor", "MGUS", "MM"])]
for s in ["Donor", "MGUS", "MM"]:
    x = bmu[bmu["stage"] == s]
    ok = x[(x["normal_PCs"] if s == "Donor" else x["malignant_PCs"]) >= 500]
    assert len(ok) == (lists["disease_status"] == s).sum()            # the rule reproduces the saved lists
    add("classifier_BM_units:" + s, len(x))
    add("classifier_eligible:" + s, len(ok))
    add("classifier_training:" + s, ((lists["disease_status"] == s) & (lists["set"] == "train")).sum())
    add("classifier_test:" + s, ((lists["disease_status"] == s) & (lists["set"] == "test")).sum())

# longitudinal samples
add("individuals_several_visits", len(lt))
add("SMM_patients_two_risk_levels", len(lt3))
flow = pd.DataFrame(flow)
flow.to_csv(OUT / "analysis_flow.tsv", sep="\t", index=False)
log("\nanalysis flow (Extended Data Fig. 1A):\n" + flow.to_string(index=False))

# ---- Supplementary Table 2, cohort sheet -----------------------------------------------------------
nice = {"stage": "Disease stage at sampling", "individuals": "Individuals", "individuals_with_BM": "Individuals with BM",
        "individuals_with_PB": "Individuals with PB", "individuals_with_BM_and_PB": "Individuals with BM and PB",
        "samples": "Samples", "BM_samples": "BM samples", "PB_samples": "PB samples",
        "malignant_PCs": "Malignant PCs", "normal_PCs": "Normal PCs"}
with pd.ExcelWriter(OUT / "supplementary_table1_cohort.xlsx") as xw:
    bystage.rename(columns=nice).replace({"LRSMM": "Low-risk SMM", "IRSMM": "Intermediate-risk SMM", "HRSMM": "High-risk SMM",
                                          "Donor": "Healthy donor", "All": "All individuals"}).to_excel(xw, sheet_name="Cohort by stage", index=False)
    design.rename_axis("Sampling design (individuals)").rename(columns={"Donor": "Healthy donors", "SMM": "SMM (first stage)"}).to_excel(
        xw, sheet_name="Sampling design")
    use.rename(columns=lambda c: c.replace("_", " ").capitalize()).to_excel(xw, sheet_name="Use in analyses", index=False)
log("\nwritten: analysis_use.tsv, supplementary_table1_cohort.xlsx")
FH.close()
