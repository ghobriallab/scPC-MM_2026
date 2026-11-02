#!/usr/bin/env python3
"""Age, sex and self-reported race by disease stage, their comparisons between stages, and stage associations adjusted
for race (Supplementary Table 3; Methods).

Purpose:      Each individual with PCs passing quality control is counted once, in the stage group of their samples
              (healthy donor, MGUS, SMM or MM; the samples of each individual fall in one of these groups). SMM risk
              groups (IMWG 2/20/20) are compared separately; the three patients sampled at two risk levels are counted
              at each level in the table, and once, at the risk level of their first sample, in the tests. Age = age at
              the first sample. Race as recorded (White, Black, Asian, Hispanic, Other), tested as White vs other
              because the other categories are small. Tests: Kruskal-Wallis (age), chi-square (sex) and, for race, a
              chi-square statistic whose P value comes from 20,000 random permutations of the stage labels (seed 42).
              Comparisons among the three patient groups (MGUS, SMM, MM), among all four groups including donors, and
              among the three SMM risk groups. Sensitivity of the stage associations to race: units and model of
              03_stage_effects_demographics.py (malignant BM PCs of one patient at one stage, at least 30 cells; outcome
              z-scored; ordinary least squares on stage coded 0-4, standard errors clustered by patient), adjusted for
              age and sex, then for age, sex and race (White vs other); units without a recorded race are left out of
              both models.

Inputs:       data/paper/cohort-clinical-metadata.csv (patient_id, sample_type, disease_status, Age, Sex, Race,
              sample_date, n_tumor_cell, n_normal_cell); results/stage_and_progression/bm_tumor_units.tsv
              (patient_id, disease_status, "MST Score", mean_expr:<signature>); SMM risk groups of the IMWG
              2/20/20 model (risk.py).

Outputs:      results/stage_and_progression/demographics_by_stage.tsv: individuals, age (median, IQR, missing), sex and race by stage group.
              results/stage_and_progression/demographics_by_stage_tests.tsv: P values of the comparisons.
              results/stage_and_progression/stage_effects_race.tsv: stage coefficients adjusted for age and sex, without and with race.
              results/stage_and_progression/demographics_by_stage_log.txt: counts, tests and model summaries.

Dependencies: Python 3.8 with requirements.txt (numpy, pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir
from risk import stages

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "demographics_by_stage_log.txt"), "w")
GROUPS = ["Donor", "MGUS", "SMM", "MM"]
RISK = ["LRSMM", "IRSMM", "HRSMM"]
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
OUTCOMES = {"MST Score": "MST score", "mean_expr:cell_cycling_G1S": "Cell cycling G1/S",
            "mean_expr:cell_cycling_G2M": "Cell cycling G2/M", "mean_expr:interferon_inducible": "Interferon inducible",
            "mean_expr:histone_chaperone_gene": "Histone/chaperone", "mean_expr:B_cell_activation": "B cell activation"}
RNG = np.random.default_rng(42)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


def perm_chi2(labels, values, n=20000):
    """chi-square statistic of a contingency table, P value from permutations of the labels"""
    labels, values = np.asarray(labels), np.asarray(values)
    obs = stats.chi2_contingency(pd.crosstab(labels, values))[0]
    hits = sum(stats.chi2_contingency(pd.crosstab(RNG.permutation(labels), values))[0] >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


# ---- 1. one row per individual (libraries with PCs passing QC) ----------------------------------------------------------
clin = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
clin = clin[clin["n_tumor_cell"].fillna(0) + clin["n_normal_cell"].fillna(0) > 0]
clin["disease_status"] = stages(clin["patient_id"], clin["sample_type"], clin["disease_status"])      # IMWG risk groups
clin["group"] = clin["disease_status"].where(clin["disease_status"].isin(["Donor", "MGUS", "MM"]), "SMM")
ind = clin.groupby("patient_id").agg(group=("group", "first"), n_groups=("group", "nunique"), age=("Age", "min"),
                                     sex=("Sex", "first"), race=("Race", "first"))
assert (ind["n_groups"] == 1).all()
log("individuals:", len(ind), dict(ind["group"].value_counts().reindex(GROUPS)))
ind["white"] = ind["race"].map(lambda r: np.nan if pd.isna(r) else float(r == "White"))
risk = clin[clin["disease_status"].isin(RISK)].groupby(["patient_id", "disease_status"]).agg(
    age=("Age", "min"), sex=("Sex", "first"), race=("Race", "first"),
    first_date=("sample_date", lambda x: pd.to_datetime(x, format="%m/%d/%y").min())).reset_index()
risk["white"] = risk["race"].map(lambda r: np.nan if pd.isna(r) else float(r == "White"))

# ---- 2. table by stage group ------------------------------------------------------------------------------------------
rows = []
for name, d in [(g, ind[ind["group"] == g]) for g in GROUPS] + [(r, risk[risk["disease_status"] == r]) for r in RISK]:
    a = d["age"].dropna()
    row = {"group": name, "individuals": len(d), "age_median": a.median(), "age_q1": a.quantile(0.25),
           "age_q3": a.quantile(0.75), "age_missing": d["age"].isna().sum(), "male": (d["sex"] == "M").sum(),
           "female": (d["sex"] == "F").sum(), "sex_missing": d["sex"].isna().sum(),
           "male_pct_of_known": 100 * (d["sex"] == "M").sum() / d["sex"].notna().sum()}
    for r in ["White", "Black", "Asian", "Hispanic", "Other"]:
        row["race_" + r] = (d["race"] == r).sum()
    row["race_missing"] = d["race"].isna().sum()
    row["white_pct_of_known"] = 100 * d["white"].mean()
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "demographics_by_stage.tsv"), sep="\t", index=False)
log(tab.round(1).to_string(index=False))

# ---- 3. tests ---------------------------------------------------------------------------------------------------------
tests = []
for cmp, d, lab in [("patients: MGUS, SMM, MM", ind[ind["group"] != "Donor"], "group"),
                    ("all: donors, MGUS, SMM, MM", ind, "group"),
                    ("SMM risk groups: LR, IR, HR",            # each patient once, at the risk level of the first sample
                     risk.sort_values("first_date").drop_duplicates("patient_id", keep="first"), "disease_status")]:
    a = d.dropna(subset=["age"])
    s = d.dropna(subset=["sex"])
    w = d.dropna(subset=["white"])
    tests.append({"comparison": cmp,
                  "age_kruskal_P": stats.kruskal(*[g["age"] for _, g in a.groupby(lab)]).pvalue,
                  "sex_chi2_P": stats.chi2_contingency(pd.crosstab(s[lab], s["sex"]))[1],
                  "race_white_vs_other_perm_P": perm_chi2(w[lab], w["white"]),
                  "n_age": len(a), "n_sex": len(s), "n_race": len(w)})
tests = pd.DataFrame(tests)
tests.to_csv(os.path.join(OUT, "demographics_by_stage_tests.tsv"), sep="\t", index=False)
log("\n" + tests.to_string(index=False))
pt = ind[ind["group"] != "Donor"].dropna(subset=["age"])
rho, p = stats.spearmanr(pt["group"].map({"MGUS": 0, "SMM": 1, "MM": 2}), pt["age"])
log("age vs stage group among patients (MGUS 0, SMM 1, MM 2): Spearman rho %.3f, P = %.3g" % (rho, p))
d = ind[ind["group"] != "Donor"]
log("donors vs patients: age Wilcoxon rank-sum P = %.3g; sex chi-square P = %.3g" % (
    stats.mannwhitneyu(ind.loc[ind["group"] == "Donor", "age"].dropna(), d["age"].dropna()).pvalue,
    stats.chi2_contingency(pd.crosstab(ind["group"] == "Donor", ind["sex"]))[1]))

# ---- 4. stage associations with race as a further covariate (units and model of 03_stage_effects_demographics.py) ------
units = pd.read_csv(result("stage_and_progression", "bm_tumor_units.tsv"), sep="\t")
bm = clin[clin["sample_type"] == "BM"]
demo = bm.groupby(["patient_id", "disease_status"]).agg(age=("Age", "mean"), sex=("Sex", "first"),
                                                        race=("Race", "first")).reset_index()
units = units.merge(demo, on=["patient_id", "disease_status"], how="left", validate="one_to_one")
units["stage"] = units["disease_status"].map(STAGES.index)
units["age10"] = units["age"] / 10
units["male"] = (units["sex"] == "M").astype(float)
units["nonwhite"] = (units["race"] != "White").astype(float)
n_units = len(units)
units = units.dropna(subset=["age", "sex", "race"]).copy()
log("\nunits with age, sex and race: %d of %d (patients %d; non-White %d)" % (
    len(units), n_units, units["patient_id"].nunique(), int(units["nonwhite"].sum())))
out = []
for col, name in OUTCOMES.items():
    y = (units[col] - units[col].mean()) / units[col].std()
    res = {}
    for model, cov in [("age_sex", ["stage", "age10", "male"]), ("age_sex_race", ["stage", "age10", "male", "nonwhite"])]:
        X = sm.add_constant(units[cov])
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": units["patient_id"]})
        lo, hi = fit.conf_int().loc["stage"]
        res[model] = fit.params["stage"]
        out.append({"outcome": name, "model": model, "stage_coef_sd": fit.params["stage"], "ci_low": lo, "ci_high": hi,
                    "P": fit.pvalues["stage"], "nonwhite_coef": fit.params.get("nonwhite", np.nan),
                    "nonwhite_P": fit.pvalues.get("nonwhite", np.nan)})
    out[-1]["pct_of_age_sex"] = 100 * res["age_sex_race"] / res["age_sex"]
    log("%-22s stage %.3f SD (P = %.2g) -> with race %.3f SD (P = %.2g), %.0f%% retained; non-White %.3f (P = %.2g)" % (
        name, out[-2]["stage_coef_sd"], out[-2]["P"], out[-1]["stage_coef_sd"], out[-1]["P"], out[-1]["pct_of_age_sex"],
        out[-1]["nonwhite_coef"], out[-1]["nonwhite_P"]))
pd.DataFrame(out).to_csv(os.path.join(OUT, "stage_effects_race.tsv"), sep="\t", index=False)
FH.close()
