#!/usr/bin/env python3
"""Stage associations of the MST score and of the five progression-associated signatures adjusted for age and sex,
alone and together with the genetic alterations (Results).

Purpose:      Units and model of 02_stage_effects_genetics.py: the mean over the malignant BM PCs of one patient at one
              disease stage (at least 30 cells); outcome z-scored; ordinary least squares on stage coded 0-4 (MGUS,
              LRSMM, IRSMM, HRSMM, MM); standard errors clustered by patient; two-sided Wald tests. Models:
                unadjusted        outcome ~ stage
                age_sex           outcome ~ stage + age (per 10 years) + sex
                age_sex_genetics  outcome ~ stage + age + sex + the genetic covariates (translocation group, HRD,
                                  Numbat 1q gain, del(1p), del(13q), del(14q), del(17p), 8q24 gain, MYC expression)
              Age of a unit = mean age at sampling over its BM libraries. Age and sex are also compared between
              stages across the units (Kruskal-Wallis test, chi-square test, Spearman's correlation of age with stage).

Inputs:       results/stage_and_progression/bm_tumor_units.tsv (patient_id, disease_status, "MST Score",
              mean_expr:<signature>, translocation, HRD, 1q+, 1p-, 13q-, 14q-, 17p-, 8q+, MYC);
              data/paper/cohort-clinical-metadata.csv (patient_id, sample_type, disease_status, Age, Sex); SMM risk
              groups of the IMWG 2/20/20 model (risk.py).

Outputs:      results/stage_and_progression/stage_effects_demographics.tsv: stage coefficient per outcome and model, % of the unadjusted coefficient retained, age and sex coefficients.
              results/stage_and_progression/age_sex_by_stage_bm_tumors.tsv: age and sex by stage in the units, Kruskal-Wallis and chi-square P values.
              results/stage_and_progression/stage_effects_demographics_log.txt: units, tests and model summaries.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir
from risk import stages

import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = result_dir("stage_and_progression")
FH = open(os.path.join(OUT, "stage_effects_demographics_log.txt"), "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
OUTCOMES = {"MST Score": "MST score", "mean_expr:cell_cycling_G1S": "Cell cycling G1/S",
            "mean_expr:cell_cycling_G2M": "Cell cycling G2/M", "mean_expr:interferon_inducible": "Interferon inducible",
            "mean_expr:histone_chaperone_gene": "Histone/chaperone", "mean_expr:B_cell_activation": "B cell activation"}
# genetic covariates as in 02_stage_effects_genetics.py (bm_tumor_units.tsv holds the translocation groups)
TRANS = ["t(11;14)", "t(4;14)", "MAF translocation", "Other IgH translocation"]
CNVS = ["1q+", "1p-", "13q-", "14q-", "17p-", "8q+"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


def z(x):
    return (x - x.mean()) / x.std()


# ---- 1. units with age and sex ----------------------------------------------------------------------------------------
units = pd.read_csv(result("stage_and_progression", "bm_tumor_units.tsv"), sep="\t")
clin = pd.read_csv(data("paper", "cohort-clinical-metadata.csv"))
clin["disease_status"] = stages(clin["patient_id"], clin["sample_type"], clin["disease_status"])      # IMWG risk groups
bm = clin[clin["sample_type"] == "BM"]
demo = bm.groupby(["patient_id", "disease_status"]).agg(age=("Age", "mean"), sex=("Sex", "first")).reset_index()
units = units.merge(demo, on=["patient_id", "disease_status"], how="left", validate="one_to_one")
assert units[["age", "sex"]].notna().all().all(), "every unit needs age and sex"
units["stage"] = units["disease_status"].map(STAGES.index)
log("units:", len(units), "| patients:", units["patient_id"].nunique())

# ---- 2. are age and sex different between stages? ---------------------------------------------------------------------
tab = units.groupby("disease_status").agg(units=("age", "size"), age_median=("age", "median"),
                                          age_q1=("age", lambda a: a.quantile(0.25)), age_q3=("age", lambda a: a.quantile(0.75)),
                                          male=("sex", lambda s: (s == "M").sum())).reindex(STAGES)
tab["male_pct"] = 100 * tab["male"] / tab["units"]
p_age = stats.kruskal(*[g["age"] for _, g in units.groupby("disease_status")]).pvalue
p_sex = stats.chi2_contingency(pd.crosstab(units["disease_status"], units["sex"]))[1]
rho_age, p_rho_age = stats.spearmanr(units["stage"], units["age"])
tab.loc["test", ["age_median", "male_pct"]] = [p_age, p_sex]
tab.to_csv(os.path.join(OUT, "age_sex_by_stage_bm_tumors.tsv"), sep="\t")
log(tab.round(2).to_string())
log("age by stage: Kruskal-Wallis P = %.3g; Spearman rho with stage %.3f (P = %.3g) | sex by stage: chi-square P = %.3g"
    % (p_age, rho_age, p_rho_age, p_sex))

# ---- 3. stage models --------------------------------------------------------------------------------------------------
X_demo = pd.DataFrame({"age (per 10 y)": units["age"] / 10, "male": (units["sex"] == "M").astype(int)})
X_gen = pd.DataFrame({t: (units["translocation"] == t).astype(int) for t in TRANS})
X_gen["HRD"] = units["HRD"]
for ev in CNVS:
    X_gen[ev] = units[ev]
X_gen["MYC (z)"] = z(units["MYC"])
groups = units["patient_id"].astype("category").cat.codes.values
DESIGNS = {"unadjusted": [], "age_sex": [X_demo], "age_sex_genetics": [X_demo, X_gen]}


def fit(y, X):
    ok = y.notna() & X.notna().all(axis=1)
    return sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="cluster", cov_kwds={"groups": groups[ok.values]}), int(ok.sum())


rows = []
for col, lab in OUTCOMES.items():
    y = z(units[col])
    base = None
    for model, extra in DESIGNS.items():
        res, n = fit(y, pd.concat([units[["stage"]]] + extra, axis=1))
        base = res.params["stage"] if model == "unadjusted" else base
        lo, hi = res.conf_int().loc["stage"]
        row = dict(outcome=lab, model=model, n_units=n, coef_sd_per_stage=res.params["stage"], ci_low=lo, ci_high=hi,
                   P=res.pvalues["stage"], retained_pct=100 * res.params["stage"] / base)
        for c in ["age (per 10 y)", "male"]:
            if c in res.params:
                row[c + " coef"], row[c + " P"] = res.params[c], res.pvalues[c]
        rows.append(row)
        log("%-22s %-17s stage %.3f SD (%.3f to %.3f) P = %.2g, %.0f%% retained%s" % (
            lab, model, res.params["stage"], lo, hi, res.pvalues["stage"], row["retained_pct"],
            "" if model == "unadjusted" else "; age %.3f (P = %.2g), male %.3f (P = %.2g)" % (
                row["age (per 10 y) coef"], row["age (per 10 y) P"], row["male coef"], row["male P"])))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage_effects_demographics.tsv"), sep="\t", index=False)
log("done")
