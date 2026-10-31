#!/usr/bin/env python3
"""Malignant PCs without detectable CNV: tumors without any CNV by cytogenetic group, stage and cell number, CNV-negative
cells within tumors with CNVs and their kNN neighborhoods, and CNV clones by stage (Results; Methods, Copy number variant
detection; Supplementary Figs. 1C, 2).

Purpose:      Malignant PCs (expanded BCR clonotype) of each patient with >= 30 of them, BM and PB pooled (a patient
              sampled at two stages takes the later stage). A cell is CNV-positive when Numbat gives p_cnv >= 0.9 for at
              least one CNV segment, otherwise CNV-negative; a tumor is CNV-positive when Numbat found any CNV segment.
              1. Tumors without any CNV by cytogenetic group, stage and cell number: two-sided Fisher's exact and
                 Wilcoxon rank-sum tests; logistic regression among IgH-translocated tumors (t(11;14), log10 cells
                 analyzed, log10 median UMIs, MGUS).
              2. CNV-negative cells of CNV-positive tumors: percentage by group (Kruskal-Wallis, rank-sum, Spearman);
                 UMIs, genes, PB share, top BCR clonotype, partner gene and kNN neighbors in the graphs of all PCs
                 (Fig. 1C) and of malignant PCs, against CNV-positive cells (paired two-sided Wilcoxon signed-rank tests,
                 tumors with >= 30 cells in each group).
              3. Tumors without CNV as neighbors of each other in the malignant-PC graph (rank-sum tests; linear
                 regressions adjusted for t(11;14), log10 malignant PCs, log10 median UMIs and MGUS).
              Also counts BM tumors (>= 100 malignant PCs) with a CNV clone by stage. UMAP rows in RandomState(42) order.

Inputs:       data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl: annotation, redcap_id, sample_type, disease_status,
              total_counts, n_genes, redcap_clonotype_rank, CCND1, CCND2, CCND3, NSD2, MAF, MAFA, MAFB.
              data/paper/anndata/plasma-cell-GEX5-umap.h5ad (Fig. 1C; obs patient_id, obsm X_umap, obsp nb_pca_distances)
              and tumor-cell-GEX5-umap.h5ad (malignant PCs; obsp nb_pca_distances).
              data/metadata/tumor-cohort-metadata-final-240325.csv: gex_id, rna_sample_id (cell names of the UMAP objects).
              data/cnv/plasma-cell-cell-level-cnv-prob-250306.pkl: p_cnv per cell and segment, tumor_name.
              data/cnv/numbat-input-table.tsv: patient_tumor, cells_used_for_numbat.
              data/paper/cnvs/cnv_segments_152_tumors.csv (patient_id, driver_arm_cnv) and clone-phylogeny-data.pkl
              (patient_id, disease_status, sample_type, GT); data/paper/cytogenetics-status.csv: patient_id,
              cytogenetics, hrd_status, FISH, WGS.

Outputs:      results/genomics/cnv_negative_tumors.tsv: one row per patient (Pt-###): stage, group, cells, CNV status.
              results/genomics/cnv_negative_cells_per_tumor.tsv: per CNV-positive tumor, CNV-negative vs positive cells.
              results/genomics/cnv_negative_tests.tsv: every test (name, detail, n, statistic, P).
              results/genomics/cnv_status_umap.csv.gz: Fig. 1C coordinates with CNV category (no identifiers).
              results/genomics/cnv_negative_cells_log.txt: counts and tests printed by the script.

Dependencies: Python 3.8 with requirements.txt (h5py, numpy, pandas, scipy, statsmodels); config.py, risk.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir
from risk import stages

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon, kruskal, spearmanr

OUT = result_dir("genomics")
FH = open(OUT / "cnv_negative_cells_log.txt", "w")
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
PARTNER = {"t(11;14)": "CCND1", "t(4;14)": "NSD2", "t(14;16)": "MAF", "t(14;20)": "MAFB",
           "t(6;14)": "CCND3", "t(8;14)": "MAFA", "t(12;14)": "CCND2"}
MIN_CELLS = 30
TESTS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n")


def add_test(name, detail, n, stat, p):
    TESTS.append(dict(test=name, detail=detail, n=n, statistic=stat, P=p))
    log("  TEST %s | %s | n=%s | stat=%s | P=%.3g" % (name, detail, n, stat, p))


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def read_obs(f, cols):
    """obs columns of an anndata >= 0.8 h5ad (categoricals stored as codes + categories)."""
    g = f["obs"]
    out = {}
    for c in cols:
        it = g[c]
        if isinstance(it, h5py.Group) and "codes" in it:
            cats, codes = np.array(dec(it["categories"][...]) + ["nan"]), it["codes"][...]
            out[c] = cats[np.where(codes >= 0, codes, len(cats) - 1)]
        else:
            out[c] = dec(it[...])
    return pd.DataFrame(out, index=dec(g[g.attrs.get("_index", "_index")][...]))


def read_knn(f, key="nb_pca_distances"):
    """kNN graph stored in obsp (CSR): returns indptr, indices."""
    g = f["obsp"][key]
    return g["indptr"][...], g["indices"][...]


def tumor_to_redcap(name):
    # Numbat tumor names use '-' for '/' in combined record IDs
    return name.replace("-", "/").split("_Tumor")[0]


# ---- 1. cells: malignant PCs, patients, cytogenetics -------------------------------------------
cols = ["annotation", "redcap_id", "sample_type", "disease_status", "total_counts", "n_genes",
        "redcap_clonotype_rank"] + sorted(set(PARTNER.values()))
ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))[cols]
for c in ["redcap_id", "annotation", "sample_type", "disease_status"]:
    ann[c] = ann[c].astype(str)
with h5py.File(str(data("paper", "anndata", "plasma-cell-GEX5-umap.h5ad")), "r") as f:
    obs = read_obs(f, ["patient_id"])
    umap = f["obsm"]["X_umap"][...]
    pc_indptr, pc_indices = read_knn(f)
    prm = f["uns"]["nb_pca"]["params"]
    log("Fig. 1C kNN graph parameters:", {k: prm[k][()] for k in prm.keys() if isinstance(prm[k], h5py.Dataset)})
# the UMAP objects name cells <barcode>-1-<rna_sample_id> (de-identified); the annotation table
# uses <barcode>-1-<gex_id>. Translate, then work in the cell order of the Fig. 1C object (the order
# of its kNN graph), keeping the annotation cell names as index.
meta = pd.read_csv(data("metadata", "tumor-cohort-metadata-final-240325.csv"))
gex2rna = meta.drop_duplicates("gex_id").set_index("gex_id")["rna_sample_id"]
gex = ann.index.str.split("-1-", n=1)
# 4 libraries (228 cells) have no gex_id in the metadata (their cells are filed under another
# library's rna_sample_id in the UMAP object): pair each with the sample sharing its barcodes among
# the cells still unmatched
h5name = pd.Index([b + "-1-" + gex2rna.get(s, "NA") for b, s in gex])
left_gex = set(s for _, s in gex) - set(gex2rna.index)
un = [x.split("-1-", 1) for x in obs.index[~obs.index.isin(h5name)]]
for g_ in left_gex:
    gbc = set(b for b, s in gex if s == g_)
    hits = pd.Series([s for b, s in un if b in gbc]).value_counts()
    gex2rna[g_] = hits.index[0]
log("libraries matched by barcode overlap:", len(left_gex))
h5name = pd.Index([b + "-1-" + gex2rna.get(s, "NA") for b, s in gex])
pos = h5name.get_indexer(obs.index)
assert (pos >= 0).all(), "%d Fig. 1C cells not matched" % (pos < 0).sum()
ann = ann.iloc[pos]
h5_to_ann = pd.Series(ann.index, index=obs.index)
ann["patient_id"] = obs["patient_id"].values
ann["disease_status"] = stages(ann["patient_id"], ann["sample_type"], ann["disease_status"])   # IMWG risk groups
red2pt = ann.loc[ann["annotation"] == "tumor", ["redcap_id", "patient_id"]].drop_duplicates()
assert red2pt["redcap_id"].is_unique
red2pt = red2pt.set_index("redcap_id")["patient_id"]

tum = ann[ann["annotation"] == "tumor"]
log("malignant PCs:", len(tum), "| patients with malignant PCs:", tum["patient_id"].nunique())
pt = pd.DataFrame({
    "n_malignant": tum.groupby("patient_id").size(),
    "n_malignant_BM": tum[tum["sample_type"] == "BM"].groupby("patient_id").size(),
    "median_umi": tum.groupby("patient_id")["total_counts"].median(),
    "median_genes": tum.groupby("patient_id")["n_genes"].median(),
    # patients sampled at two stages are given the later stage
    "stage": tum.groupby("patient_id")["disease_status"].agg(lambda s: max(s.unique(), key=STAGES.index)),
})
pt["n_malignant_BM"] = pt["n_malignant_BM"].fillna(0).astype(int)
pt["stage3"] = pt["stage"].replace({"LRSMM": "SMM", "IRSMM": "SMM", "HRSMM": "SMM"})
pt = pt[pt["n_malignant"] >= MIN_CELLS]
log("patients with >= 30 malignant PCs (Numbat cohort in the paper):", len(pt))

cyto = pd.read_csv(data("paper", "cytogenetics-status.csv"), index_col=0)
cyto = cyto[cyto.index.str.endswith("tumor:0")].set_index("patient_id")
pt = pt.join(cyto[["cytogenetics", "hrd_status", "FISH", "WGS"]])
log("patients without a cytogenetic call:", pt["cytogenetics"].isna().sum())
pt["group"] = np.select([pt["cytogenetics"] == "HRD", pt["cytogenetics"] == "t(11;14)",
                         pt["cytogenetics"] == "Other"],
                        ["HRD", "t(11;14)", "No translocation or HRD"], "Other IgH translocation")
pt["hrd_by_dna"] = (pt["cytogenetics"] == "HRD") & ((pt["FISH"] == "HRD") | (pt["WGS"] == "HRD"))

# ---- 2. Numbat: tumor and cell CNV status ------------------------------------------------------
nb = pd.read_csv(data("cnv", "numbat-input-table.tsv"), sep="\t")
nb["patient_id"] = nb["patient_tumor"].map(tumor_to_redcap).map(red2pt)
pt["cells_numbat"] = nb.groupby("patient_id")["cells_used_for_numbat"].sum()
log("Numbat input: %d tumors; patients in the cohort without a Numbat run: %d"
    % (len(nb), pt["cells_numbat"].isna().sum()))

segs = pd.read_csv(data("paper", "cnvs", "cnv_segments_152_tumors.csv"))
pt["cnv_positive"] = pt.index.isin(segs["patient_id"])
pt["n_segments"] = segs.groupby("patient_id").size()
pt["n_driver_arm_cnv"] = segs.dropna(subset=["driver_arm_cnv"]).groupby("patient_id")["driver_arm_cnv"].nunique()
pt[["n_segments", "n_driver_arm_cnv"]] = pt[["n_segments", "n_driver_arm_cnv"]].fillna(0).astype(int)

cp = pd.read_pickle(data("cnv", "plasma-cell-cell-level-cnv-prob-250306.pkl"))
cell = cp.groupby(level=0).agg(max_p=("p_cnv", "max"), tumor_name=("tumor_name", "first"))
del cp
cell["cnv_cell"] = np.where(cell["max_p"] >= 0.9, "CNV-positive", "CNV-negative")
cell["patient_id"] = cell["tumor_name"].map(tumor_to_redcap).map(red2pt)
cell = cell.join(tum[["sample_type", "total_counts", "n_genes", "redcap_clonotype_rank"]
                     + sorted(set(PARTNER.values()))], how="inner")
log("analyzed malignant cells in CNV-positive tumors:", len(cell), "| CNV-positive:",
    (cell["cnv_cell"] == "CNV-positive").sum(), "| CNV-negative:", (cell["cnv_cell"] == "CNV-negative").sum())
chk = cell.groupby("patient_id")["cnv_cell"].apply(lambda s: (s == "CNV-positive").any())
log("tumors with segments:", pt["cnv_positive"].sum(), "| with >= 1 cell p_cnv >= 0.9:", chk.sum(),
    "| agree:", set(chk[chk].index) == set(pt.index[pt["cnv_positive"]]))
cnt = cell.groupby(["patient_id", "cnv_cell"]).size().unstack(fill_value=0)
pt["cells_analyzed_cnvpos_tumor"] = cnt.sum(axis=1)
pt["pct_cnv_negative_cells"] = 100 * cnt["CNV-negative"] / cnt.sum(axis=1)

# CNV clones by stage (Supplementary Fig. 1C: BM samples with >= 100 malignant PCs, CNV clone of the phylogeny table)
cl = pd.read_pickle(data("paper", "cnvs", "clone-phylogeny-data.pkl"))
cl = cl[cl["GT"].notna() & (cl["sample_type"] == "BM")]
cl["disease_status"] = stages(cl["patient_id"], cl["sample_type"], cl["disease_status"])
bm = tum[tum["sample_type"] == "BM"].groupby(["disease_status", "patient_id"]).size()
bm = bm[bm >= 100].reset_index()
bm["clone"] = [(d, p) in set(zip(cl["disease_status"], cl["patient_id"])) for d, p in zip(bm["disease_status"], bm["patient_id"])]
bm["stage3"] = bm["disease_status"].replace({"LRSMM": "SMM", "IRSMM": "SMM", "HRSMM": "SMM"})
log("BM tumors with >= 100 malignant PCs and a CNV clone (clones >= 100 cells), with a clone / all, by stage:",
    bm.groupby("stage3")["clone"].agg(["sum", "count"]).to_dict("index"))
log("  patients with a BM CNV clone (any number of BM cells), by stage:",
    cl.drop_duplicates(["disease_status", "patient_id"])["disease_status"].value_counts().to_dict())
log("  of those without a CNV clone, Numbat found no CNV segment at all in",
    (~bm.loc[~bm["clone"], "patient_id"].map(pt["cnv_positive"]).fillna(False)).sum(), "of", (~bm["clone"]).sum())

# ---- 3. Tumor level: which tumors have no detectable CNV? ---------------------------------------
pt["cnv_negative"] = ~pt["cnv_positive"]
log("\nTUMOR LEVEL: tumors without any CNV: %d of %d" % (pt["cnv_negative"].sum(), len(pt)))
log(pd.crosstab(pt["group"], pt["cnv_negative"], margins=True).to_string())
log(pd.crosstab(pt["cytogenetics"], pt["cnv_negative"]).to_string())
log(pd.crosstab(pt["stage3"], pt["cnv_negative"], margins=True).to_string())
log("HRD tumors with HRD confirmed by FISH or WGS:", pt["hrd_by_dna"].sum(), "of", (pt["group"] == "HRD").sum(),
    "| CNV-negative among them:", (pt["hrd_by_dna"] & pt["cnv_negative"]).sum())

igh = pt[pt["group"].isin(["t(11;14)", "Other IgH translocation"])]
tab = pd.crosstab(igh["group"] == "t(11;14)", igh["cnv_negative"]).reindex(index=[True, False], columns=[True, False], fill_value=0)
orr, p = fisher_exact(tab.values)
add_test("Fisher two-sided", "CNV-negative tumors: t(11;14) %d/%d vs other IgH translocations %d/%d (OR %.2f)"
         % (tab.iloc[0, 0], tab.iloc[0].sum(), tab.iloc[1, 0], tab.iloc[1].sum(), orr), len(igh), round(orr, 3), p)
nonhrd = pt[pt["group"] != "HRD"]
tab = pd.crosstab(nonhrd["group"] == "t(11;14)", nonhrd["cnv_negative"]).reindex(index=[True, False], columns=[True, False], fill_value=0)
orr, p = fisher_exact(tab.values)
add_test("Fisher two-sided", "CNV-negative tumors: t(11;14) vs all other non-HRD tumors %d/%d (OR %.2f)"
         % (tab.iloc[1, 0], tab.iloc[1].sum(), orr), len(nonhrd), round(orr, 3), p)
tab = pd.crosstab(pt["stage3"] == "MGUS", pt["cnv_negative"]).reindex(index=[True, False], columns=[True, False], fill_value=0)
orr, p = fisher_exact(tab.values)
add_test("Fisher two-sided", "CNV-negative tumors: MGUS %d/%d vs SMM and MM %d/%d (OR %.2f)"
         % (tab.iloc[0, 0], tab.iloc[0].sum(), tab.iloc[1, 0], tab.iloc[1].sum(), orr), len(pt), round(orr, 3), p)
tab = pd.crosstab(pt["stage3"], pt["group"])
log("cytogenetic group by stage:\n" + tab.to_string())
for var, lab in [("cells_numbat", "malignant PCs analyzed by Numbat"), ("n_malignant", "malignant PCs"),
                 ("median_umi", "median UMIs per malignant PC")]:
    a, b = pt.loc[pt["cnv_negative"], var].dropna(), pt.loc[pt["cnv_positive"], var].dropna()
    st, p = mannwhitneyu(a, b, alternative="two-sided")
    add_test("Wilcoxon rank-sum two-sided", "%s: CNV-negative tumors median %.0f vs CNV-positive %.0f" % (lab, a.median(), b.median()),
             "%d vs %d" % (len(a), len(b)), st, p)
    a, b = igh.loc[igh["cnv_negative"], var].dropna(), igh.loc[igh["cnv_positive"], var].dropna()
    st, p = mannwhitneyu(a, b, alternative="two-sided")
    add_test("Wilcoxon rank-sum two-sided", "IgH-translocated only, %s: CNV-negative median %.0f vs CNV-positive %.0f" % (lab, a.median(), b.median()),
             "%d vs %d" % (len(a), len(b)), st, p)
t11 = igh[igh["group"] == "t(11;14)"]
a, b = t11.loc[t11["cnv_negative"], "cells_numbat"], t11.loc[t11["cnv_positive"], "cells_numbat"]
log("t(11;14) tumors: cells analyzed, CNV-negative median %.0f vs CNV-positive %.0f" % (a.median(), b.median()))

# sensitivity: logistic regression among IgH-translocated tumors
d = igh.dropna(subset=["cells_numbat"]).copy()
X = pd.DataFrame({"t(11;14)": (d["group"] == "t(11;14)").astype(float),
                  "log10 cells analyzed": np.log10(d["cells_numbat"]),
                  "log10 median UMIs": np.log10(d["median_umi"]),
                  "MGUS": (d["stage3"] == "MGUS").astype(float)})
try:
    fit = sm.Logit(d["cnv_negative"].astype(float), sm.add_constant(X)).fit(disp=0)
    ci = np.exp(fit.conf_int())
    for k in X.columns:
        add_test("Logistic regression (IgH-translocated tumors)", "OR %s %.2f (95%% CI %.2f-%.2f)"
                 % (k, np.exp(fit.params[k]), ci.loc[k, 0], ci.loc[k, 1]), len(d), round(np.exp(fit.params[k]), 3), fit.pvalues[k])
except Exception as e:  # complete separation is possible with few events
    log("logistic regression failed:", e)

# ---- 4. Cell level: CNV-negative cells inside CNV-positive tumors -------------------------------
log("\nCELL LEVEL (CNV-positive tumors)")
pos = pt[pt["cnv_positive"]]
log("per-tumor %% CNV-negative cells: median %.1f (IQR %.1f-%.1f); pooled %.1f%%"
    % (pos["pct_cnv_negative_cells"].median(), pos["pct_cnv_negative_cells"].quantile(.25),
       pos["pct_cnv_negative_cells"].quantile(.75), 100 * (cell["cnv_cell"] == "CNV-negative").mean()))
log(pos.groupby("group")["pct_cnv_negative_cells"].describe().round(1).to_string())
groups = [g["pct_cnv_negative_cells"].values for _, g in pos.groupby("group")]
st, p = kruskal(*groups)
add_test("Kruskal-Wallis", "% CNV-negative cells across the 4 cytogenetic groups", len(pos), st, p)
a, b = pos.loc[pos["group"] == "t(11;14)", "pct_cnv_negative_cells"], pos.loc[pos["group"] == "Other IgH translocation", "pct_cnv_negative_cells"]
st, p = mannwhitneyu(a, b, alternative="two-sided")
add_test("Wilcoxon rank-sum two-sided", "%% CNV-negative cells: t(11;14) median %.1f vs other IgH translocations %.1f" % (a.median(), b.median()),
         "%d vs %d" % (len(a), len(b)), st, p)
a, b = pos.loc[pos["group"] == "HRD", "pct_cnv_negative_cells"], pos.loc[pos["group"] != "HRD", "pct_cnv_negative_cells"]
st, p = mannwhitneyu(a, b, alternative="two-sided")
add_test("Wilcoxon rank-sum two-sided", "%% CNV-negative cells: HRD median %.1f vs non-HRD %.1f" % (a.median(), b.median()),
         "%d vs %d" % (len(a), len(b)), st, p)
r, p = spearmanr(pos["median_umi"], pos["pct_cnv_negative_cells"])
add_test("Spearman", "per tumor: median UMIs vs %% CNV-negative cells (rho %.2f)" % r, len(pos), round(r, 3), p)

# per tumor: CNV-negative vs CNV-positive cells (tumors with >= 30 cells in each group)
g = cell.groupby(["patient_id", "cnv_cell"])
per = pd.DataFrame({
    "n": g.size(),
    "median_umi": g["total_counts"].median(),
    "median_genes": g["n_genes"].median(),
    "pct_PB": g["sample_type"].apply(lambda s: 100 * (s == "PB").mean()),
    "pct_top_clonotype": g["redcap_clonotype_rank"].apply(lambda s: 100 * (s == 1).mean()),
}).unstack()
per.columns = ["%s_%s" % (a, "neg" if b == "CNV-negative" else "pos") for a, b in per.columns]
per = per.join(pos[["group", "cytogenetics", "stage", "pct_cnv_negative_cells"]])
for t, gene in PARTNER.items():
    ids = per.index[per["cytogenetics"] == t]
    sub = cell[cell["patient_id"].isin(ids)]
    m = sub.groupby(["patient_id", "cnv_cell"])[gene].mean().unstack()
    per.loc[ids, "partner_gene"] = gene
    per.loc[ids, "partner_expr_neg"] = m.get("CNV-negative")
    per.loc[ids, "partner_expr_pos"] = m.get("CNV-positive")
    nrm = ann[(ann["annotation"] == "normal") & ann["patient_id"].isin(ids)]
    nn = nrm.groupby("patient_id").size()
    per.loc[ids, "partner_expr_normal"] = nrm.groupby("patient_id")[gene].mean()[nn >= MIN_CELLS]
both = per[(per["n_neg"] >= MIN_CELLS) & (per["n_pos"] >= MIN_CELLS)]
log("CNV-positive tumors with >= 30 CNV-negative and >= 30 CNV-positive cells:", len(both), "of", len(per))
for var, lab in [("median_umi", "median UMIs"), ("median_genes", "median genes"), ("pct_PB", "% PB cells"),
                 ("pct_top_clonotype", "% cells with the tumor's top clonotype"), ("partner_expr", "translocation partner gene (mean log expression)")]:
    s = both.dropna(subset=[var + "_neg", var + "_pos"])
    if len(s) < 5:
        continue
    st, p = wilcoxon(s[var + "_neg"], s[var + "_pos"], alternative="two-sided")
    add_test("Wilcoxon signed-rank two-sided", "%s: CNV-negative median %.3g vs CNV-positive %.3g" % (lab, s[var + "_neg"].median(), s[var + "_pos"].median()),
             len(s), st, p)
s = both.dropna(subset=["partner_expr_neg", "partner_expr_normal"])
if len(s) >= 5:
    st, p = wilcoxon(s["partner_expr_neg"], s["partner_expr_normal"], alternative="two-sided")
    add_test("Wilcoxon signed-rank two-sided", "partner gene: CNV-negative malignant median %.2f vs same patient's normal PCs %.2f"
             % (s["partner_expr_neg"].median(), s["partner_expr_normal"].median()), len(s), st, p)

# ---- 5. Do CNV-negative cells cluster together? kNN neighborhoods ------------------------------
log("\nNEIGHBORHOODS")
cat = pd.Series("Normal PC", index=ann.index)
cat[ann["annotation"] == "tumor"] = "Malignant, not analyzed"
cat[ann["patient_id"].map(pt["cnv_negative"]).fillna(False).astype(bool) & (ann["annotation"] == "tumor")] = "Malignant, tumor without CNV"
cat[cell.index] = cell["cnv_cell"].map({"CNV-positive": "Malignant, CNV detected", "CNV-negative": "Malignant, no CNV detected (tumor with CNVs)"})
log(cat.value_counts().to_string())
code = {k: i for i, k in enumerate(["Normal PC", "Malignant, not analyzed", "Malignant, CNV detected",
                                    "Malignant, no CNV detected (tumor with CNVs)", "Malignant, tumor without CNV"])}
rng = np.random.RandomState(42)
order = rng.permutation(len(cat))
pd.DataFrame({"umap1": umap[order, 0].round(4), "umap2": umap[order, 1].round(4),
              "category": cat.values[order]}).to_csv(OUT / "cnv_status_umap.csv.gz", index=False)


def neighbor_stats(indptr, indices, labels, pt_codes, is_normal):
    """per cell: fraction of kNN neighbors that are normal PCs, from the same patient, and CNV-negative."""
    nn = np.diff(indptr)
    row = np.repeat(np.arange(len(nn)), nn)
    f = lambda w: np.bincount(row, weights=w.astype(float), minlength=len(nn)) / np.maximum(nn, 1)
    same = pt_codes[row] == pt_codes[indices]
    neg = labels[indices] == code["Malignant, no CNV detected (tumor with CNVs)"]
    negtum = labels[indices] == code["Malignant, tumor without CNV"]
    return pd.DataFrame({"frac_normal": f(is_normal[indices]), "frac_same_patient": f(same),
                         "frac_same_patient_cnvneg": f(same & neg),
                         "frac_other_patient_cnvneg": f(~same & (neg | negtum)),
                         "n_other_patient": np.bincount(row, weights=(~same).astype(float), minlength=len(nn)),
                         "n_other_patient_cnvneg_tumor": np.bincount(row, weights=(~same & negtum).astype(float), minlength=len(nn))})


# (a) Fig. 1C graph, all PCs: are CNV-negative malignant cells next to normal PCs or to their own tumor?
labels = cat.map(code).values
ptc = pd.factorize(ann["patient_id"])[0]
nbr = neighbor_stats(pc_indptr, pc_indices, labels, ptc, (ann["annotation"] == "normal").values)
nbr.index = ann.index
del pc_indptr, pc_indices
c = cell.join(nbr[["frac_normal", "frac_same_patient", "frac_other_patient_cnvneg"]])
m = c.groupby(["patient_id", "cnv_cell"])[["frac_normal", "frac_same_patient", "frac_other_patient_cnvneg"]].mean().unstack()
m.columns = ["%s_%s" % (a, "neg" if b == "CNV-negative" else "pos") for a, b in m.columns]
per = per.join(m)
both = per[(per["n_neg"] >= MIN_CELLS) & (per["n_pos"] >= MIN_CELLS)]
for var, lab in [("frac_normal", "Fig. 1C graph, fraction of neighbors that are normal PCs"),
                 ("frac_same_patient", "Fig. 1C graph, fraction of neighbors from the same patient"),
                 ("frac_other_patient_cnvneg", "Fig. 1C graph, fraction of neighbors that are CNV-negative malignant cells of other patients")]:
    st, p = wilcoxon(both[var + "_neg"], both[var + "_pos"], alternative="two-sided")
    add_test("Wilcoxon signed-rank two-sided", "%s: CNV-negative median %.3f vs CNV-positive %.3f"
             % (lab, both[var + "_neg"].median(), both[var + "_pos"].median()), len(both), st, p)

# (b) malignant-only graph: within its tumor, do CNV-negative cells sit next to each other?
with h5py.File(str(data("paper", "anndata", "tumor-cell-GEX5-umap.h5ad")), "r") as f:
    tidx = pd.Index(h5_to_ann.reindex(dec(f["obs"][f["obs"].attrs.get("_index", "_index")][...])).values)
    t_indptr, t_indices = read_knn(f)
assert tidx.notna().all()
lab_t = cat.reindex(tidx).map(code).values
pt_t = pd.factorize(ann.loc[tidx, "patient_id"])[0]
nbt = neighbor_stats(t_indptr, t_indices, lab_t, pt_t, np.zeros(len(tidx), bool))
nbt.index = tidx
del t_indptr, t_indices
c = cell.join(nbt[["frac_same_patient", "frac_same_patient_cnvneg"]], rsuffix="_t")
c["neg_share_of_same_patient_nbrs"] = c["frac_same_patient_cnvneg"] / c["frac_same_patient"].replace(0, np.nan)
m = c.groupby(["patient_id", "cnv_cell"])["neg_share_of_same_patient_nbrs"].mean().unstack()
per["negshare_nbrs_of_neg"] = m.get("CNV-negative")
per["negshare_nbrs_of_pos"] = m.get("CNV-positive")
per["negshare_expected"] = per["pct_cnv_negative_cells"] / 100
both = per[(per["n_neg"] >= MIN_CELLS) & (per["n_pos"] >= MIN_CELLS)].copy()
both["enrichment"] = both["negshare_nbrs_of_neg"] / both["negshare_expected"]
log("within-tumor co-location of CNV-negative cells: share of CNV-negative cells among same-tumor neighbors of "
    "CNV-negative cells / tumor share; median %.2f (IQR %.2f-%.2f)" % (both["enrichment"].median(),
    both["enrichment"].quantile(.25), both["enrichment"].quantile(.75)))
st, p = wilcoxon(both["negshare_nbrs_of_neg"], both["negshare_nbrs_of_pos"], alternative="two-sided")
add_test("Wilcoxon signed-rank two-sided", "malignant-only graph, share of CNV-negative cells among same-tumor neighbors: "
         "of CNV-negative cells median %.3f vs of CNV-positive cells %.3f" % (both["negshare_nbrs_of_neg"].median(),
         both["negshare_nbrs_of_pos"].median()), len(both), st, p)
per["enrichment_cnvneg_colocation"] = per["negshare_nbrs_of_neg"] / per["negshare_expected"]

# (c) across patients: do tumors without CNV sit next to each other?
nbt["patient_id"] = ann.loc[tidx, "patient_id"].values
agg = nbt.groupby("patient_id")[["frac_same_patient", "n_other_patient", "n_other_patient_cnvneg_tumor"]].agg(
    {"frac_same_patient": "mean", "n_other_patient": "sum", "n_other_patient_cnvneg_tumor": "sum"})
agg["share_cnvneg_tumor_among_other_patient_nbrs"] = agg["n_other_patient_cnvneg_tumor"] / agg["n_other_patient"].replace(0, np.nan)
pt = pt.join(agg[["frac_same_patient", "share_cnvneg_tumor_among_other_patient_nbrs"]].rename(
    columns={"frac_same_patient": "frac_nbrs_same_patient_malignant_graph"}))
log("share of all malignant cells that belong to tumors without CNV: %.3f" % ((cat == "Malignant, tumor without CNV").sum()
    / (ann["annotation"] == "tumor").sum()))
for sub, lab in [(pt, "all tumors"), (pt[pt["group"].isin(["t(11;14)", "Other IgH translocation"])], "IgH-translocated"),
                 (pt[pt["group"] == "t(11;14)"], "t(11;14)")]:
    for var in ["frac_nbrs_same_patient_malignant_graph", "share_cnvneg_tumor_among_other_patient_nbrs"]:
        a, b = sub.loc[sub["cnv_negative"], var].dropna(), sub.loc[sub["cnv_positive"], var].dropna()
        if len(a) < 3:
            continue
        st, p = mannwhitneyu(a, b, alternative="two-sided")
        add_test("Wilcoxon rank-sum two-sided", "%s, %s: tumors without CNV median %.3f vs with CNV %.3f" % (lab, var, a.median(), b.median()),
                 "%d vs %d" % (len(a), len(b)), st, p)
# the same, adjusted for t(11;14), tumor size, sequencing depth and MGUS (linear regression)
for sub, lab in [(pt, "all tumors"), (pt[pt["group"] != "HRD"], "non-HRD tumors")]:
    for var in ["frac_nbrs_same_patient_malignant_graph", "share_cnvneg_tumor_among_other_patient_nbrs"]:
        sub = sub.dropna(subset=[var])
        X = pd.DataFrame({"no CNV": sub["cnv_negative"].astype(float), "t(11;14)": (sub["group"] == "t(11;14)").astype(float),
                          "log10 malignant PCs": np.log10(sub["n_malignant"]), "log10 median UMIs": np.log10(sub["median_umi"]),
                          "MGUS": (sub["stage3"] == "MGUS").astype(float)})
        fit = sm.OLS(sub[var], sm.add_constant(X)).fit()
        for k in ["no CNV", "t(11;14)"]:
            add_test("Linear regression (%s)" % lab, "%s: coefficient of %s %.3f (95%% CI %.3f-%.3f), adjusted for the other covariates"
                     % (var, k, fit.params[k], fit.conf_int().loc[k, 0], fit.conf_int().loc[k, 1]), len(sub), round(fit.params[k], 4), fit.pvalues[k])

# ---- 6. save -----------------------------------------------------------------------------------
pt.index.name = "patient_id"
pt.to_csv(OUT / "cnv_negative_tumors.tsv", sep="\t")
per.index.name = "patient_id"
per.to_csv(OUT / "cnv_negative_cells_per_tumor.tsv", sep="\t")
pd.DataFrame(TESTS).to_csv(OUT / "cnv_negative_tests.tsv", sep="\t", index=False)
log("done")
