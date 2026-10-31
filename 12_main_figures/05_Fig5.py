#!/usr/bin/env python3
"""Figure 5: clinical relevance, external validation and genomic correlates of the MST score.

Purpose:      A, SMM patients by IMWG 2/20/20 risk group: mean MST score (top) and percentage of MM-like malignant PCs
              (bottom), progressors marked; progressed versus non-progressed (two-sided Wilcoxon rank-sum test), adjusted
              for risk group (likelihood-ratio test) and Spearman's correlation with risk group. B, MST score versus BM
              plasma cells, M-spike and FLC ratio (Spearman). C, MST scores in three published single-cell studies (AUROC
              of MGUS versus MM with 95% CI). D, CoMMpass MST scores and progression-free survival by score group (cut-offs
              0.75 and 0.80). E, MST score by WGS MM-like score. F, the 27 genomic alterations selected by LASSO. G,
              classifier features correlated with the LASSO residuals.
              Every printed statistic is read from a saved table, or recomputed and checked against it.

Inputs:       results/stage_and_progression: smm_progression_patient_scores.tsv, smm_progression_results.tsv,
              mm_like_cells_tumors.tsv, mm_like_cells_smm.tsv; results/classifiers: external_mst_units.tsv,
              external_mst_stats.tsv; results/figure_inputs: patient_id_mean_probability.csv.gz, wgs_samples.tsv,
              wgs_lasso_coef.tsv, wgs_lasso_carriers.tsv, wgs_residual_features_lasso.tsv; data/paper/
              cohort-clinical-metadata.csv; data/commpass: mm_classifier_results_250313.csv,
              MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv.

Outputs:      figures/Fig5.svg, Fig5.pdf; figures/source_data/Fig5_*.tsv (none for CoMMpass patient-level data).

Dependencies: Python 3.13 with requirements_figures.txt; figlib (figstyle, palettes, layout, plots, checks, paths).
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figlib"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]      # figlib modules, config.py
import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

import names
import palettes as P
import paths
import stage_tests as ST
import plots
from checks import figure_checks, file_checks, report
from figstyle import style, rich_text, p_runs, rho_runs, n_runs, join_runs, rng, SMALL, BASE, TITLE, MINUS
from layout import FigureLayout, Slot, new_figure, axes_mm, save

LAYOUT = FigureLayout("Fig5", [
    Slot("smm_progression", "A", 0, 0, 50, 82),
    Slot("clinical_metrics", "B", 52, 0, 131, 38),
    Slot("external_studies", "C", 52, 40, 131, 42),
    Slot("commpass_km", "D", 0, 84, 64, 50),
    Slot("wgs_score", "E", 66, 84, 44, 50),
    Slot("lasso", "F", 112, 84, 71, 84),
    Slot("residual_features", "G", 0, 136, 110, 32),
], max_height=170)

SMM = ["LRSMM", "IRSMM", "HRSMM"]
STAGES = ["MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]


def source(key, df):
    df.to_csv(paths.SOURCE_DATA + "/Fig5_%s.tsv" % key, sep="\t", index=False)


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol * max(1.0, abs(b)), (a, b)


# ---- A ---------------------------------------------------------------------------------------------------------------
def panel_smm_progression(sf):
    d = pd.read_csv(paths.result("stage_and_progression", "smm_progression_patient_scores.tsv"), sep="\t")
    t = pd.read_csv(paths.result("stage_and_progression", "mm_like_cells_tumors.tsv"), sep="\t")
    d = d.merge(t[["patient_id", "disease_status", "frac_mm_like"]], on=["patient_id", "disease_status"], how="left")
    d["pct"] = 100 * d["frac_mm_like"]
    assert len(d) == 115 and d["pct"].notna().all()
    S = pd.read_csv(paths.result("stage_and_progression", "mm_like_cells_smm.tsv"), sep="\t").set_index("group").loc["all SMM"]
    r1 = pd.read_csv(paths.result("stage_and_progression", "smm_progression_results.tsv"), sep="\t").iloc[0]
    close(S["mst_P_progressed_vs_not"], r1["wilcoxon_p"])
    close(S["mst_LRT_P_added_to_risk_group"], r1["lrt_p_adding_mst"])
    d = d.sort_values("patient_id").reset_index(drop=True)
    d["jit"] = rng("fig5-smm").uniform(-0.2, 0.2, len(d))           # one x per patient, shared by both rows
    d["group"] = d["disease_status"]
    rows = [("mst", "MST score", 2.0, (0, 1.0), np.arange(0, 1.01, 0.2), "mst_"),
            ("pct", "Malignant PCs with\nMST score > 0.5 (%)", 40.5, (-4, 100), np.arange(0, 101, 20), "")]
    key_lines = 3
    for col, ylab, y0, ylim, yt, pre in rows:
        ax = axes_mm(sf, 12.0, y0, 36.5, 31.0)
        tops = plots.box_strip(ax, d.assign(value=d[col]), SMM, P.TUMOR, "fig5-smm-" + col, jitter_col="jit",
                               highlight="progressed", n_labels="below" if col == "pct" else None)
        if col == "mst":
            plots.refline(ax, 0.5)
            ax.set_xticklabels([])
        ax.set_ylim(*ylim)
        plots.brackets(ax, [dict(i=0, j=2, color=P.BRACKET_STAGE,
                                 runs=join_runs([("Spearman\u2019s ", "")] + rho_runs(S[pre + "spearman_rho_risk_group"]),
                                                p_runs(S[pre + "spearman_P_risk_group"])))], tops, ymin=ylim[0],
                       key="sp" + col)
        # keys inside the frame, above the bracket: extend the axis by the height of the key block
        lo, hi = ax.get_ylim()
        h_pt = plots.axes_size_pt(ax)[1]
        need = key_lines * SMALL * 1.3 + 2.5
        ax.set_ylim(lo, lo + (hi - lo) / (1 - need / h_pt))
        ax.set_yticks(yt)
        plots.plain_ticks(ax.yaxis)
        ax.set_ylabel(ylab)
        plots.key(ax, 0.0, 1.0, [([("Progressed to MM (", ""), ("n", "i"), (" = %d)" % d["progressed"].sum(), "")],
                                   P.PROGRESSOR)], marker="D", ms=2.6, size=SMALL, row_pt=SMALL * 1.3,
                  transform=plots.pts(ax, 1.5, -1.0, ax.transAxes), gid="ax-prog" + col)
        for k, runs in enumerate([[("Progressed vs non-progressed, ", "")] + p_runs(S[pre + "P_progressed_vs_not"]),
                                  [("Adjusted for risk group, ", "")] + p_runs(S[pre + "LRT_P_added_to_risk_group"])]):
            rich_text(ax, 0, 1, runs, size=SMALL, va="top",
                      transform=plots.pts(ax, 1.5, -1.0 - (k + 1) * SMALL * 1.3 - 0.4, ax.transAxes), gid="ax-test%s%d" % (col, k))
    source("smm_progression", d[["patient_id", "disease_status", "progressed", "mst", "pct"]].rename(
        columns={"pct": "percent_mm_like_malignant_pcs"}))


# ---- B ---------------------------------------------------------------------------------------------------------------
def panel_clinical_metrics(sf):
    pp = ST.score_units()                                               # IMWG risk groups, pooled units (risk.py)
    md = pd.read_csv(paths.inputs("paper", "cohort-clinical-metadata.csv"))
    md["disease_status"] = ST.imwg_stages(md["patient_id"], md["sample_type"], md["disease_status"])
    tum = pp[pp["annotation"] == "tumor"].rename(columns={"patient_id": "sample"}).copy()   # one point per tumor unit
    tum["id"] = tum["sample_type"] + ":" + tum["disease_status"] + ":" + tum["sample"]
    tum = tum.set_index("id")
    meta = md[md["patient_id"].isin(tum["sample"])][["sample_type", "disease_status", "patient_id", "BMBx", "Mspike",
                                                     "FLC"]].drop_duplicates().copy()
    meta["id"] = meta["sample_type"] + ":" + meta["disease_status"] + ":" + meta["patient_id"]
    meta.loc[meta["BMBx"] == "<10", "BMBx"] = 10
    meta["BMBx"] = pd.to_numeric(meta["BMBx"])
    for col in ("BMBx", "Mspike", "FLC"):
        v = meta.loc[meta[col].notna(), ["id", col]].groupby("id")[col].mean()
        v = v[v.index.isin(tum.index)]
        tum.loc[v.index, col] = v.values
    expect = {"BMBx": (230, 0.6186, 1.12e-25), "Mspike": (195, 0.3856, 2.58e-08), "FLC": (235, 0.5202, 1.08e-17)}
    specs = [("BMBx", "BM plasma cells", "BM plasma cells (%)", False, 20, (0, 85), [0, 20, 40, 60, 80]),
             ("Mspike", "M-spike", "M-spike (g/dL)", False, 2, (0, 6.2), [0, 2, 4, 6]),
             ("FLC", "FLC ratio", "Involved/uninvolved FLC ratio", True, 20, None, None)]
    plots.key(sf, 5.0 / sf.w_mm, 1 - 0.4 / sf.h_mm, [(s, P.TUMOR[s]) for s in STAGES], ncol=5, size=SMALL, ms=2.6,
              gid="pn-stagekey")
    out = []
    for k, (col, title, xlab, logx, cut, xlim, xt) in enumerate(specs):
        x0 = 10 + 43 * k
        ax = axes_mm(sf, x0, 11.5, 30, 18)
        d = tum[tum[col].notna()]
        rho, p = stats.spearmanr(d[col], d["MST Score"])
        n, r_, p_ = expect[col]
        assert len(d) == n and abs(rho - r_) < 1e-4 and abs(p - p_) / p_ < 0.01, (col, len(d), rho, p)
        plots.scatter_groups(ax, d[col], d["MST Score"], d["disease_status"], P.TUMOR, "fig5-clin-" + col, s=2.0,
                             order=STAGES)
        if logx:                                                        # plain labels, no mathtext exponents
            ax.set_xscale("log")
            ax.set_xlim(0.2, 900)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(["1", "10", "100"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ln = ax.axvline(cut, color="black", lw=0.5, ls=(0, (1, 1.5)), zorder=1)
        ln.set_gid("ref-imwg")
        plots.refline(ax, 0.5)
        if xlim:
            pad = 0.03 * (xlim[1] - xlim[0])                            # dots clear of the frame
            ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
            ax.set_xticks(xt)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1.0])
        plots.plain_ticks(ax.yaxis)
        ax.set_ylabel("MST score" if k == 0 else "")
        if k:
            ax.set_yticklabels([])
        ax.set_xlabel(xlab)
        rich_text(sf, (x0 + 15) / sf.w_mm, 1 - 5.6 / sf.h_mm, [(title, "")], size=BASE, ha="center", gid="pn-t%d" % k)
        rich_text(sf, (x0 + 15) / sf.w_mm, 1 - 9.6 / sf.h_mm, join_runs([("Spearman\u2019s ", "")] + rho_runs(rho), p_runs(p)),
                  size=SMALL, ha="center", gid="pn-s%d" % k)
        out.append(d[["sample", "disease_status", "sample_type", col, "MST Score"]].assign(variable=col).rename(
            columns={col: "value", "sample": "patient_id", "MST Score": "mst"}))
    source("clinical_metrics", pd.concat(out))


# ---- C ---------------------------------------------------------------------------------------------------------------
def panel_external_studies(sf):
    units = pd.read_csv(paths.result("classifiers", "external_mst_units.tsv"), sep="\t")
    st = pd.read_csv(paths.result("classifiers", "external_mst_stats.tsv"), sep="\t").set_index(["study", "annotation"])
    stage3 = {"MGUS": "MGUS", "SMM": "IRSMM", "MM": "MM"}               # SMM is not split by risk in these studies
    colors = {"normal:" + s: P.NORMAL[stage3[s]] for s in stage3}
    colors.update({"tumor:" + s: P.TUMOR[stage3[s]] for s in stage3})
    order = ["normal:MGUS", "normal:SMM", "normal:MM", "tumor:MGUS", "tumor:SMM", "tumor:MM"]
    for k, study in enumerate(["Ledergor et al.", "Boiarsky et al.", "Dang et al."]):
        x0 = 9 + 43 * k
        ax = axes_mm(sf, x0, 7.5, 33, 20)
        u = units[units["study"] == study].copy()
        u["group"] = u["annotation"] + ":" + u["disease_status"]
        tops = plots.box_strip(ax, u.rename(columns={"mst": "value"}), order, colors, "fig5-ext-%d" % k, point=1.9,
                               labels=["MGUS", "SMM", "MM"] * 2, label_size=SMALL, n_labels="row")
        plots.refline(ax, 0.5)
        sep = ax.axvline(2.5, color="black", lw=0.5, ls=(0, (2, 1.5)))
        sep.set_gid("ref-sep")
        s = st.loc[(study, "tumor")]
        ax.set_ylim(0.1, 1.0)
        plots.brackets(ax, [dict(i=3, j=5, color=P.BRACKET_STAGE, lines=[
            [("AUROC %.2f" % s["auroc_MGUS_vs_MM"], "")],
            [("95%% CI %.2f–%.2f" % (s["auroc_lo_MGUS_vs_MM"], s["auroc_hi_MGUS_vs_MM"]), "")],
            p_runs(s["wilcoxon_p_MGUS_vs_MM"])])], {3: tops[3], 4: tops[4], 5: tops[5]}, ymin=0.1, key="auc%d" % k)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("MST score" if k == 0 else "")
        ax.set_title(study, fontsize=BASE, pad=2)
        for xc, lab in [(1, "Normal"), (4, "Malignant")]:
            rich_text(ax, xc, 0, [(lab, "")], size=SMALL, ha="center", va="top",
                      transform=plots.pts(ax, 0, -19.5, plots.blended_transform_factory(ax.transData, ax.transAxes)),
                      gid="pn-grp%d%s" % (k, lab))
    source("external_studies", units[["study", "sample_id", "disease_status", "annotation", "n_cells", "mst"]])


# ---- D ---------------------------------------------------------------------------------------------------------------
def panel_commpass_km(sf):
    cm = pd.read_csv(paths.inputs("commpass", "mm_classifier_results_250313.csv"))
    e = pd.read_csv(paths.inputs("commpass", "MMRF_CoMMpass_IA22_BM_1_expr_sig_score.tsv"), sep="\t")
    k = cm.merge(e[["PUBLIC_ID", "censpfs1", "ttcpfs1"]], on="PUBLIC_ID")
    score = k["exr_norm:MM_score"]
    lo, hi = 0.75, 0.80                                                  # cut-offs of the score groups
    k["group"] = np.where(score > hi, "High", np.where(score > lo, "Medium", "Low"))       # group = above the cut-off
    k["time"], k["event"] = k["ttcpfs1"] / 365, k["censpfs1"]                            # years = days / 365
    grp = ["Low", "Medium", "High"]
    assert len(k) == 754 and k["group"].value_counts().reindex(grp).tolist() == [354, 289, 111]
    rich_text(sf, 5.0 / sf.w_mm, 1 - 2.6 / sf.h_mm, [("CoMMpass (", ""), ("n", "i"), (" = %d)" % len(k), "")],
              size=BASE, gid="pn-title")
    ax = axes_mm(sf, 9.5, 6, 9, 29)
    ymin, ymax = 0.48, 0.95
    for g, a, b in [("Low", ymin, lo), ("Medium", lo, hi), ("High", hi, ymax)]:
        sp = ax.axhspan(a, b, color=P.LOW_MED_HIGH[g], alpha=0.22, lw=0, zorder=0)
        sp.set_gid("bg-band" + g)
    plots.violin(ax, score.values, color="white", key="fig5-cmp-violin", point=0.9)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_ylabel("MST score")
    ak = axes_mm(sf, 30, 6, 32, 25)
    tabs = plots.km(ak, k, grp, P.LOW_MED_HIGH)
    ak.set_xlim(0, 8.2)
    ak.set_xticks([0, 2, 4, 6, 8])
    ak.set_yticks([0, 0.5, 1.0])
    ak.set_xlabel("Years", labelpad=1)
    ak.set_ylabel("Progression-free\nsurvival")
    pv = plots.logrank_p(k)
    close(pv, 0.01506979464991648, 1e-6)
    rich_text(sf, 62 / sf.w_mm, 1 - 4.8 / sf.h_mm, [("Log-rank ", "")] + p_runs(pv), size=SMALL, ha="right",
              gid="pn-logrank")                                         # above the plot: the median lines stay clear
    plots.key(ak, 0.62, 0.99, [(g, P.LOW_MED_HIGH[g]) for g in grp], line=True, size=SMALL, ms=3.2, gid="ax-kmkey")
    risk = {g: [int(T[T[:, 0] >= t, 1][0]) if np.any(T[:, 0] >= t) else 0 for t in (0, 2, 4, 6, 8)] for g, T in tabs.items()}
    assert risk == {"Low": [354, 165, 101, 63, 11], "Medium": [289, 130, 69, 40, 6], "High": [111, 43, 21, 16, 1]}
    ar = axes_mm(sf, 30, 40, 32, 8.5, sharex=ak)
    plots.at_risk(ar, tabs, grp, P.LOW_MED_HIGH, [0, 2, 4, 6, 8])
    rich_text(sf, 9.5 / sf.w_mm, 1 - 38.6 / sf.h_mm, [("Patients at risk", "")], size=SMALL, gid="pn-atrisk")


# ---- E ---------------------------------------------------------------------------------------------------------------
def panel_wgs_score(sf):
    w = pd.read_csv(paths.cache("wgs_samples.tsv"), sep="\t")
    rho, p = stats.spearmanr(w["wgs_mm_like_score"], w["mst"])
    close(rho, 0.2087, 2e-4)
    w["group"] = w["wgs_mm_like_score"].astype(int).astype(str)
    order = [str(v) for v in range(int(w["wgs_mm_like_score"].min()), int(w["wgs_mm_like_score"].max()) + 1)]
    ax = axes_mm(sf, 9, 10, 22, 27)
    tops = plots.box_strip(ax, w.rename(columns={"mst": "value"}), order, {g: "white" for g in order}, "fig5-wgs",
                           box_face="white", point_col="disease_status", point_colors=P.TUMOR, point=2.0, width=0.65,
                           labels=[o.replace("-", MINUS) for o in order], n_labels="row")
    plots.refline(ax, 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("MST score")
    ax.set_xlabel("WGS MM-like score", labelpad=1)
    rich_text(sf, 20 / sf.w_mm, 1 - 3.2 / sf.h_mm, [("Patients with WGS (", ""), ("n", "i"), (" = %d)" % len(w), "")], size=BASE,
              ha="center", gid="pn-t")
    rich_text(sf, 20 / sf.w_mm, 1 - 7.0 / sf.h_mm, join_runs(rho_runs(rho), p_runs(p)), size=SMALL, ha="center",
              gid="pn-s")
    plots.key(sf, 33 / sf.w_mm, 1 - 12 / sf.h_mm, [(s, P.TUMOR[s]) for s in STAGES], size=SMALL, ms=2.6,
              gid="pn-stagekey")
    source("wgs_score", w[["patient_id", "disease_status", "wgs_mm_like_score", "mst"]])


# ---- F ---------------------------------------------------------------------------------------------------------------
def panel_lasso(sf):
    co = pd.read_csv(paths.cache("wgs_lasso_coef.tsv"), sep="\t")
    car = pd.read_csv(paths.cache("wgs_lasso_carriers.tsv"), sep="\t")
    assert len(co) == 27
    # display names: deletions with a minus sign; translocations gene first (the paper's naming)
    rename = {"t(11;14)(CCND1)": "CCND1 [t(11;14)]", "t(MYC;IgH/K/L)": "MYC [t(MYC;IgH/K/L)]"}
    show = lambda m: rename.get(m, m.replace("-", MINUS, 1) if m.startswith("-") else m)
    labels = [show(m) for m in co["mutation"]]
    car["mutation"] = [show(m) for m in car["mutation"]]
    rich_text(sf, 36 / sf.w_mm, 1 - 2.8 / sf.h_mm, [("LASSO coefficients (27 genomic alterations)", "")], size=BASE, ha="center",
              gid="pn-title")
    a1 = axes_mm(sf, 21, 6, 20, 64)
    plots.hbar(a1, labels, co["coefficient"].values, "#7F7F7F", height=0.72)
    a1.tick_params(axis="y", length=0, pad=6.5, labelsize=SMALL)
    a1.set_xlabel("Coefficient")
    a1.set_xlim(-0.15, 0.3)
    a1.set_xticks([-0.1, 0, 0.1, 0.2])
    a1.set_xticklabels([MINUS + "0.1", "0", "0.1", "0.2"])
    for y, flag in enumerate(co["in_wgs_mm_like_score"]):
        if flag:
            a1.plot([0], [y], marker="s", ms=2.2, color=P.BRACKET_MAL, ls="none", clip_on=False, zorder=5,
                    transform=plots.pts(a1, -3.6, 0, plots.blended_transform_factory(a1.transAxes, a1.transData)))
    a2 = axes_mm(sf, 43, 6, 14, 64, sharey=a1)
    car["group"] = car["mutation"]
    plots.box_strip(a2, car.rename(columns={"mst": "value"}), labels, {m: "white" for m in labels}, "fig5-lasso",
                    box_face="white", point_col="disease_status", point_colors=P.TUMOR, point=1.4, width=0.6,
                    horizontal=True, n_labels=None, edge=0.2)
    ln = a2.axvline(0.5, color=P.REFLINE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref-line")
    a2.set_xlim(0, 1)
    a2.set_xticks([0, 0.5, 1])
    a2.set_xticklabels(["0", "0.5", "1"])
    a2.tick_params(axis="y", left=False, labelleft=False)
    a2.set_xlabel("MST score")
    a3 = axes_mm(sf, 59, 6, 9, 64, sharey=a1)
    bars = a3.barh(range(len(co)), co["carriers"].values, height=0.72, color="#4D4D4D", zorder=2)
    for b in bars:
        b.set_gid("bar")
    for y, v in enumerate(co["carriers"].values):
        rich_text(a3, v, y, [("%d" % v, "")], size=SMALL, va="center", transform=plots.pts(a3, 1.0, 0, a3.transData),
                  gid="pn-cnt%d" % y)
    a3.tick_params(axis="y", left=False, labelleft=False)
    a3.set_xlim(0, 60)
    a3.set_xticks([0, 50])
    a3.set_xlabel("Patients")
    for sp in ("top", "right"):
        a3.spines[sp].set_visible(False)
    plots.key(sf, 3 / sf.w_mm, 1 - 77.5 / sf.h_mm, [("Mutation in the WGS MM-like score", P.BRACKET_MAL)], marker="s",
              ms=2.2, size=SMALL, gid="pn-mmkey")
    source("lasso", co)


# ---- G ---------------------------------------------------------------------------------------------------------------
def panel_residual_features(sf):
    r = pd.read_csv(paths.cache("wgs_residual_features_lasso.tsv"), sep="\t")
    r = r[r["fdr"] < 0.05].sort_values("rho").reset_index(drop=True)
    assert len(r) == 3
    rich_text(sf, 5.0 / sf.w_mm, 1 - 2.8 / sf.h_mm, [("Non-mutational features correlated with the LASSO residuals", "")],
              size=BASE, gid="pn-title")
    ax = axes_mm(sf, 34, 7, 54, 14)
    color = lambda f: P.SIGNATURE.get(f.split(":")[-1], "#808080")      # signatures as in Fig. 3A; CNV-like scores gray
    plots.hbar(ax, [names.feature(f) for f in r["feature"]], r["rho"].values, [color(f) for f in r["feature"]],
               height=0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.6, 0.4)
    ax.set_xticks([-0.5, -0.25, 0, 0.25])
    ax.set_xticklabels([MINUS + "0.5", MINUS + "0.25", "0", "0.25"])
    ax.set_xlabel("Spearman’s ρ with residuals")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    tr = plots.blended_transform_factory(ax.transAxes, ax.transData)
    for y, q in enumerate(r["fdr"].values):                      # exact q (Benjamini-Hochberg) of each correlation
        rich_text(ax, 1, y, p_runs(q, "q"), size=SMALL, va="center", transform=plots.pts(ax, 2.0, 0, tr), gid="pn-q%d" % y)
    source("residual_features", r)


PANELS = {"smm_progression": panel_smm_progression, "clinical_metrics": panel_clinical_metrics,
          "external_studies": panel_external_studies, "commpass_km": panel_commpass_km, "wgs_score": panel_wgs_score,
          "lasso": panel_lasso, "residual_features": panel_residual_features}


def main():
    with style():
        fig, pn = new_figure(LAYOUT)
        for key, fn in PANELS.items():
            fn(pn[key])
        issues = figure_checks(fig, LAYOUT)
        svg, pdf = save(fig, LAYOUT)
        fi, ntext = file_checks(svg, pdf, LAYOUT)
        print(report("Fig5", issues + fi, "SVG text elements: %d" % ntext))
        print("\n".join(issues + fi) or "ALL CHECKS PASS")


if __name__ == "__main__":
    main()
