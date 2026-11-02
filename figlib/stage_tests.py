"""Per-patient units and the stage tests drawn as brackets in the stage box plots (Figs. 3C, 4D,E;
Supplementary Figs. 3, 4 and 7A and Extended Data Fig. 8D).

Purpose:      A unit is the mean of the malignant or of the normal PCs of one patient, disease stage and tissue; SMM
              risk groups follow the IMWG 2/20/20 model (risk.py at the repository root). In each block (BM normal, BM malignant,
              PB normal, PB malignant) one bracket joins a reference stage (MM, MGUS or donors) to the stages that
              differ from it (two-sided Wilcoxon rank-sum tests, Benjamini-Hochberg within the block); one red bracket
              per tissue marks malignant versus normal PCs (stage-wise tests, Benjamini-Hochberg across stages). Each
              bracket carries the largest q of the comparisons it summarizes, and every q is checked against the table
              written by 09_stage_and_progression/01_stage_comparisons.py.

Inputs:       results/figure_inputs: per-sample-mean-expr-score.csv.gz, patient_id_mean_probability.csv.gz,
              pcs_cells.npz; data/h5/agg/plasma-cell-GEX5-annotation-250207.pkl (cells per unit);
              data/metadata/tumor-cohort-metadata-final-240325.csv (record -> patient label);
              results/stage_and_progression/stage_comparisons.tsv.

Outputs:      signature_units(), score_units(), tests(), brackets() and check_against_saved(), imported by the figure
              scripts.

Dependencies: numpy, pandas, scipy, statsmodels; paths.py, risk.py.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest as multi

import paths
from risk import CLINICAL_RISK as FIX, stages as imwg_stages

STAGES = ["Donor", "MGUS", "LRSMM", "IRSMM", "HRSMM", "MM"]
BLOCKS = [("BM", "normal"), ("BM", "tumor"), ("PB", "normal"), ("PB", "tumor")]
GROUPS = ["%s:%s:%s" % (s, t, a) for t, a in BLOCKS for s in (STAGES if a == "normal" else STAGES[1:])]
GENE = {"t(11;14)": "CCND1", "t(12;14)": "CCND2", "t(6;14)": "CCND3", "t(4;14)": "NSD2", "t(14;16)": "MAF", "t(8;14)": "MAFA",
        "t(14;20)": "MAFB", "HRD": "HRD", "Other": "Other", "None": "Other"}
CYTO = ["CCND1", "CCND2", "CCND3", "NSD2", "MAF", "MAFA", "MAFB", "HRD", "Other"]
_CACHE = {}


def _ncell():
    cells = pd.read_pickle(paths.inputs("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl"))
    cells = cells[["redcap_id", "sample_type", "annotation", "disease_status"]].astype(str)
    meta = pd.read_csv(paths.inputs("metadata", "tumor-cohort-metadata-final-240325.csv"))
    cells["patient_id"] = cells["redcap_id"].map(meta.drop_duplicates("redcap_id").set_index("redcap_id")["patient_id"])
    return cells.drop(columns="redcap_id")


def cells():
    if "cells" not in _CACHE:
        _CACHE["cells"] = _ncell()
    return _CACHE["cells"]


def _clinical_groups(df, patient_col):
    """IMWG 2/20/20 risk groups (risk.py); units that become one are pooled (cell-count weighted means)."""
    n = cells().groupby(["disease_status", "sample_type", "patient_id", "annotation"]).size()
    d = df.copy()
    d["n"] = [n.get((r["disease_status"], r["sample_type"], r[patient_col], r["annotation"]), np.nan) for _, r in d.iterrows()]
    d["disease_status"] = imwg_stages(d[patient_col], d["sample_type"], d["disease_status"])
    num = [c for c in d.columns if c.startswith(("mean_expr:", "auc:")) or c in ("Malignancy Score", "MST Score")]
    out = []
    for _, g in d.groupby(["disease_status", "sample_type", patient_col, "annotation"], sort=False):
        r = g.iloc[0].copy()
        if len(g) > 1:
            w = g["n"].astype(float)
            for c in num:
                r[c] = np.average(g[c].astype(float), weights=w)
            r["n"] = w.sum()
        out.append(r)
    d = pd.DataFrame(out)
    d["group"] = d["disease_status"] + ":" + d["sample_type"] + ":" + d["annotation"]
    d = d[d["group"].isin(GROUPS)].copy()
    d["cytogenetics"] = d["cytogenetics"].map(GENE)
    return d.rename(columns={patient_col: "patient_id"})


def majority_cytogenetics():
    """Cytogenetic group of each malignant unit = the most frequent call among its cells, with IMWG risk groups
    (pcs_cells.npz)."""
    if "cyto" not in _CACHE:
        z = np.load(paths.cache("pcs_cells.npz"))
        cat = lambda k: z[k + "__cats"].astype(str)[z[k + "__codes"]]
        c = pd.DataFrame({k: cat(k) for k in ("patient_id", "sample_type", "annotation", "disease_status", "cytogenetics")})
        c = c[c["annotation"] == "tumor"]
        c["disease_status"] = imwg_stages(c["patient_id"], c["sample_type"], c["disease_status"])
        m = c.groupby(["disease_status", "sample_type", "patient_id"])["cytogenetics"].agg(lambda s: s.value_counts().index[0])
        _CACHE["cyto"] = m.map(GENE)
    return _CACHE["cyto"]


def _cyto_by_majority(d, table):
    """Every malignant unit takes the majority call of its cells, so that all figures show the same groups."""
    m = majority_cytogenetics()
    t = d["annotation"] == "tumor"
    keys = list(zip(d.loc[t, "disease_status"], d.loc[t, "sample_type"], d.loc[t, "patient_id"]))
    old = list(d.loc[t, "cytogenetics"])
    # units without malignant cells in the cell table (10 PB units of the signature table, all without values) keep theirs
    new = [m[k] if (k in m.index and isinstance(o, str)) else o for k, o in zip(keys, old)]
    changed = [k for k, o, n in zip(keys, old, new) if isinstance(o, str) and o != n]
    assert changed in ([], [("LRSMM", "PB", "Pt-129")]), (table, changed)
    d.loc[t, "cytogenetics"] = new
    return d


def signature_units():
    if "sig" not in _CACHE:
        s = pd.read_csv(paths.cache("per-sample-mean-expr-score.csv.gz"), index_col=0)
        _CACHE["sig"] = _cyto_by_majority(_clinical_groups(s, "patient_id"), "signature units")
    return _CACHE["sig"]


def score_units():
    if "prob" not in _CACHE:
        p = pd.read_csv(paths.cache("patient_id_mean_probability.csv.gz"), index_col=0)
        _CACHE["prob"] = _cyto_by_majority(_clinical_groups(p.rename(columns={"sample": "patient_id"}), "patient_id"), "score units")
    return _CACHE["prob"]


# ---- tests -------------------------------------------------------------------------------------------------------
def _cross(df, col):
    cats = list(df["disease_status"].astype(str).unique())
    rows = [("%s,%s" % (x, y), scipy.stats.ranksums(df.loc[df["disease_status"] == x, col],
                                                    df.loc[df["disease_status"] == y, col])[1]) for x, y in combinations(cats, 2)]
    res = pd.DataFrame(rows, columns=["comparison", "p"])
    res["fdr"] = multi.fdrcorrection(res["p"], alpha=0.05, method="indep", is_sorted=False)[1] if len(res) else []
    return res


def _tvn(normal, tumor, col):
    cats = sorted(set(normal["disease_status"].astype(str)) & set(tumor["disease_status"].astype(str)), key=STAGES.index)
    p = [scipy.stats.ranksums(normal.loc[normal["disease_status"] == c, col], tumor.loc[tumor["disease_status"] == c, col])[1]
         for c in cats]
    return pd.DataFrame({"comparison": cats, "p": p, "fdr": multi.fdrcorrection(p, alpha=0.05, method="indep", is_sorted=False)[1]})


def tests(units, col):
    df = units[units[col].notna()]
    sub = lambda t, a: df[(df["sample_type"] == t) & (df["annotation"] == a)]
    out = {"%s:%s" % (t, a): _cross(sub(t, a), col) for t, a in BLOCKS}
    out["BM:TvN"] = _tvn(sub("BM", "normal"), sub("BM", "tumor"), col)
    out["PB:TvN"] = _tvn(sub("PB", "normal"), sub("PB", "tumor"), col)
    return out


def check_against_saved(panel, t):
    """Every q value must equal the one saved by 09_stage_and_progression/01_stage_comparisons.py."""
    ref = pd.read_csv(paths.result("stage_and_progression", "stage_comparisons.tsv"), sep="\t")
    ref = ref[ref["panel"] == panel]
    assert len(ref), panel
    n = 0
    for key, df in t.items():
        r = ref[ref["test"] == key].set_index("comparison")["fdr"]
        for comp, q in zip(df["comparison"], df["fdr"]):
            assert abs(r[comp] - q) <= 1e-9 * max(1, abs(q)), (panel, key, comp, r[comp], q)
            n += 1
    return n


# ---- bracket rules -----------------------------------------------------------------------------------------------
def _anchor(df, cutoff):
    """The reference stage and the stages significantly different from it."""
    if df.empty:
        return None
    X = df[df["fdr"] < cutoff]
    if X.empty:
        return None
    counts = pd.Series(np.array([c.split(",") for c in X["comparison"]]).reshape(-1)).value_counts()
    maxg = counts[counts == counts.max()].index
    if len(maxg.intersection({"MM", "MGUS", "Donor"})) == 0:
        return None
    anchor, partner = "MM", "MGUS"
    if "MM" not in maxg:
        anchor = "MGUS" if "MGUS" in maxg else "Donor"
        partner = "MM"
    order = {s: k for k, s in enumerate(STAGES)}
    rows = [(anchor, -1.0)]
    for _, r in X[[anchor in c.split(",") for c in X["comparison"]]].iterrows():
        g = [x for x in r["comparison"].split(",") if x != anchor][0]
        rows.append((g, r["fdr"]))
    f = pd.DataFrame(rows, columns=["group", "fdr"])
    f["order"] = f["group"].map(order)
    f = f.sort_values("order").reset_index(drop=True)
    f["block"] = (np.abs(f["order"].diff(1)) == 1) | (np.abs(f["order"].diff(-1)) == 1)
    f.loc[f["group"] == anchor, "block"] = True
    if f["block"].sum() > 2:
        f = f[f["block"] & (f["fdr"] != -1)]
    elif (f["group"] == partner).sum() == 1:
        f = f[f["group"].isin([anchor, partner]) & (f["fdr"] != -1)]
    else:
        return None
    return anchor, list(f["group"]), float(f["fdr"].max())


def brackets(units, col, t, cutoff, tvn_rule, score_rule=False):
    """Bracket specs in x positions of GROUPS: dicts {kind, anchor_x, partner_xs, q}."""
    pos = {g: k for k, g in enumerate(GROUPS)}
    out = []
    for t_, a in BLOCKS:
        key = "%s:%s" % (t_, a)
        if score_rule:                         # scores: only blocks with > 2 stages above 0.5
            b = units[(units["sample_type"] == t_) & (units["annotation"] == a)]
            if b.loc[b[col] > 0.5, "group"].nunique() <= 2:
                continue
        r = _anchor(t[key], cutoff)
        if r is None:
            continue
        anchor, partners, q = r
        out.append(dict(kind="stage", block=key, anchor_x=pos["%s:%s" % (anchor, key)],
                        partner_xs=[pos["%s:%s" % (p, key)] for p in partners], q=q, n=len(partners)))
    for tissue in ("BM", "PB"):
        f = t[tissue + ":TvN"]["fdr"].values
        if tvn_rule == "signature":            # signatures: max of the significant q if all but one are significant
            fin = f[f < 0.05].max() if (f < 0.05).sum() >= len(f) - 1 else 1.0
        else:                                  # scores: max of all q if all but one are below 0.05
            fin = f.max() if (f < 0.05).sum() >= len(f) - 1 else 1.0
        if fin < cutoff:
            nx = [pos[g] for g in GROUPS if g.endswith(tissue + ":normal")]
            tx = [pos[g] for g in GROUPS if g.endswith(tissue + ":tumor")]
            out.append(dict(kind="tvn", block=tissue, anchor_x=float(np.mean(nx)), partner_xs=[float(np.mean(tx))], q=float(fin),
                            n=len(f), span_i=(min(nx), max(nx)), span_j=(min(tx), max(tx))))
    return out
