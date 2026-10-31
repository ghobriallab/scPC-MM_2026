#!/usr/bin/env python3
"""Cell-level, de-identified arrays for the UMAP panels, from the processed objects (large inputs).

Purpose:      Every output keeps the cell order of its embedding and only what the panels draw: UMAP coordinates, the
              categories the cells are colored by (Pt-### / HD-## labels, disease stage, tissue, PC status, cytogenetic
              call), per-cell scores and the expression of the genes shown. Cell barcodes and library, sample and record
              IDs are never written. Parts (run all, or name some: python3 02_extract_cells.py pcs tumors):
                pcs           all PCs: UMAP, categories, malignancy and MST scores (Figs. 1C, 4D,E; Supplementary Figs. 2C,
                              5, 8, 11A)
                targets       log-normalized expression of the 16 therapeutic targets in all PCs (Supplementary Fig. 5A)
                tumors        malignant PCs: UMAP, categories and scaled expression of the nine translocation partner genes
                              (Fig. 2C-G; Supplementary Figs. 4C,D, 6)
                tumors_cnv    1q and 13q Numbat calls, call probabilities and CNV-like scores of the malignant PCs, the
                              scores quantile-transformed to a normal distribution (random_state 42) (Supplementary Fig. 11B)
                longitudinal  PCs of the three SMM patients sampled at two risk levels (Supplementary Figs. 9A,
                              14C)
                external      PCs of three external studies with the per-cell MST score of the final model (Supplementary
                              Fig. 14E); four samples of one study are excluded at the request of its authors
                signatures    per-cell activity of the 16 signatures (Supplementary Figs. 8, 9A, 14C)
                senescence    per-cell SenMayo senescence score (Supplementary Fig. 11A)
              The large per-cell tables are loaded one at a time (about 50 GB of memory).

Inputs:       data/paper: anndata/ (plasma-cell-GEX5-umap.h5ad, tumor-cell-GEX5-umap.h5ad, longitudinal-samples-umap.h5ad),
              classifier/ (cell-level-prediction-knn30.pkl, patient_id_mean_probability.pkl),
              signatures/cell-level-module-gene-expr-auc-final.pkl; data/h5/agg: plasma-cell-GEX5-log1p-250207.h5ad,
              plasma-cell-GEX5-annotation-250207.pkl; data/h5/external/<study>/scaled_umap.h5ad; data/signature: score/
              (cell-level-copy-number-full-gene-expr.pkl, cell-level-senescence-expr-auc.pkl), external/
              <study>_features-knn30.pkl; data/cnv/plasma-cell-cnv-cytoband-matrix-250109.pkl; data/classifier: the final
              MGUS-MM model, datasets/ (training and test sets, feature_columns/); data/metadata/
              tumor-cohort-metadata-final-240325.csv.

Outputs:      results/figure_inputs: pcs_cells.npz, pcs_targets.npz, tumors_cells.npz, tumors_cnv.npz,
              longitudinal_cells.npz, external_<study>.npz, pcs_signatures.npz, longitudinal_signatures.npz,
              pcs_senescence.npz, extract_cells_log.txt.

Dependencies: Python 3.8 with requirements.txt (h5py, numpy, pandas, scikit-learn 1.0.2); config.py.
"""
import os
import pickle
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

OUT = str(result_dir("figure_inputs"))
LOG = open(os.path.join(OUT, "extract_cells_log.txt"), "a")
PCS, TUM, LONG = [str(data("paper", "anndata", n)) for n in ("plasma-cell-GEX5-umap.h5ad", "tumor-cell-GEX5-umap.h5ad",
                                                              "longitudinal-samples-umap.h5ad")]
PRED = data("paper", "classifier", "cell-level-prediction-knn30.pkl")
META = data("metadata", "tumor-cohort-metadata-final-240325.csv")
TARGETS = ["TNFRSF17", "GPRC5D", "SDC1", "CD38", "SLAMF7", "FCRL5", "CD46", "CD47", "CD48", "NCAM1", "ITGB7", "CD74", "LY9",
           "TNFRSF13B", "CD19", "BCL2"]
PARTNERS = ["CCND1", "CCND2", "CCND3", "NSD2", "FGFR3", "MAF", "MAFA", "MAFB", "MYC"]
T0 = time.time()


def log(*a):
    s = "[%5.0fs] " % (time.time() - T0) + " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def obs_index(f):
    g = f["obs"]
    return dec(g[g.attrs.get("_index", "_index")][...])


def obs_cat(f, k):
    """(codes, categories) of an obs column (categorical, or string converted to categorical)."""
    e = f["obs"][k]
    if isinstance(e, h5py.Group) and "codes" in e:
        return e["codes"][...].astype(np.int32), dec(e["categories"][...])
    v = dec(e[...])
    cats, codes = np.unique(v, return_inverse=True)
    return codes.astype(np.int32), cats


SAFE = re.compile(r"^((Pt|HD)-\d+)(:[A-Za-z0-9+;()]+)*$|^[A-Za-z][A-Za-z0-9 +;(),/*.-]*$")


def check_categories(name, cats):
    """Categories written must be paper labels or plain category names: no long digit runs (internal record numbers)."""
    bad = [c for c in cats if c not in ("nan", "None", "") and (re.search(r"\d{5,}", c) or not SAFE.match(c))]
    assert not bad, "%s: %d categories look like internal IDs" % (name, len(bad))


def put_cats(store, f, cols, public=()):
    """Categorical obs columns as codes + categories; `public` columns (sample names of published datasets) skip the
    identifier check."""
    for k in cols:
        if k not in f["obs"]:
            log("  (no obs column %s)" % k)
            continue
        codes, cats = obs_cat(f, k)
        if k not in public:
            check_categories(k, cats)
        store[k + "__codes"], store[k + "__cats"] = codes, cats


def predictions(names):
    """Per-cell malignancy ('tumor') and MST ('MM') probabilities, in the order of `names`."""
    p = pd.read_pickle(PRED)
    log("  predictions:", p.shape, list(p.columns)[:8])
    mal = [c for c in ("tumor", "Malignancy Score") if c in p.columns][0]
    mst = [c for c in ("MM", "MST Score", "scPC-MM Score") if c in p.columns][0]
    p = p.reindex(names)
    log("  cells with a score: %d of %d (columns %s, %s)" % (p[mst].notna().sum(), len(names), mal, mst))
    return p[mal].values.astype(np.float32), p[mst].values.astype(np.float32)


def save(name, store):
    np.savez_compressed(os.path.join(OUT, name), **store)
    log("wrote %s: %s" % (name, ", ".join("%s%s" % (k, list(np.shape(v))) for k, v in store.items())))


def part_pcs():
    with h5py.File(PCS, "r") as f:
        names = obs_index(f)
        s = {"umap": f["obsm"]["X_umap"][...].astype(np.float32)}
        put_cats(s, f, ["patient_id", "sample_type", "annotation", "disease_status", "isotype_status", "cytogenetics",
                        "hrd_status", "tumor_id"])
    assert len(names) == 1649290, len(names)
    s["malignancy"], s["mst"] = predictions(names)
    save("pcs_cells.npz", s)


def csr_columns(f, X, var_names, genes, row_order=None):
    """Columns `genes` of a CSR matrix group X (cells x genes), read sequentially in row blocks."""
    col = {g: i for i, g in enumerate(var_names)}
    want = np.array([col.get(g, -1) for g in genes])
    log("  genes found: %d of %d" % ((want >= 0).sum(), len(genes)))
    lookup = np.full(len(var_names), -1, dtype=np.int64)          # matrix column -> output column
    for j, c in enumerate(want):
        if c >= 0:
            lookup[c] = j
    indptr = X["indptr"][...]
    n = len(indptr) - 1
    out = np.zeros((n, len(genes)), dtype=np.float32)
    step = 50000
    for a in range(0, n, step):
        b = min(n, a + step)
        lo, hi = indptr[a], indptr[b]
        idx, dat = X["indices"][lo:hi], X["data"][lo:hi]
        rows = np.repeat(np.arange(a, b), np.diff(indptr[a:b + 1]))
        j = lookup[idx]
        sel = j >= 0
        out[rows[sel], j[sel]] = dat[sel]
        if a % 500000 == 0:
            log("  rows %d / %d" % (a, n))
    return out


def part_targets():
    with h5py.File(PCS, "r") as f:
        names = obs_index(f)
        if "raw" in f and "X" in f["raw"]:
            log("  using the raw matrix of the PC object")
            var = dec(f["raw"]["var"][f["raw"]["var"].attrs.get("_index", "_index")][...])
            X = csr_columns(f, f["raw"]["X"], var, TARGETS)
            save("pcs_targets.npz", {"expr": X, "genes": np.array(TARGETS)})
            return
    log("  no raw matrix: log1p object matched by cell")
    with h5py.File(data("h5", "agg", "plasma-cell-GEX5-log1p-250207.h5ad"), "r") as g:
        lnames = obs_index(g)
        var = dec(g["var"][g["var"].attrs.get("_index", "_index")][...])
        X = csr_columns(g, g["X"], var, TARGETS)
    pos = pd.Series(np.arange(len(lnames)), index=lnames).reindex(names)
    assert pos.notna().all(), "cells missing from the log1p object: %d" % pos.isna().sum()
    save("pcs_targets.npz", {"expr": X[pos.values.astype(int)], "genes": np.array(TARGETS)})


def part_tumors():
    with h5py.File(TUM, "r") as f:
        names = obs_index(f)
        s = {"umap": f["obsm"]["X_umap"][...].astype(np.float32)}
        put_cats(s, f, ["patient_id", "sample_type", "annotation", "disease_status", "cytogenetics", "hrd_status", "tumor_id"])
        var = list(dec(f["var"][f["var"].attrs.get("_index", "_index")][...]))
        cols = [var.index(g) for g in PARTNERS]
        X = f["X"]
        log("  X:", type(X).__name__, getattr(X, "shape", None))
        out = np.zeros((len(names), len(PARTNERS)), dtype=np.float32)
        step = 20000
        for a in range(0, len(names), step):
            out[a:a + step] = X[a:a + step, :][:, cols]
        s["partner_expr"], s["partner_genes"] = out, np.array(PARTNERS)
    save("tumors_cells.npz", s)


def internal_to_paper():
    """Internal cell names (<barcode>-1-<gex_id>: the annotation and CNV tables) -> names of the paper objects
    (<barcode>-1-<rna_sample_id>), as in 04_genomics/14_cnv_negative_cells.py: cohort metadata, and the libraries without
    a gex_id there paired with the sample sharing their barcodes among the paper cells still unmatched. Every paper cell is
    matched exactly once; internal cells outside the paper objects are left out."""
    ann = pd.read_pickle(data("h5", "agg", "plasma-cell-GEX5-annotation-250207.pkl")).index.astype(str)
    with h5py.File(PCS, "r") as f:
        paper = pd.Index(obs_index(f))
    meta = pd.read_csv(META)
    gex2rna = meta.drop_duplicates("gex_id").set_index("gex_id")["rna_sample_id"].to_dict()
    sp = [x.split("-1-", 1) for x in ann]
    name = lambda: pd.Index([b_ + "-1-" + gex2rna.get(g, "NA") for b_, g in sp])
    h5name = name()
    left = set(g for _, g in sp) - set(gex2rna)
    un = {}
    for x in paper[~paper.isin(h5name)]:
        b_, smp = x.split("-1-", 1)
        un.setdefault(b_, []).append(smp)
    for g in left:
        hits = pd.Series([smp for b_, gg in sp if gg == g for smp in un.get(b_, [])]).value_counts()
        gex2rna[g] = hits.index[0]
    h5name = name()
    pos = h5name.get_indexer(paper)
    assert (pos >= 0).all(), "%d paper cells not matched" % (pos < 0).sum()
    log("  internal names -> paper names: %d paper cells matched (%d libraries paired by barcodes)" % (len(paper), len(left)))
    return pd.Series(paper.values, index=ann[pos])


def by_paper_names(df, mapping):
    """Rows of a table indexed by internal cell names, re-indexed by paper names (cells outside the paper dropped)."""
    df = df.loc[df.index.astype(str).isin(mapping.index)]
    df.index = mapping.reindex(df.index.astype(str)).values
    assert df.index.is_unique
    return df


def part_tumors_cnv():
    """Supplementary Fig. 11B: Numbat 1q/13q calls (no call = neutral; amp/bamp = amplification; del/bdel = deletion),
    call probability (no call = 0) and the CNV-like score quantile-transformed to a normal distribution over the
    malignant cells (QuantileTransformer, random_state 42, scikit-learn 1.0.2)."""
    from sklearn.preprocessing import QuantileTransformer
    with h5py.File(TUM, "r") as f:
        names = obs_index(f)
    s = {}
    mapping = internal_to_paper()
    scores = pd.read_pickle(data("signature", "score", "cell-level-copy-number-full-gene-expr.pkl"))
    if not scores.index.astype(str).isin(names).any():
        scores = by_paper_names(scores, mapping)
    for arm in ("1q", "13q"):
        c = "mean_expr:%s_full_log_fc" % arm
        v = scores[c].reindex(names)
        log("  %s CNV-like score: cells with a value %d of %d" % (arm, v.notna().sum(), len(names)))
        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        s[arm + "_fc"] = v.values.astype(np.float32)
        s[arm + "_qt"] = qt.fit_transform(v.values[:, None])[:, 0].astype(np.float32)
    del scores
    status = pd.read_pickle(data("cnv", "plasma-cell-cnv-cytoband-matrix-250109.pkl"))
    log("  CNV matrix:", status.shape)
    status = status[["1q_arm", "13q_arm", "cnv_prob:1q_arm", "cnv_prob:13q_arm"]]
    if not status.index.astype(str).isin(names).any():
        status = by_paper_names(status, mapping)
    st = status.reindex(names)
    del status
    for arm in ("1q", "13q"):
        raw = st[arm + "_arm"].astype(object)
        log("  %s calls on malignant cells:" % arm, raw.fillna("(none)").astype(str).value_counts().to_dict())
        lab = raw.astype(object).where(raw.notna(), "Neutral").replace({"amp": "Amplification", "bamp": "Amplification",
                                                                       "del": "Deletion", "bdel": "Deletion"})
        cats = np.array(["Amplification", "Neutral", "Deletion"])
        assert set(lab.unique()) <= set(cats), sorted(set(lab.unique()) - set(cats))
        s[arm + "_call__codes"] = pd.Categorical(lab, categories=cats).codes.astype(np.int8)
        s[arm + "_call__cats"] = cats
        s[arm + "_prob"] = st["cnv_prob:%s_arm" % arm].fillna(0).values.astype(np.float32)
    save("tumors_cnv.npz", s)


def part_longitudinal():
    with h5py.File(LONG, "r") as f:
        names = obs_index(f)
        keys = list(f["obsm"].keys())
        log("  obsm keys:", keys, "| obs keys:", [k for k in f["obs"].keys() if not k.startswith("_")][:40])
        s = {"umap": f["obsm"]["X_umap_pca" if "X_umap_pca" in f["obsm"] else "X_umap"][...].astype(np.float32)}
        put_cats(s, f, ["patient_id", "sample_type", "annotation", "disease_status"])
    s["malignancy"], s["mst"] = predictions(names)
    save("longitudinal_cells.npz", s)


def part_external():
    from sklearn.preprocessing import QuantileTransformer
    model = pickle.load(open(data("classifier", "sc-disease-model-knn30-RF.pkl"), "rb"))
    ref = pd.concat([pd.read_pickle(data("classifier", "datasets", "sc-train-cohort-smoothed-expr_134samples.pkl")),
                     pd.read_pickle(data("classifier", "datasets", "sc-test-cohort-smoothed-expr_47samples.pkl"))])
    ref = ref[~ref["disease_status"].isin(["LRSMM", "IRSMM", "HRSMM"])]
    ref_cols = pd.read_pickle(data("signature", "external", "ledergor_natmedicine2018_features-knn30.pkl")).columns
    ref = ref[["annotation", "disease_status"] + list(ref_cols)]
    load = lambda n: [x.rstrip() for x in open(data("classifier", "datasets", "feature_columns", n))]
    canonical = load("auc_16_gene_modules.txt") + load("mean_expr_fc_driver_cnv_arms.txt")
    for study, feat, h5, drop in [("ledergor", "ledergor_natmedicine2018", "CA3Data_Ledergor2018", []),
                                  ("boiarsky", "boiarsky_natcommun2022", "Boiarsky_NatCommun2022", []),
                                  ("dang", "dang_cancercell2023", "Dang_CancerCell2023",
                                   ["MGUS02s", "MGUS18", "SMM02s", "SMM05s"])]:
        path = data("h5", "external", h5, "scaled_umap.h5ad")
        with h5py.File(path, "r") as f:
            names = obs_index(f)
            s = {"umap": f["obsm"]["X_umap"][...].astype(np.float32)}
            put_cats(s, f, ["sample_id", "disease_status", "annotation"],     # published datasets: labels are public
                     public=("sample_id", "disease_status", "annotation"))
        feats = pd.read_pickle(data("signature", "external", "%s_features-knn30.pkl" % feat)).loc[names]
        sid = s["sample_id__cats"][s["sample_id__codes"]]
        keep = ~np.isin(sid, drop)                                   # four samples excluded at the authors' request
        feats = feats.join(pd.DataFrame({"annotation": s["annotation__cats"][s["annotation__codes"]],
                                       "disease_status": s["disease_status__cats"][s["disease_status__codes"]]},
                                      index=names))
        df = pd.concat([feats[keep], ref])
        auc = list(df.columns.intersection(load("auc_16_gene_modules.txt")))
        arms = list(df.columns.intersection(load("mean_expr_fc_driver_cnv_arms.txt")))
        assert auc + arms == canonical
        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        X = df[auc].join(pd.DataFrame(qt.fit_transform(df[arms]), index=df.index, columns=arms)).loc[feats.index[keep]]
        mst = np.full(len(names), np.nan, dtype=np.float32)
        mst[keep] = model.predict_proba(X.values)[:, list(model.classes_).index("MM")]
        s["mst"], s["kept"] = mst, keep
        save("external_%s.npz" % study, s)


def part_signatures():
    expr = pd.read_pickle(data("paper", "signatures", "cell-level-module-gene-expr-auc-final.pkl"))
    cols = [c for c in expr.columns if c.startswith("mean_expr:")]
    log("  signature table:", expr.shape, "| mean_expr columns:", len(cols), cols[:20])
    for path, name in [(PCS, "pcs_signatures.npz"), (LONG, "longitudinal_signatures.npz")]:
        with h5py.File(path, "r") as f:
            names = obs_index(f)
        m = expr[cols].reindex(names)
        log("  %s: cells with scores %d of %d" % (name, m.notna().all(1).sum(), len(names)))
        save(name, {"scores": m.values.astype(np.float32), "names": np.array(cols)})


def part_senescence():
    """Per-cell SenMayo senescence score (mean log-normalized expression of the SenMayo genes) for Supplementary Fig. 11A,
    from the per-cell senescence table (internal cell names, mapped to the names of the paper objects). The log compares
    its per-sample means with the kNN-smoothed per-sample feature of the classifier."""
    with h5py.File(PCS, "r") as f:
        names = obs_index(f)
        obs = {k: obs_cat(f, k) for k in ("disease_status", "sample_type", "annotation", "patient_id")}
    s = pd.read_pickle(data("signature", "score", "cell-level-senescence-expr-auc.pkl"))[["mean_expr:SenMayo"]]
    if not s.index.astype(str).isin(names).any():
        s = by_paper_names(s, internal_to_paper())
    v = s["mean_expr:SenMayo"].reindex(names)
    log("  SenMayo: cells with a score %d of %d" % (v.notna().sum(), len(names)))
    unit = pd.Series(v.values).groupby([pd.Series(c[codes]) for codes, c in [obs[k] for k in obs]]).mean()
    unit.index = [":".join(map(str, k)) for k in unit.index]
    ref = pd.read_pickle(data("paper", "classifier", "patient_id_mean_probability.pkl"))["mean_expr:SenMayo"]
    both = ref.index.intersection(unit.index)
    log("  per-sample means vs the classifier's kNN-smoothed per-sample feature: %d units, median %.3f vs %.3f"
        % (len(both), unit[both].median(), ref[both].median()))
    save("pcs_senescence.npz", {"senmayo": v.values.astype(np.float32)})


PARTS = {"pcs": part_pcs, "targets": part_targets, "tumors": part_tumors, "tumors_cnv": part_tumors_cnv,
         "longitudinal": part_longitudinal, "external": part_external, "signatures": part_signatures,
         "senescence": part_senescence}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PARTS):
        log("==== part", name)
        PARTS[name]()
    LOG.close()
