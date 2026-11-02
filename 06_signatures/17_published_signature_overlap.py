#!/usr/bin/env python3
"""Gene overlap between the 16 signatures and 42 published myeloma and plasma cell signatures (Extended Data Fig. 5B;
Results; Methods, comparison with published signatures).

Purpose:      For each of the 672 pairs of one consensus signature and one published signature (42 signatures from ten
              studies): the Jaccard index on the full gene lists, and a one-sided hypergeometric test for enrichment
              with the 4,980 genes used as Hotspot input as the universe (the only genes that could enter a signature;
              both lists restricted to it), with Benjamini-Hochberg correction over all 672 pairs.

Inputs:       data/signature/hotspot/consensus-modules_genes.csv: index = gene, column context = signature.
              data/signature/hotspot/mm-BM-4980-full-DEs_hotspot_input.h5ad: var index = the 4,980 input genes.
              data/pathway/external/, published gene lists (columns used):
                myeloma_nmf_Boiarsky_etal_NatCommun2022.csv: index = gene; context, description
                myeloma_scrna_DaVia_etal_Blood2025.csv: gene, context (== healthy_PC_signature)
                GEP_risk_score_Heuck-Shaughnessy_etal_Leukemia2014-Blood2007.csv: index = gene; context
                ctc_signature_Lightbody_etal_NatCancer2025.csv: index = gene
                nfkb_index_Annunziata_etal_CancerCell2007.txt: one gene per line
                ras_signature_Lin_etal_Blood2019.csv: index = gene; context
                microarray_signature_Zhan_etal_Blood2006.csv: index = gene; context
                CoMMPass_12RnaSubtypes_NatGenetics2024.csv: gene, RNA_Subtype_Name
                senmayo_Saul_etal_NatCommun_2022.txt: one gene per line

Outputs:      results/signatures/published_signature_overlap_pairs.tsv: each pair: sizes, overlap, Jaccard, P, q, shared genes.
              results/signatures/published_signature_overlap_log.txt: gene counts and the strongest overlaps.

Dependencies: Python 3.8 with requirements.txt (pandas, scipy, statsmodels, h5py); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result_dir

import h5py
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

EXT = data("pathway", "external")
OUT = result_dir("signatures")
FH = open(os.path.join(OUT, "published_signature_overlap_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    FH.write(s + "\n")


# ---- the 16 signatures and the Hotspot gene universe ------------------------------------------
gm16 = pd.read_csv(data("signature", "hotspot", "consensus-modules_genes.csv"), index_col=0)
ours = {k: set(v.index.astype(str)) for k, v in gm16.groupby("context")}
with h5py.File(str(data("signature", "hotspot", "mm-BM-4980-full-DEs_hotspot_input.h5ad")), "r") as f:
    key = f["var"].attrs.get("_index", "_index")
    U = set(x.decode() if isinstance(x, bytes) else str(x) for x in f["var"][key][...])
log("signatures:", len(ours), "| signature genes:", sum(len(v) for v in ours.values()), "| universe (Hotspot input):", len(U),
    "| signature genes outside the universe:", len(set.union(*ours.values()) - U))

# ---- published signatures -----------------------------------------------------------------------
ext = []   # (study, label, genes)
b = pd.read_csv(EXT / "myeloma_nmf_Boiarsky_etal_NatCommun2022.csv", index_col=0)
for (c, d), g in b.groupby(["context", "description"]):
    ext.append(("Boiarsky 2022", "%s: %s" % (c, d.split(" (")[0]), set(g.index.astype(str))))   # short label
dv = pd.read_csv(EXT / "myeloma_scrna_DaVia_etal_Blood2025.csv")
ext.append(("Da Vià 2025", "healthy PC signature", set(dv.loc[dv["context"] == "healthy_PC_signature", "gene"].astype(str))))
gp = pd.read_csv(EXT / "GEP_risk_score_Heuck-Shaughnessy_etal_Leukemia2014-Blood2007.csv", index_col=0)
for c, g in gp.groupby("context"):
    ext.append(("Shaughnessy 2007" if c == "GEP70" else "Heuck 2014", c, set(g.index.astype(str))))
lb = pd.read_csv(EXT / "ctc_signature_Lightbody_etal_NatCancer2025.csv", index_col=0)
ext.append(("Lightbody 2025", "CTC signature", set(lb.index.astype(str))))
ext.append(("Annunziata 2007", "NF-kB index", set(x.strip() for x in open(EXT / "nfkb_index_Annunziata_etal_CancerCell2007.txt") if x.strip())))
rs = pd.read_csv(EXT / "ras_signature_Lin_etal_Blood2019.csv", index_col=0)
for c, g in rs.groupby("context"):
    ext.append(("Lin 2019", c.replace("siganture", "signature").replace("_", " "), set(g.index.astype(str))))
zh = pd.read_csv(EXT / "microarray_signature_Zhan_etal_Blood2006.csv", index_col=0)
for c, g in zh.groupby("context"):
    ext.append(("Zhan 2006", c.replace("_", "-") + " subgroup", set(g.index.astype(str))))
cp = pd.read_csv(EXT / "CoMMPass_12RnaSubtypes_NatGenetics2024.csv", index_col=0)
cp = cp.loc[cp["gene"].notna()]
for c, g in cp.groupby("RNA_Subtype_Name"):
    ext.append(("Skerget 2024", c.replace("_", " ") + " subtype", set(g["gene"].astype(str))))
ext.append(("Saul 2022", "SenMayo", set(x.strip() for x in open(EXT / "senmayo_Saul_etal_NatCommun_2022.txt") if x.strip())))
log("published signatures:", len(ext), "from", len(set(e[0] for e in ext)), "studies")

# ---- overlap statistics for every pair ------------------------------------------------------------
N = len(U)
rows = []
for s, S in ours.items():
    for study, lab, E in ext:
        k = len(S & E)
        Su, Eu = S & U, E & U          # both lists restricted to the universe for the test
        ku = len(Su & Eu)
        rows.append({"signature": s, "study": study, "published_signature": lab, "n_signature": len(S),
                     "n_published": len(E), "n_published_in_universe": len(Eu), "overlap": k,
                     "jaccard": k / len(S | E), "expected_overlap": len(Su) * len(Eu) / N,
                     "p_enrichment": hypergeom.sf(ku - 1, N, len(Eu), len(Su)) if ku else 1.0,
                     "shared_genes": ",".join(sorted(S & E))})
pairs = pd.DataFrame(rows)
pairs["q"] = multipletests(pairs["p_enrichment"], method="fdr_bh")[1]
pairs["published"] = pairs["study"] + ", " + pairs["published_signature"]
pairs.sort_values("jaccard", ascending=False).to_csv(os.path.join(OUT, "published_signature_overlap_pairs.tsv"), sep="\t",
                                                     index=False)
log("pairs:", len(pairs), "| Jaccard median %.3f, 95th percentile %.3f, max %.3f" % (
    pairs["jaccard"].median(), pairs["jaccard"].quantile(0.95), pairs["jaccard"].max()),
    "| pairs with Jaccard > 0.05:", int((pairs["jaccard"] > 0.05).sum()), "| pairs with q < 0.05:", int((pairs["q"] < 0.05).sum()))
for _, r in pairs.sort_values("jaccard", ascending=False).head(12).iterrows():
    log("  %-28s %-55s J %.3f  overlap %d  q %.2g" % (r["signature"], r["published"][:55], r["jaccard"], r["overlap"], r["q"]))

FH.close()
