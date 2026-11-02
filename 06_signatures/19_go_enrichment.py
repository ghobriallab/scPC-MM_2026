#!/usr/bin/env python3
"""Gene Ontology and Reactome terms enriched among the genes of each of the 16 signatures (Extended Data Fig. 5A).

Purpose:      Enrichment computed offline from the annotation files of g:Profiler version e112_eg59_p19, the version
              named in the Methods. For each signature and data source (GO:BP, GO:MF, GO:CC, Reactome): one-sided
              hypergeometric test of every term against the genes annotated in the source (g:Profiler's "annotated"
              domain), with Benjamini-Hochberg FDR over all terms of the source. Signature genes are matched by Ensembl
              ID through the 10x reference used for alignment (GRCh38-2020-A), so that renamed genes (e.g. HIST1H2AC =
              H2AC6) keep their ID. The table keeps, for every signature, the 45 terms shown in the figure (three chosen
              per signature); a term name found in several sources keeps its GO version. The domain, query sizes and
              gene counts are logged next to the values returned by the g:Profiler web service for the same version.

Inputs:       data/paper/signatures/consensus-modules_genes.csv: index = gene, column context = signature.
              results/signatures/gene_ids_10x_GRCh38-2020-A.tsv: gene_id, symbol (written by 18_gene_ids.py).
              data/pathway/gprofiler_e112_eg59_p19/hsapiens.{GO:BP,GO:MF,GO:CC,REAC}.ENSG.gmt: term id, name and
              Ensembl IDs of each term (public g:Profiler data archive, version e112_eg59_p19).

Outputs:      results/signatures/go_enrichment_selected.tsv: selected term x signature: source, term, sizes, count, P, FDR.
              results/signatures/go_enrichment_log.txt: query sizes and checks.

Dependencies: Python 3.8 with requirements.txt (pandas, numpy, scipy, statsmodels); config.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data, result, result_dir

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

OUT = result_dir("signatures")
LOG = open(os.path.join(OUT, "go_enrichment_log.txt"), "w")
SOURCES = ["GO:BP", "GO:MF", "GO:CC", "REAC"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


# three representative terms per signature (the terms shown in Extended Data Fig. 5A)
ENRICHED = OrderedDict([
    ("cell_cycling_G2M", ["cell cycle process", "mitotic spindle organization", "cell cycle G2/M phase transition"]),
    ("cell_cycling_G1S", ["cell cycle process", "DNA replication", "positive regulation of cell cycle G1/S phase transition"]),
    ("B_cell_activation", ["B cell activation", "ribonucleoside catabolic process", "DNA cytosine deamination"]),
    ("MHC-II_immune_checkpoint", ["MHC class II protein complex binding", "regulation of leukocyte cell-cell adhesion",
                                  "antigen receptor-mediated signaling pathway"]),
    ("FOS-JUN_stress_response", ["positive regulation of RNA metabolic process", "transcription factor AP-1 complex",
                                 "DNA-binding transcription activator activity"]),
    ("MCL1-BCL2_signature", ["regulation of metabolic process", "activation of immune response",
                             "regulation of programmed cell death"]),
    ("circulatory_capacity", ["positive regulation of plasma membrane repair", "supramolecular fiber organization",
                              "membrane raft organization"]),
    ("interferon_inducible", ["defense response", "response to cytokine", "inflammatory response"]),
    ("HRD_signature", ["ribosomal subunit", "cytoplasmic translation", "extracellular exosome"]),
    ("myeloma_oncogene", ["cell migration", "cell surface receptor signaling pathway", "cell leading edge"]),
    ("secretory_granule", ["secretory granule membrane", "secretory vesicle", "nuclear envelope lumen"]),
    ("MAF*_signature", ["intracellular signaling cassette", "cell adhesion", "actin filament-based process"]),
    ("vacuole_ER_signaling", ["extracellular exosome", "endoplasmic reticulum", "vacuole"]),
    ("MHC-I_gene_module", ["MHC class I protein complex", "positive regulation of cytokine production",
                           "regulation of lymphocyte mediated immunity"]),
    ("histone_chaperone_gene", ["chaperone cofactor-dependent protein refolding", "response to unfolded protein",
                                "Nucleosome assembly"]),
    ("extracellular_matrix", ["regulation of collagen fibril organization", "cell adhesion", "extracellular matrix"]),
])

# signature genes -> Ensembl IDs of the 10x reference
mods = pd.read_csv(data("paper", "signatures", "consensus-modules_genes.csv"), index_col=0)
ref = pd.read_csv(result("signatures", "gene_ids_10x_GRCh38-2020-A.tsv"), sep="\t")
sym2id = ref.drop_duplicates("symbol").set_index("symbol")["gene_id"]
order = list(mods["context"].unique())                       # signatures in the order of the gene table (query order)
queries = OrderedDict()
for m in order:
    genes = list(mods.index[mods["context"] == m])
    ids = set(sym2id.reindex(genes).dropna())
    queries[m] = ids
    log("%-26s %3d genes, %3d with an Ensembl ID" % (m, len(genes), len(ids)))

# annotations: term -> genes, per source; domain = genes annotated in the source
rows = []
for src in SOURCES:
    terms = {}
    with open(data("pathway", "gprofiler_e112_eg59_p19", "hsapiens.%s.ENSG.gmt" % src)) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            terms[f[0]] = (f[1], set(f[2:]))
    root = [t for t in terms if t.endswith(":0000000")]            # the source's root term holds the annotated domain
    domain = set().union(*[g for _, g in terms.values()])
    log("%s: %d terms, annotated domain %d genes" % (src, len(terms), len(domain)))
    ids, names, sizes = list(terms), [terms[t][0] for t in terms], np.array([len(terms[t][1] & domain) for t in terms])
    for m, q in queries.items():
        qd = q & domain
        k = np.array([len(terms[t][1] & qd) for t in ids])
        p = hypergeom.sf(k - 1, len(domain), sizes, len(qd))       # P(X >= k)
        p[k == 0] = 1.0
        fdr = multipletests(p, method="fdr_bh")[1]
        want = set(sum(ENRICHED.values(), []))
        for i, t in enumerate(ids):
            if names[i] in want:
                rows.append((src, t, names[i], m, len(domain), len(qd), sizes[i], int(k[i]), p[i], fdr[i]))
res = pd.DataFrame(rows, columns=["source", "term_id", "name", "signature", "domain", "query_size", "term_size", "count",
                                  "p", "fdr"])
# a name in several sources keeps its GO version
keep = []
for nm, df in res.groupby("name"):
    srcs = df["source"].unique()
    keep.append(df[df["source"].str.startswith("GO:")] if len(srcs) > 1 else df)
res = pd.concat(keep)
res.to_csv(os.path.join(OUT, "go_enrichment_selected.tsv"), sep="\t", index=False)

# checks against the values returned by the g:Profiler web service (same version)
bp = res[res["source"] == "GO:BP"]
qs = bp.drop_duplicates("signature").set_index("signature")["query_size"].reindex(order)
log("GO:BP domain %d (g:Profiler web service: 21017)" % bp["domain"].iloc[0])
log("GO:BP query sizes in signature order:", list(qs.values[:10]),
    "(g:Profiler web service: 168, 158, 140, 107, 115, 81, 61, 53, -, 32)")
ccp = bp[bp["name"] == "cell cycle process"].set_index("signature")["count"].reindex(order)
log("'cell cycle process' genes per signature:", list(ccp.values[:14]),
    "(g:Profiler web service: 11, 14, 9, 8, 7, 69, 6, 5, -, 4, 2, 2, 1, 34)")
for nm, sig in [("B cell activation", "B_cell_activation"), ("cell cycle process", "cell_cycling_G2M")]:
    r = res[(res["name"] == nm) & (res["signature"] == sig)].iloc[0]
    log("%s in %s: %d genes, P = %.3g, FDR = %.3g" % (nm, sig, r["count"], r["p"], r["fdr"]))
missing = sorted(set(sum(ENRICHED.values(), [])) - set(res["name"]))
log("selected terms not found in e112:", missing or "none")
LOG.close()
