#!/usr/bin/env python3
"""BCR clonotypes of the PCs of one individual with dandelion, added to the gene expression object of that individual
(Methods, scBCR-seq data preprocessing and BCR filtering).

Purpose:      For one individual (a value of the individual column of data/vdj/bcr_preprocess_metadata_367_samples.csv),
              reads the dandelion-preprocess output of each of its BCR libraries, renames the cells as in the gene
              expression (GEX) data, keeps the cells present in the GEX object of the individual, removes cells whose
              locus_status starts with "Extra VDJ" (more than one heavy chain) or is "Orphan IGH" (heavy chain only),
              cross-checks the contigs with the GEX object (dandelion check_contigs, productive contigs only, library
              type ig) and calls clonotypes (dandelion find_clones). Change-O clones are also defined: Hamming-distance
              threshold estimated (saved), then set to 0.15, and define_clones with doublets counted; this step is
              skipped if Change-O fails or with --skip-changeo. Adds clone sizes (also capped at 5) and transfers the
              BCR data to the GEX object. The script restarts itself with PYTHONHASHSEED=0 if the variable is not set.

Inputs:       <bcr_dir>/<individual>_meta.csv (written by 01_dandelion_preprocess.sh; first column = BCR library).
              <bcr_dir>/<library>/dandelion/all_contig_dandelion.tsv (dandelion-preprocess output of each library).
              data/h5/pat/clustered/annot/<individual>.h5ad (PCs of the individual; cell names <barcode>-<GEX library>,
              the GEX library named as the BCR library with _VDJ_BCR replaced by _GEX_5).
              In file names, "/" in an individual name is written "-".

Outputs:      data/amp1q/bcr/firstpass_h5ddl/<individual>_vdj.h5ddl (dandelion object, read by 03_cohort_bcr_tables.py),
              <individual>_vdj.h5ad (GEX object with the BCR data), <individual>_auto_cutoff.txt (estimated Change-O
              threshold), <individual>_auto_cutoff.pdf and <individual>_manual_cutoff.pdf (distance distributions).

Dependencies: Python 3.8 with dandelion 0.3.3 and its dependencies (pandas 2.0.3, numpy 1.24.4, scipy 1.10.1), scanpy
              and rpy2, with R and SHazaM and with Change-O for the Change-O clones; config.py. Run once per individual
              of data/vdj/bcr_preprocess_metadata_367_samples.csv (columns sample, prefix, individual), e.g.
                  awk -F, 'NR > 1 {print $3}' data/vdj/bcr_preprocess_metadata_367_samples.csv | sort -u |
                      while read -r p; do
                          python3 03_malignant_cells/02_dandelion_clonotypes.py --in-dir <bcr_dir> "$p"
                      done
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # config.py
from config import data

# hash-based orderings must not change between runs: restart with PYTHONHASHSEED=0 if it is not set
if not os.getenv("PYTHONHASHSEED"):
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import gc
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import rpy2
import scanpy as sc
import dandelion as ddl


def main(args):
    stem = args.individual.replace("/", "-")         # the individual as written in file names
    meta = pd.read_csv(os.path.join(args.in_dir, stem + "_meta.csv"), index_col=0)
    out = data("amp1q", "bcr", "firstpass_h5ddl")
    out.mkdir(parents=True, exist_ok=True)
    prefix = str(out / stem)

    print("Load GEX data")
    adata = sc.read(data("h5", "pat", "clustered", "annot", stem + ".h5ad"))

    print("Load VDJ data")
    vdj_list = []
    for bcr_id in meta.index:
        vdj_list.append(ddl.read_10x_airr(os.path.join(args.in_dir, bcr_id, "dandelion", "all_contig_dandelion.tsv")))
    vdj = ddl.concat(vdj_list)

    print("Rename cells as in the GEX data")
    # <BCR library>_<barcode> -> <barcode>-<GEX library>, where _VDJ_BCR in the library name becomes _GEX_5
    vdj.data["cell_id"] = ["-".join(cell_id.replace("_VDJ_BCR_", "_GEX_5::").split("::")[::-1])
                           for cell_id in vdj.data["cell_id"]]
    vdj.update_metadata()

    print("Keep cells present in the GEX data")
    vdj = vdj[vdj.data["cell_id"].isin(adata.obs.index)]

    print("Remove cells with multiple heavy chains or a heavy chain only")
    vdj = vdj[~vdj.metadata["locus_status"].str.match("Extra VDJ") & (vdj.metadata["locus_status"] != "Orphan IGH")]
    gc.collect()

    print("Cross-check contigs with the GEX data")
    vdj, adata = ddl.pp.check_contigs(vdj, adata, productive_only=True, library_type="ig")

    print("Clonotypes with dandelion")
    ddl.tl.find_clones(vdj)
    gc.collect()

    # DataFrame.iteritems, called when tables are passed to R, is DataFrame.items in pandas 2
    pd.DataFrame.iteritems = pd.DataFrame.items

    if not args.skip_changeo:
        print("Clones with Change-O")
        try:
            ddl.pp.calculate_threshold(vdj, model="ham", save_plot=prefix + "_auto_cutoff.pdf")
            with open(prefix + "_auto_cutoff.txt", "w") as fo:
                fo.write("{}\n".format(vdj.threshold))
            ddl.pp.calculate_threshold(vdj, model="ham", manual_threshold=0.15, save_plot=prefix + "_manual_cutoff.pdf")
            ddl.tl.define_clones(vdj, key_added="changeo_clone_id", doublets="count")
            gc.collect()
        except (rpy2.rinterface_lib.embedded.RRuntimeError, AttributeError):
            print("    Change-O failed: skipped")

    print("Clone sizes")
    ddl.tl.clone_size(vdj)
    ddl.tl.clone_size(vdj, max_size=5)
    gc.collect()

    print("Transfer the BCR data to the GEX object")
    ddl.tl.transfer(adata, vdj)
    gc.collect()

    print("Save")
    adata.write(prefix + "_vdj.h5ad")
    vdj.write(prefix + "_vdj.h5ddl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCR clonotypes of one individual with dandelion")
    parser.add_argument("--in-dir", default=".",
                        help="folder given to 01_dandelion_preprocess.sh (default: the current folder)")
    parser.add_argument("--skip-changeo", action="store_true", help="skip the Change-O clones")
    parser.add_argument("individual", help="a value of the individual column of the BCR library table")
    main(parser.parse_args())
