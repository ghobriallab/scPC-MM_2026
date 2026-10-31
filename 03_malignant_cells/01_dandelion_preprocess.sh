#!/usr/bin/env bash
# ============================================================================
# Purpose:      Runs the dandelion preprocessing pipeline (dandelion-preprocess) on the scBCR-seq contigs of every
#               individual, one individual at a time (Methods, scBCR-seq data preprocessing and BCR filtering). For each
#               value of the individual column of the BCR library table, the rows of that individual are written to
#               <individual>_meta.csv (same columns; "/" in an individual name is written "-" in file names), and
#               dandelion-preprocess is run on it with --filter_to_high_confidence (high-confidence contigs only) and
#               --keep_trailing_hyphen_number (barcodes keep their "-1" suffix), other parameters at their defaults.
#               The pipeline re-annotates V(D)J genes with IgBLAST, reassigns heavy chain V alleles with TIgGER using
#               the genotype of each individual (germlines reconstructed when coverage is too low for genotyping),
#               reassigns constant region calls and quantifies somatic mutations.
# Inputs:       data/vdj/bcr_preprocess_metadata_367_samples.csv (one row per BCR library: sample, prefix, individual;
#               written by the cohort step).
#               <bcr_dir>/<sample>/: Cell Ranger vdj output of each BCR library (all_contig.fasta,
#               all_contig_annotations.csv), one folder per value of the sample column.
# Outputs:      <bcr_dir>/<individual>_meta.csv: the BCR libraries of each individual (read by
#               02_dandelion_clonotypes.py).
#               <bcr_dir>/<sample>/dandelion/all_contig_dandelion.tsv and the other dandelion-preprocess outputs of
#               each library, and the genotyping files of each individual.
# Dependencies: bash, awk; Singularity with the public sc-dandelion container image of dandelion 0.3.3 (file
#               sc-dandelion_latest.sif; set DANDELION_SIF to its path, default ./sc-dandelion_latest.sif). The data
#               folder follows config.py: SCPCMM_DATA, or data/ at the repository root.
# Usage:        bash 03_malignant_cells/01_dandelion_preprocess.sh <bcr_dir>
# ============================================================================
set -uo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: bash $0 <bcr_dir>   (folder with one Cell Ranger vdj output folder per BCR library)" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${SCPCMM_DATA:-$REPO_DIR/data}"
SIF="${DANDELION_SIF:-sc-dandelion_latest.sif}"
# absolute paths, as the script changes folder below
case "$DATA_DIR" in
    /*) ;;
    *) DATA_DIR="$PWD/$DATA_DIR" ;;
esac
case "$SIF" in
    /*) ;;
    *) SIF="$PWD/$SIF" ;;
esac
TABLE="$DATA_DIR/vdj/bcr_preprocess_metadata_367_samples.csv"

# dandelion-preprocess runs in the folder that holds the library folders
cd "$1" || exit 1

# column of the table that holds the individual
COL=$(head -n 1 "$TABLE" | tr -d '\r' | tr ',' '\n' | grep -n -x 'individual' | cut -d: -f1)
if [ -z "$COL" ]; then
    echo "no column named individual in $TABLE" >&2
    exit 1
fi
N=$(awk -F, -v c="$COL" '{sub(/\r$/, "")} NR > 1 {print $c}' "$TABLE" | sort -u | wc -l | tr -d ' ')

k=0
while read -r individual; do
    k=$((k + 1))
    echo "individual $k of $N"
    meta_csv="${individual//\//-}_meta.csv"
    # header and the BCR libraries of this individual
    awk -F, -v c="$COL" -v ind="$individual" '{sub(/\r$/, "")} NR == 1 || $c == ind' "$TABLE" > "$meta_csv"
    singularity run -B "$PWD" "$SIF" dandelion-preprocess \
        --filter_to_high_confidence \
        --keep_trailing_hyphen_number \
        --meta "$meta_csv" < /dev/null
done < <(awk -F, -v c="$COL" '{sub(/\r$/, "")} NR > 1 {print $c}' "$TABLE" | sort -u)
