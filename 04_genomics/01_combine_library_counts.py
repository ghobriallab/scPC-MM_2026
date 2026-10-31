#!/usr/bin/env python3
"""Merge the count matrices of the libraries of one tumor into one 10x-format HDF5 file, the count input of Numbat
(Methods, Copy number variant detection).

Purpose:      Reads the per-library count matrices given with --h5-paths (10x HDF5 files named
              <gex_id>_out_filtered.h5), appends -<gex_id> to every cell barcode, makes gene names unique,
              concatenates the libraries (anndata concat, merge="same") and writes the merged matrix in the 10x HDF5
              layout of the inputs (CellRanger version 2 or 3, detected from the last input file; matrix stored as
              genes x cells, CSC). Run once per tumor, with the libraries of that tumor.

Inputs:       The per-library 10x HDF5 count matrices of one tumor (command line): matrix, barcodes and the features
              name, id, feature_type and genome.

Outputs:      The merged 10x HDF5 file named on the command line (--countmat of 03_run_numbat.R).

Dependencies: Python 3.8 with requirements.txt (anndata, numpy, scanpy, scipy) and PyTables (tables).
"""
import logging
import os
from argparse import ArgumentParser
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import tables


def detect_cellranger_version_h5(filename: str) -> int:
    """CellRanger version (2 or 3) of a 10x HDF5 file: version 3 files have a 'features' group under 'matrix'."""
    with tables.open_file(filename, 'r') as f:
        version = 2
        try:
            getattr(f.root.matrix, 'features')
            version = 3
        except tables.NoSuchNodeError:
            pass
    return version


def write_matrix_to_cellranger_h5(
        cellranger_version: int,
        output_file: str,
        gene_names: np.ndarray,
        barcodes: np.ndarray,
        count_matrix: sp.csc_matrix,
        feature_types: Optional[np.ndarray] = None,
        gene_ids: Optional[np.ndarray] = None,
        genomes: Optional[np.ndarray] = None) -> bool:
    """Write a cells x genes count matrix to a CellRanger-layout HDF5 file (stored transposed, genes x cells)."""
    assert isinstance(count_matrix, sp.csc_matrix), \
        "The count matrix must be csc_matrix format in order to write to HDF5."
    assert gene_names.size == count_matrix.shape[1], \
        "The number of gene names must match the number of columns in the count matrix."
    if gene_ids is not None:
        assert gene_names.size == gene_ids.size, \
            f"The number of gene_names {gene_names.shape} must match the number of gene_ids {gene_ids.shape}."
    if feature_types is not None:
        assert gene_names.size == feature_types.size, \
            f"The number of gene_names {gene_names.shape} must match the number of feature_types {feature_types.shape}."
    if genomes is not None:
        assert gene_names.size == genomes.size, \
            "The number of gene_names must match the number of genome designations."
    assert barcodes.size == count_matrix.shape[0], \
        "The number of barcodes must match the number of rows in the count matrix."

    # rows become genes and columns cells, as in CellRanger files
    count_matrix = count_matrix.transpose().tocsc()

    try:
        with tables.open_file(output_file, "w", title="CellBender remove-background output") as f:
            if cellranger_version == 2:
                group = f.create_group("/", "matrix", "Counts after merging H5 files")
                f.create_array(group, "gene_names", gene_names)
                if gene_ids is not None:
                    f.create_array(group, "genes", gene_ids)
            elif cellranger_version == 3:
                group = f.create_group("/", "matrix", "Counts after merging H5 files")
                feature_group = f.create_group(group, "features", "Genes and other features measured")
                f.create_array(feature_group, "name", gene_names)
                if gene_ids is not None:
                    f.create_array(feature_group, "id", gene_ids)
                if feature_types is not None:
                    f.create_array(feature_group, "feature_type", feature_types)
                if genomes is not None:
                    f.create_array(feature_group, "genome", genomes)
            else:
                raise NotImplementedError(f'Trying to save to CellRanger v{cellranger_version} format, which '
                                          f'is not implemented.')

            # both versions: barcodes and the sparse count data
            f.create_array(group, "barcodes", barcodes)
            f.create_array(group, "data", count_matrix.data)
            f.create_array(group, "indices", count_matrix.indices)
            f.create_array(group, "indptr", count_matrix.indptr)
            f.create_array(group, "shape", count_matrix.shape)

        logging.info(f"Succeeded in writing CellRanger v{cellranger_version} format output to file {output_file}")
        return True

    except Exception:
        logging.warning(f"Encountered an error writing output to file {output_file}.  Output may be incomplete.")
        return False


def main(args):
    target = []
    for input_file in args.h5_paths:
        cellranger_version = detect_cellranger_version_h5(input_file)
        gex_id = os.path.basename(input_file).replace("_out_filtered.h5", "")
        adata = sc.read_10x_h5(input_file)
        adata.var_names_make_unique()
        # cell names: <barcode>-<gex_id>
        adata.obs.index = [x + '-' + gex_id for x in adata.obs.index]
        target.append(adata)

    target = ad.concat(target, merge="same")

    write_matrix_to_cellranger_h5(
        cellranger_version=cellranger_version,
        output_file=args.output_h5,
        gene_names=np.array(target.var_names, dtype=str),
        gene_ids=np.array(target.var['gene_ids'].values, dtype=str),
        feature_types=np.array(target.var['feature_types'].values, dtype=str),
        genomes=np.array(target.var['genome'].values, dtype=str),
        barcodes=np.array(target.obs_names, dtype=str),
        count_matrix=sp.csc_matrix(target.X)
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Combine multiple 10X(-like) H5 files")
    parser.add_argument("output_h5", help="output H5 file name")
    parser.add_argument("--h5-paths", nargs="+", help="H5 files", required=True)
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
