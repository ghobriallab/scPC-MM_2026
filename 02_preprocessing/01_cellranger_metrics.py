#!/usr/bin/env python3
"""One row of the Cell Ranger metrics of one library, for the library QC table.

Purpose:      Reads the metrics summary that Cell Ranger writes for one library (a header line starting with
              "Estimated Number of Cells" and one line of values), removes the thousands separators inside quoted
              numbers, turns percentages into fractions rounded to 3 decimals and prints one comma-separated row: the
              library name (file name up to ".metrics") followed by the values in the order of the Cell Ranger header.
              Run once per library and collect the rows in one table, e.g.
              for f in <folder>/*.metrics_summary.csv; do python3 02_preprocessing/01_cellranger_metrics.py "$f"; done

Inputs:       <library>.metrics_summary.csv: Cell Ranger metrics summary of one library, named by library (path given on
              the command line).

Outputs:      One comma-separated line on standard output.

Dependencies: Python 3.8 (standard library only).
"""
import os
import sys


def metrics_row(path):
    """Library name and metric values of one Cell Ranger metrics summary file."""
    text = ""
    for line in open(path):
        if line.startswith("Estimated"):                # header line
            continue
        in_quote = False
        for char in line.rstrip("\n"):
            if char == '"':
                in_quote = not in_quote
            elif char == ",":
                if not in_quote:                        # field separator; a comma inside quotes separates thousands
                    text += "\t"
            else:
                text += char
    values = text.split("\t")
    for i, value in enumerate(values):
        if value.endswith("%"):
            values[i] = str(round(float(value.rstrip("%")) / 100, 3))
    library = os.path.basename(path).split(".metrics")[0]
    return [library] + values


print(",".join(metrics_row(sys.argv[1])))
