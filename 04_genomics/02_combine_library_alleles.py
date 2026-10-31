#!/usr/bin/env python3
"""Merge the allele counts of the libraries of one tumor into one table with harmonized phasing, the allele input of
Numbat (Methods, Copy number variant detection).

Purpose:      Reads the per-library allele tables given with --allele-tsv (Numbat pileup_and_phase output, files named
              <gex_id>_allele_counts.tsv.gz; missing files are skipped), appends -<gex_id> to every cell barcode and
              sorts the rows by snp_id in natural order. With one library the table is written as is. With several, the
              phasing is harmonized per chromosome: neighboring SNPs whose recombination probability
              P = (1 - exp(-2 d)) / 2 (d, distance in cM) is <= 0.05 form haplotype blocks; in each block, every SNP
              phased in several libraries with discordant genotypes (GT) votes to keep the phase of the libraries that
              share the GT of the library with the most cells at that SNP and to swap the others (votes reversed when
              this conflicts with earlier votes of the block); libraries with swap votes and no more keep than swap
              votes have their GT flipped over the whole block. Any SNP still discordant then takes its most frequent
              GT.

Inputs:       The per-library allele tables of one tumor (command line): cell, snp_id, CHROM, cM, GT and the other
              columns of the Numbat allele table.

Outputs:      The merged allele table (gzip TSV) named on the command line (--alleledf of 03_run_numbat.R).

Dependencies: Python 3.8 with requirements.txt (numpy, pandas) and natsort.
"""
import os.path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from natsort import index_natsorted


def main(args):
    target = []
    not_exist = []
    for input_file in args.allele_tsv:
        if not os.path.exists(input_file):
            not_exist.append(input_file)
            continue
        gex_id = os.path.basename(input_file).replace("_allele_counts.tsv.gz", "")
        df = pd.read_csv(input_file, sep="\t")
        # cell names: <barcode>-<gex_id>
        df['cell'] = np.array(df['cell']) + '-' + gex_id
        df['gex_id'] = gex_id
        target.append(df.copy())
    if len(target) == 0:
        print("can't find files")
        return

    df = pd.concat(target)
    df.sort_values(by="snp_id", key=lambda x: np.argsort(index_natsorted(df["snp_id"])), inplace=True)

    if len(not_exist) > 0:
        print("can't find samples " + ", ".join(not_exist) + " ....skipped")

    # one library: written as is
    if len(df['gex_id'].unique()) == 1:
        df.drop(columns='gex_id', inplace=True)
        df.to_csv(args.output_tsv, sep="\t", index=False, compression="gzip")
        return

    print("merging genotype... ")
    df_harmonized = []
    for chrom, chunk in df.groupby("CHROM"):
        udf = chunk.copy()

        snps = chunk[['snp_id', 'cM']].drop_duplicates().reset_index(drop=True)
        snps['d'] = snps['cM'].diff()
        snps['P'] = np.nan

        # recombination probability between neighboring SNPs from their genetic distance
        snps.loc[~snps['d'].isna(), 'P'] = (1 - np.exp(-2 * snps.loc[~snps['d'].isna(), 'd'])) / 2

        # haplotype blocks: runs of SNPs linked to the SNP before them (P <= 0.05)
        snps['res'] = np.where(snps['P'] <= 0.05, 'hap', 'ind')
        snps['group'] = snps['res'].ne(snps['res'].shift()).cumsum()

        for _, hap in snps[snps['res'] == 'hap'].groupby('group'):
            beg = min(hap.index) - 1
            end = max(hap.index)
            groups = list(snps.loc[beg:end, 'snp_id'])

            gex_dict = {'swap': dict(), 'keep': dict()}
            for snp_id, x in chunk[chunk['snp_id'].isin(groups)].groupby('snp_id'):
                # SNP phased in one library only
                if len(list(x['gex_id'].unique())) == 1:
                    continue
                # same GT in every library
                if len(list(x['GT'].unique())) == 1:
                    continue
                # discordant GTs: the library with the most cells at this SNP sets the reference GT
                cells = pd.DataFrame(x['gex_id'].value_counts())
                max_gex_id = cells['gex_id'].idxmax()
                GT = list(x.loc[x['gex_id'] == max_gex_id, 'GT'].unique())[0]
                a = set(x.loc[x['GT'] == GT, 'gex_id'].unique())
                b = set(x.loc[x['GT'] != GT, 'gex_id'].unique())
                a_flag = 'keep'
                b_flag = 'swap'
                if len(set(gex_dict['swap'].keys()).intersection(a)) > 0 or \
                        len(set(gex_dict['keep'].keys()).intersection(b)) > 0:
                    a_flag = 'swap'
                    b_flag = 'keep'
                for a_id in a:
                    gex_dict[a_flag][a_id] = gex_dict[a_flag].get(a_id, 0) + 1
                for b_id in b:
                    gex_dict[b_flag][b_id] = gex_dict[b_flag].get(b_id, 0) + 1

            # swap the phase of the libraries voted 'swap' over the block
            overlap = set(gex_dict['keep'].keys()).intersection(set(gex_dict['swap'].keys()))
            for i in overlap:
                if gex_dict['keep'][i] > gex_dict['swap'][i]:
                    gex_dict['swap'].pop(i)
            swap = list(gex_dict['swap'].keys())
            index = udf['snp_id'].isin(groups) & udf['gex_id'].isin(swap)
            udf.loc[index, 'GT'] = ['|'.join(gt.split('|')[::-1]) for gt in udf.loc[index, 'GT']]

        # remaining discordant SNPs: most frequent GT
        for snp_id, x in udf.groupby('snp_id'):
            if len(list(x['gex_id'].unique())) == 1:
                continue
            if len(list(x['GT'].unique())) == 1:
                continue
            cells = pd.DataFrame(x['GT'].value_counts())
            udf.loc[udf['snp_id'] == snp_id, 'GT'] = cells['GT'].idxmax()
        df_harmonized.append(udf.copy())

    df_harmonized = pd.concat(df_harmonized).drop(columns='gex_id')
    df_harmonized.to_csv(args.output_tsv, sep="\t", index=False, compression="gzip")
    print("harmonizing genotype... done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Combine allele TSV per patient")
    parser.add_argument("output_tsv", help="output allele TSV file name")
    parser.add_argument("--allele-tsv", nargs="+", help="allele TSV files", required=True)
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
