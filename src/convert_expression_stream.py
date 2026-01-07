#!/usr/bin/env python3
"""Stream-convert large expression TSV/CSV to data/expression.csv

This reads the input file line-by-line to avoid high memory usage,
strips Ensembl version suffixes (ENSG...\.15 -> ENSG...), and ensures
unique gene IDs by adding a numeric suffix for duplicates.

Usage:
  python src/convert_expression_stream.py --in data/TCGA-BRCA.star_counts.tsv --out data/expression.csv

此文件用于将原始表达数据表转换为适合下游分析的格式，
主要处理Ensembl基因ID的版本号问题，并确保基因ID的唯一性
输出文件为逗号分隔的CSV格式，第一列为基因ID。
"""
import argparse
import os
import csv


def strip_version(gene):
    return gene.split(".", 1)[0]


def process(inpath, outpath, sep="\t"):
    with open(inpath, "r", encoding="utf-8", errors="replace") as fin:
        # read header
        header = fin.readline().rstrip("\n")
        if not header:
            raise SystemExit("Empty input")
        # try split by tab, fallback to comma
        cols = header.split(sep)
        if len(cols) < 2:
            cols = header.split(",")
            sep = ","
        samples = cols[1:]  # first is Ensembl_ID

        outdir = os.path.dirname(outpath)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        seen = {}
        with open(outpath, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["Ensembl_ID"] + samples)

            for line in fin:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split(sep)
                if len(parts) < 2:
                    # try comma
                    parts = line.rstrip("\n").split(",")
                gene_raw = parts[0].strip().strip('"')
                gene = strip_version(gene_raw)
                values = parts[1:]
                # ensure length matches header (pad/truncate)
                if len(values) < len(samples):
                    values += [""] * (len(samples) - len(values))
                elif len(values) > len(samples):
                    values = values[: len(samples)]

                if gene in seen:
                    seen[gene] += 1
                    gene_out = f"{gene}_dup{seen[gene]}"
                else:
                    seen[gene] = 0
                    gene_out = gene

                writer.writerow([gene_out] + values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inpath", required=True)
    p.add_argument("--out", dest="outpath", default="data/expression.csv")
    args = p.parse_args()
    process(args.inpath, args.outpath)


if __name__ == "__main__":
    main()
