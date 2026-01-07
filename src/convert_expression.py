#!/usr/bin/env python3
"""Convert raw expression table to pipeline-friendly data/expression.csv

Usage:
  python src/convert_expression.py --in data/TCGA-BRCA.star_counts.tsv --out data/expression.csv
If --in is omitted the script will try to autodetect a suitable file in data/.
此文件用于将原始表达数据表转换为适合下游分析的格式，
主要处理Ensembl基因ID的版本号问题，并确保基因ID的唯一性。

输出文件为逗号分隔的CSV格式，第一列为基因ID。
"""
import argparse
import os
import pandas as pd


def find_candidate(data_dir="data"):
    names = os.listdir(data_dir)
    candidates = [
        n
        for n in names
        if (
            n.lower().endswith((".tsv", ".txt", ".csv"))
            and (
                "expr" in n.lower()
                or "count" in n.lower()
                or "fpkm" in n.lower()
                or "star" in n.lower()
            )
        )
    ]
    # fallback: any tsv/csv
    if not candidates:
        candidates = [n for n in names if n.lower().endswith((".tsv", ".csv", ".txt"))]
    return os.path.join(data_dir, candidates[0]) if candidates else None


def load_table(path):
    # Try reading as TSV first with robust options (skip comment lines)
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=0,
            index_col=0,
            engine="python",
            comment="#",
            low_memory=False,
        )
        if df.shape[0] > 0 and df.shape[1] > 0:
            return df
    except Exception:
        pass
    # fallback to CSV
    try:
        df = pd.read_csv(
            path,
            sep=",",
            header=0,
            index_col=0,
            engine="python",
            comment="#",
            low_memory=False,
        )
        if df.shape[0] > 0 and df.shape[1] > 0:
            return df
    except Exception:
        pass
    # last resort: try pandas autodetect (may be slow)
    try:
        df = pd.read_table(
            path, header=0, index_col=0, comment="#", engine="python", low_memory=False
        )
        if df.shape[0] > 0 and df.shape[1] > 0:
            return df
    except Exception:
        pass
    raise RuntimeError(f"Failed to read table: {path}")


def strip_ensembl_version(index):
    # Remove version suffix after dot (ENSG... .15 -> ENSG...)
    return index.to_series().astype(str).str.split(".").str[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inpath", default=None)
    p.add_argument("--out", dest="outpath", default="data/expression.csv")
    args = p.parse_args()

    inpath = args.inpath
    if inpath is None:
        cand = find_candidate("data")
        if cand is None:
            raise SystemExit("No candidate expression file found in data/")
        inpath = cand

    print("Loading", inpath)
    df = load_table(inpath)

    # Normalize index (remove Ensembl versions)
    orig_index = df.index.astype(str)
    new_index = strip_ensembl_version(df.index)
    df.index = new_index

    # If duplicate gene IDs after stripping version, aggregate by mean
    if df.index.duplicated().any():
        print(
            "Warning: duplicate gene IDs after stripping versions — aggregating by mean"
        )
        df = df.groupby(df.index).mean()

    # Save as comma-separated with gene IDs as first column
    outdir = os.path.dirname(args.outpath)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_csv(args.outpath, index=True)
    print("Saved converted expression to", args.outpath)


if __name__ == "__main__":
    main()
