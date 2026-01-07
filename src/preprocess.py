#!/usr/bin/env python3
"""Preprocess expression and metadata for the BRCA pipeline.
功能：
- 读取表达矩阵（行=基因，列=样本）和样本元数据（包含 Subtype）
- 自动检测是否为原始 counts 并执行 CPM->log2 或直接 log2(x+1)
- 过滤低信息基因（高比例零或零方差）
- 对基因按行做 z-score 标准化
- 将对齐后的预处理结果保存到 `data/` 目录

输出文件：
- `data/expression_preprocessed.csv`：log-normalized 并过滤后的表达矩阵
- `data/expression_scaled.csv`：按基因 z-score 标准化后的矩阵
- `data/metadata_preprocessed.csv`：与表达矩阵对齐的元数据（索引 = 样本条码）
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd


def try_read_expr(path: str) -> pd.DataFrame:
    # try TSV then CSV
    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep, header=0, index_col=0, engine="python")
            if df.shape[0] > 0 and df.shape[1] > 0:
                return df
        except Exception:
            continue
    # last resort
    df = pd.read_table(path, header=0, index_col=0, engine="python")
    return df


def try_read_meta(path: str) -> pd.DataFrame:
    # accept tsv or csv
    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep, header=0, engine="python")
            if df.shape[0] > 0:
                return df
        except Exception:
            continue
    df = pd.read_csv(path, header=0, engine="python")
    return df


def detect_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.values
    # ignore NaN when computing percentiles
    try:
        p99 = float(np.nanpercentile(vals, 99))
    except Exception:
        p99 = float(np.nanmax(vals))
    print(f"[normalize] 99th percentile = {p99:.3f}")
    if p99 > 1000:
        # counts -> CPM -> log2(CPM+1)
        print(
            "[normalize] Detected raw counts (p99>1000). Converting to CPM then log2(CPM+1)."
        )
        col_sums = df.sum(axis=0)
        # avoid division by zero
        col_sums = col_sums.replace(0, np.nan)
        cpm = df.div(col_sums, axis=1) * 1e6
        out = np.log2(cpm + 1)
        return out
    elif p99 > 50:
        print("[normalize] Values moderately large (p99>50). Applying log2(x+1).")
        return np.log2(df + 1)
    else:
        print("[normalize] Values look like log-scale (p99<=50). No log applied.")
        return df


def filter_genes(df: pd.DataFrame, max_zero_frac: float = 0.8) -> pd.DataFrame:
    # remove genes with too many zeros and zero variance
    zero_frac = (df == 0).sum(axis=1) / float(df.shape[1])
    keep_mask = zero_frac <= max_zero_frac
    kept = df.loc[keep_mask]
    var = kept.var(axis=1)
    kept = kept.loc[var > 0]
    print(f"[filter] genes kept {kept.shape[0]} / {df.shape[0]}")
    return kept


def zscore_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=1)
    stds = df.std(axis=1).replace(0, np.nan)
    z = df.sub(means, axis=0).div(stds, axis=0)
    z = z.fillna(0)
    return z


def align_samples(
    expr: pd.DataFrame, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr_samples = [str(x) for x in expr.columns]
    meta_index = [str(x) for x in meta.index]
    common = [s for s in expr_samples if s in meta_index]
    if not common:
        # try match by patient (first 12 chars)
        meta_map = {s[:12]: s for s in meta_index}
        common = [s for s in expr_samples if s[:12] in meta_map]
        if common:
            mapped = [meta_map[s[:12]] for s in common]
            expr = expr.loc[:, common]
            meta = meta.loc[mapped]
            meta.index = common  # align indices to expression samples
            return expr, meta
        else:
            raise SystemExit("No matching samples between expression and metadata")
    else:
        expr = expr.loc[:, common]
        meta = meta.loc[common]
        return expr, meta


def save_outputs(
    expr_filt: pd.DataFrame, expr_z: pd.DataFrame, meta: pd.DataFrame, outdir: str
):
    os.makedirs(outdir, exist_ok=True)
    p1 = os.path.join(outdir, "expression_preprocessed.csv")
    p2 = os.path.join(outdir, "expression_scaled.csv")
    p3 = os.path.join(outdir, "metadata_preprocessed.csv")
    expr_filt.to_csv(p1)
    expr_z.to_csv(p2)
    meta.to_csv(p3)
    print(f"[save] wrote {p1}, {p2}, {p3}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expr", default="data/expression.csv")
    p.add_argument("--meta", default="data/metadata.csv")
    p.add_argument("--outdir", default="data")
    p.add_argument("--max-zero-frac", type=float, default=0.8)
    args = p.parse_args()

    if not os.path.exists(args.expr):
        print("Expression file not found:", args.expr)
        sys.exit(1)
    if not os.path.exists(args.meta):
        print("Metadata file not found:", args.meta)
        sys.exit(1)

    expr = try_read_expr(args.expr)
    expr = expr.apply(pd.to_numeric, errors="coerce")

    meta = try_read_meta(args.meta)
    # prefer a 'sample' column
    if "sample" in meta.columns:
        meta = meta.set_index("sample")
    elif meta.index.name is None:
        meta = meta.set_index(meta.columns[0])

    print(f"[info] expr shape {expr.shape}, meta shape {meta.shape}")

    expr_aligned, meta_aligned = align_samples(expr, meta)
    print(
        f"[info] aligned samples: {expr_aligned.shape[1]} samples, {expr_aligned.shape[0]} genes"
    )

    expr_norm = detect_and_normalize(expr_aligned)
    expr_filt = filter_genes(expr_norm, max_zero_frac=args.max_zero_frac)
    expr_z = zscore_per_gene(expr_filt)

    save_outputs(expr_filt, expr_z, meta_aligned, args.outdir)

    # brief stats
    print("[summary]")
    print(" samples:", expr_aligned.shape[1])
    print(" genes (after filter):", expr_filt.shape[0])


if __name__ == "__main__":
    main()
