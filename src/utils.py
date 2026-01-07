'''
utils 的 Docstring
提供用于加载、预处理表达数据和元数据的实用函数，
以及用于保存结果和绘制图形的辅助功能。
'''
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_expression(path, transpose=False):
    df = pd.read_csv(path, index_col=0)
    if transpose:
        df = df.T
    return df


def load_metadata(path, sample_id_col=None):
    df = pd.read_csv(path, index_col=0)
    return df


def align_data(expr: pd.DataFrame, meta: pd.DataFrame):
    samples = list(set(expr.columns) & set(meta.index))
    expr = expr.loc[:, samples]
    meta = meta.loc[samples, :]
    return expr, meta


def preprocess_expression(expr: pd.DataFrame, min_variance=0.0, log_transform=True):
    mat = expr.copy()
    if log_transform:
        mat = np.log2(mat + 1)
    # remove zero-variance genes
    variances = mat.var(axis=1)
    if min_variance > 0:
        keep = variances > min_variance
        mat = mat.loc[keep, :]
    return mat


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion(cm, labels, outpath, figsize=(6, 5)):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_feature_heatmap(expr, features, meta_labels, outpath, max_genes=50):
    feat = features[:max_genes]
    sub = expr.loc[feat, :]
    # z-score by gene
    sub_z = (sub - sub.mean(axis=1).values.reshape(-1, 1)) / (sub.std(axis=1).values.reshape(-1, 1) + 1e-8)
    # order samples by label
    order = np.argsort(meta_labels)
    plt.figure(figsize=(min(14, len(sub.columns) / 5 + 4), max(6, len(feat) / 3)))
    sns.heatmap(sub_z.iloc[:, order], cmap='vlag', center=0, yticklabels=True, xticklabels=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
