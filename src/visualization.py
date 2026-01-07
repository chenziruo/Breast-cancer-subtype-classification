#!/usr/bin/env python3
"""Visualization utilities for pipeline results.

生成并保存：
- 混淆矩阵（基于训练后模型在全数据上的预测）
- 特征重要性（RandomForest）
- 每类 ROC 曲线（若模型支持 predict_proba）
- 选中特征热图
"""
from __future__ import annotations
import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils import (
    load_expression,
    load_metadata,
    align_data,
    preprocess_expression,
    plot_confusion,
    plot_feature_heatmap,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_feature_importances(model, feature_names, outpath, top_n=20):
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ attribute; skipping importance plot")
        return
    fi = model.feature_importances_
    idx = np.argsort(fi)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = fi[idx]
    plt.figure(figsize=(8, max(4, top_n * 0.25)))
    plt.barh(range(len(names))[::-1], vals, color="C0")
    plt.yticks(range(len(names)), names[::-1])
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_roc_multiclass(clf, X, y, classes, outpath):
    # requires predict_proba
    if not hasattr(clf, "predict_proba"):
        print(
            "Model has no predict_proba; skipping ROC plot for",
            getattr(clf, "__class__", clf),
        )
        return
    # compute probabilities on given data
    y_bin = label_binarize(y, classes=range(len(classes)))
    prob = clf.predict_proba(X)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(classes):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_roc_from_scores(y, prob, classes, outpath):
    """Plot multiclass ROC given true labels `y` and probability matrix `prob`.

    `prob` should be shape (n_samples, n_classes) matching `classes` order.
    """
    y_bin = label_binarize(y, classes=range(len(classes)))
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(classes):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def visualize_all(
    expr_path,
    meta_path,
    selected_path,
    model_dir,
    outdir,
    label_col="Subtype",
    transpose=False,
):
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "plots"))

    expr = load_expression(expr_path, transpose=transpose)
    meta = load_metadata(meta_path)
    expr, meta = align_data(expr, meta)

    # prepare labels
    if label_col not in meta.columns:
        raise ValueError(f"label column {label_col} not found in metadata")
    labels_raw = meta[label_col].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(labels_raw.values)
    classes = le.classes_.tolist()

    # preprocess expression same as pipeline
    expr_proc = preprocess_expression(expr, min_variance=0.0, log_transform=True)

    # read selected features
    sel = pd.read_csv(selected_path, header=0).iloc[:, 0].astype(str).tolist()
    sel_present = [s for s in sel if s in expr_proc.index]
    if len(sel_present) == 0:
        raise ValueError("No selected features present in expression matrix")

    X = expr_proc.loc[sel_present, :].T.values  # samples x features

    # iterate models
    for model_name in ["random_forest", "svc"]:
        model_path = os.path.join(model_dir, f"model_{model_name}.joblib")
        if not os.path.exists(model_path):
            print("Model not found:", model_path)
            continue
        clf = joblib.load(model_path)

        # predictions on full data (for visualization only)
        y_pred = clf.predict(X)

        # confusion matrix
        cm = confusion_matrix(y, y_pred, labels=range(len(classes)))
        plot_confusion(
            cm, classes, os.path.join(outdir, "plots", f"confusion_{model_name}.png")
        )

        # classification report -> save
        report = classification_report(
            y, y_pred, target_names=classes, output_dict=True
        )
        with open(
            os.path.join(outdir, f"report_{model_name}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # feature importance (if available)
        try:
            plot_feature_importances(
                clf,
                sel_present,
                os.path.join(outdir, "plots", f"feature_importance_{model_name}.png"),
            )
        except Exception as e:
            print("Failed to plot feature importances for", model_name, e)

        # ROC — use cross-validated probabilities to avoid evaluating on training data
        if hasattr(clf, "predict_proba"):
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                y_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
                plot_roc_from_scores(
                    y,
                    y_score,
                    classes,
                    os.path.join(outdir, "plots", f"roc_cv_{model_name}.png"),
                )
            except Exception as e:
                print("Failed to compute CV ROC for", model_name, e)
        else:
            print("Model has no predict_proba; skipping ROC for", model_name)

    # heatmap of selected features
    try:
        plot_feature_heatmap(
            expr_proc,
            sel_present,
            labels_raw.values,
            os.path.join(outdir, "plots", "features_heatmap.png"),
        )
    except Exception as e:
        print("Failed to plot feature heatmap", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expr", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--selected", required=True)
    p.add_argument("--model-dir", default="results/preprocessed_run")
    p.add_argument("--outdir", default="results/visualization")
    p.add_argument("--label", default="Subtype")
    p.add_argument("--transpose", action="store_true")
    args = p.parse_args()

    visualize_all(
        args.expr,
        args.meta,
        args.selected,
        args.model_dir,
        args.outdir,
        label_col=args.label,
        transpose=args.transpose,
    )


if __name__ == "__main__":
    main()
