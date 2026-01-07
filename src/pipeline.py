'''
pipeline 的 Docstring

用于训练和评估基于表达数据的分类模型。

'''
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from utils import load_expression, load_metadata, align_data, preprocess_expression, save_json, plot_confusion, plot_feature_heatmap


def feature_select_f_test(expr_df, labels, top_n=200):
    # expr_df: genes x samples
    X = expr_df.values.T
    y = labels
    F, p = f_classif(X, y)
    idx = np.argsort(F)[::-1][:top_n]
    genes = expr_df.index[idx].tolist()
    return genes


def l1_selection(expr_df, labels, C=0.1, max_iter=5000):
    X = expr_df.T.values
    y = labels
    lr = LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=max_iter, multi_class='ovr')
    lr.fit(X, y)
    coef = np.abs(lr.coef_).sum(axis=0)
    keep_idx = np.where(coef > 1e-6)[0]
    genes = expr_df.index[keep_idx].tolist()
    return genes


def train_and_evaluate(X, y, labels_unique, outdir):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'svc': SVC(probability=True, kernel='rbf', random_state=42)
    }

    for name, clf in models.items():
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        report = classification_report(y, y_pred, target_names=labels_unique, output_dict=True)
        cm = confusion_matrix(y, y_pred, labels=range(len(labels_unique)))
        results[name] = {'accuracy': acc, 'macro_f1': f1, 'report': report}
        # save confusion matrix plot
        plot_confusion(cm, labels_unique, os.path.join(outdir, f'plots/cm_{name}.png'))
        # train final model on full data and save
        clf.fit(X, y)
        joblib.dump(clf, os.path.join(outdir, f'model_{name}.joblib'))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', required=True)
    parser.add_argument('--meta', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--top-n', type=int, default=200)
    parser.add_argument('--transpose', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'plots'), exist_ok=True)

    expr = load_expression(args.expr, transpose=args.transpose)
    meta = load_metadata(args.meta)
    expr, meta = align_data(expr, meta)

    label_col = args.label
    if label_col not in meta.columns:
        raise ValueError(f'label column {label_col} not found in metadata')

    labels_raw = meta[label_col].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(labels_raw.values)
    labels_unique = le.classes_.tolist()

    expr_proc = preprocess_expression(expr, min_variance=0.0, log_transform=True)

    # Feature selection: f-test then L1
    top_genes = feature_select_f_test(expr_proc, y, top_n=args.top_n)
    expr_top = expr_proc.loc[top_genes, :]
    l1_genes = l1_selection(expr_top, y, C=0.1)

    # If L1 returns empty, fall back to top_genes
    if len(l1_genes) == 0:
        selected = top_genes
    else:
        # keep intersection in original order
        selected = [g for g in top_genes if g in l1_genes]
        if len(selected) == 0:
            selected = l1_genes

    # save selected features
    pd.Series(selected).to_csv(os.path.join(args.outdir, 'selected_features.csv'), index=False, header=['gene'])

    X = expr_proc.loc[selected, :].T.values

    results = train_and_evaluate(X, y, labels_unique, args.outdir)

    save = {
        'selected_count': len(selected),
        'selected_features': selected,
        'cv_results_summary': results,
        'label_mapping': {int(v): k for v, k in enumerate(labels_unique)}
    }
    # save metrics
    import json
    with open(os.path.join(args.outdir, 'metrics_cv.json'), 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=2)

    # heatmap of selected features
    try:
        plot_feature_heatmap(expr_proc, selected, labels_raw.values, os.path.join(args.outdir, 'plots', 'features_heatmap.png'))
    except Exception:
        pass

    print('Pipeline finished. Results in', args.outdir)


if __name__ == '__main__':
    main()
