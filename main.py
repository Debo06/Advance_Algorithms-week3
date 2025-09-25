from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from model_gbm import build_gbm, evaluate, cv_check

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"]) if "target" in ds.frame else ds.data
    y = ds.target
    return X, y

def targeted_eda(X: pd.DataFrame, y: pd.Series) -> None:
    # Example histogram for a single feature
    feature = "mean radius" if "mean radius" in X.columns else X.columns[0]
    plt.figure()
    plt.hist(X[feature].values, bins=30)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title("Histogram: " + feature)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "hist_radius_mean.png"))
    plt.close()

    # Correlation heatmap using matplotlib only
    corr = X.corr(numeric_only=True)
    plt.figure(figsize=(6,5))
    im = plt.imshow(corr.values, aspect='auto')
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "corr_heatmap.png"))
    plt.close()

def plot_roc(y_true, y_prob, path):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_confusion(cm, path):
    plt.figure()
    plt.imshow(cm, cmap=None)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_feature_importance(model, feature_names, path):
    importances = model.named_steps['gbm'].feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    names = np.array(feature_names)[idx]
    vals = importances[idx]
    plt.figure(figsize=(7,5))
    plt.barh(range(len(vals)), vals[::-1])
    plt.yticks(range(len(vals)), names[::-1], fontsize=7)
    plt.xlabel("Importance")
    plt.title("Top Feature Importances (GBM)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Week 3 GBM — Disease Diagnosis")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--early-stop", action="store_true")
    args = ap.parse_args()

    X, y = load_data()
    targeted_eda(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    pre = build_preprocessor(X_train)
    gbm = build_gbm(n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    subsample=args.subsample,
                    random_state=args.seed,
                    early_stop=args.early_stop)

    pipe = Pipeline(steps=[("prep", pre), ("gbm", gbm)])
    # CV check on training set
    cv_auc = cv_check(pipe, X_train, y_train, cv_splits=5)

    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)
    print("Hold-out metrics:", json.dumps(metrics, indent=2))
    print(f"Stratified 5-fold CV ROC-AUC (train): {cv_auc:.3f}")

    # Plots
    from sklearn.metrics import confusion_matrix
    y_prob = pipe.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))
    plot_roc(y_test, y_prob, os.path.join(FIG_DIR, "roc_curve.png"))
    plot_confusion(cm, os.path.join(FIG_DIR, "confusion_matrix.png"))

    # Feature importance + names after ColumnTransformer
    # Derive feature names from preprocessor
    prep = pipe.named_steps['prep']
    feature_names = []
    if hasattr(prep, 'transformers_'):
        for name, trans, cols in prep.transformers_:
            if name == 'cat' and hasattr(trans, 'get_feature_names_out'):
                fn = list(trans.get_feature_names_out(cols))
            elif name == 'num' and cols != 'drop':
                fn = list(cols)
            else:
                fn = []
            feature_names.extend(fn)
    else:
        feature_names = list(X_train.columns)

    plot_feature_importance(pipe, feature_names, os.path.join(FIG_DIR, "feature_importance.png"))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"holdout": metrics, "cv_auc_train": cv_auc}, f, indent=2)

if __name__ == "__main__":
    main()