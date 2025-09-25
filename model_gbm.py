from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def build_gbm(n_estimators: int = 150,
              learning_rate: float = 0.05,
              max_depth: int = 3,
              subsample: float = 0.9,
              random_state: int = 42,
              early_stop: bool = False,
              n_iter_no_change: int = 10,
              validation_fraction: float = 0.1) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
        validation_fraction=validation_fraction if early_stop else 0.1,
        n_iter_no_change=n_iter_no_change if early_stop else None,
        tol=1e-4
    )
    return model

def evaluate(model, X_test, y_test, pos_label=1) -> Dict[str, float]:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "tn": int(confusion_matrix(y_test, preds)[0,0]),
        "fp": int(confusion_matrix(y_test, preds)[0,1]),
        "fn": int(confusion_matrix(y_test, preds)[1,0]),
        "tp": int(confusion_matrix(y_test, preds)[1,1]),
    }

def cv_check(model, X_train, y_train, cv_splits: int = 5) -> float:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
    return float(np.mean(scores))