"""
ML helpers for agents: feature prep and simple model training using scikit-learn.

Functions:
- prepare_features(df, feature_cols, label_col) -> (X, y)
- train_random_forest_classifier(X, y, **kwargs) -> model
- train_test_split_and_eval(df, feature_cols, label_col, test_size=0.2, random_state=42)

Notes:
- Designed for quick iteration; agents can swap or extend with other models.
"""
from __future__ import annotations

from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def prepare_features(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df_clean = df.dropna(subset=feature_cols + [label_col]).copy()
    X = df_clean[feature_cols].values
    y = df_clean[label_col].values
    return X, y


def train_random_forest_classifier(X: np.ndarray, y: np.ndarray, **kwargs) -> RandomForestClassifier:
    model = RandomForestClassifier(**kwargs)
    model.fit(X, y)
    return model


def train_test_split_and_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    rf_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rf_kwargs = rf_kwargs or {}
    X, y = prepare_features(df, feature_cols, label_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = train_random_forest_classifier(X_train, y_train, **rf_kwargs)
    preds = model.predict(X_test)
    proba = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
    }
    if proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass
    return {
        'model': model,
        'metrics': metrics,
        'features': feature_cols,
        'label': label_col,
    }
