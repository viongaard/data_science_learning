from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, f1_score,
                             silhouette_score, davies_bouldin_score)
import numpy as np
import torch
import pandas as pd


def cross_entropy_loss(y_true, y_pred_proba):
    """Оценивает качество модели, используя Cross-Entropy Loss"""

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()

    y_true = torch.tensor(y_true, dtype=torch.int64)
    y_pred_proba = torch.tensor(y_pred_proba, dtype=torch.float32)

    return torch.nn.CrossEntropyLoss()(y_pred_proba, y_true).item()


def zero_one_loss(y_true, y_pred):
    """Оценивает качество модели, используя Zero-One Loss"""

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    y_true = torch.tensor(y_true, dtype=torch.int64)
    y_pred = torch.tensor(y_pred, dtype=torch.int64)

    return torch.mean((y_true != y_pred).float()).item()


def evaluate_clustering(X, labels):
    """Оценивает кластеризацию"""

    metrics = {}
    metrics['Silhouette Score'] = silhouette_score(X, labels)
    metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)

    return metrics


def evaluate_classification(model, X_test, y_test):
    """Оценивает модель классификации по Cross-Entropy Loss и Zero-One Loss."""

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted')
    }
    if y_pred_proba is not None:
        metrics['Cross-Entropy Loss'] = cross_entropy_loss(y_test, y_pred_proba)
    metrics['Zero-One Loss'] = zero_one_loss(y_test, y_pred)

    return metrics


def evaluate_regression(model, X_test, y_test):
    """Оценивает регрессионную модель по RMSE и R^2."""

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {'RMSE': rmse, 'R2': r2}
