import logging

import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from config import RANDOM_SEED
from scripts.models.features import build_tfidf_vectorizer

logger = logging.getLogger(__name__)


def _to_binary(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, 0, 1)


def _log_oneclass_metrics(pipeline, texts_val, y_val, prefix: str):
    y_pred_raw = pipeline.predict(texts_val)
    y_scores_raw = pipeline.decision_function(texts_val)
    y_prob = 1 - (y_scores_raw - y_scores_raw.min()) / (y_scores_raw.max() - y_scores_raw.min() + 1e-10)
    y_pred = _to_binary(y_pred_raw)

    if len(set(y_val)) < 2 or len(set(y_pred)) < 2:
        mlflow.log_metric(f"{prefix}_n_val", len(y_val))
        mlflow.log_metric(f"{prefix}_n_pred_positive", int(y_pred.sum()))
        logger.info(f"  {prefix} WARNING: only one class in validation/prediction, skipping metrics")
        return

    if len(set(y_val)) > 1:
        mlflow.log_metric(f"{prefix}_auroc", roc_auc_score(y_val, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    mlflow.log_metrics({
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_fpr": fpr,
    })

    logger.info(f"  {prefix} Accuracy: {accuracy:.4f}")
    logger.info(f"  {prefix} Precision (AI): {precision:.4f}")
    logger.info(f"  {prefix} Recall (AI): {recall:.4f}")
    logger.info(f"  {prefix} F1: {f1:.4f}")
    logger.info(f"  {prefix} FPR: {fpr:.4f}")


def train_oneclass_svm(texts_train, texts_val, y_val,
                       experiment_name: str = "oneclass_svm"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        model_params = {
            "nu": 0.05,
            "kernel": "rbf",
            "gamma": "auto",
        }
        mlflow.log_params({"model_type": "OneClassSVM", **model_params})

        pipeline = Pipeline([
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", OneClassSVM(**model_params)),
        ])
        pipeline.fit(texts_train)

        _log_oneclass_metrics(pipeline, texts_val, y_val, "val")
        mlflow.sklearn.log_model(pipeline, "model", pip_requirements=[])

        return pipeline, run.info.run_id


def train_isolation_forest(texts_train, texts_val, y_val,
                           experiment_name: str = "oneclass_if"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        model_params = {
            "n_estimators": 200,
            "max_samples": "auto",
            "contamination": 0.05,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        mlflow.log_params({"model_type": "IsolationForest", **model_params})

        pipeline = Pipeline([
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", IsolationForest(**model_params)),
        ])
        pipeline.fit(texts_train)

        _log_oneclass_metrics(pipeline, texts_val, y_val, "val")
        mlflow.sklearn.log_model(pipeline, "model", pip_requirements=[])

        return pipeline, run.info.run_id
