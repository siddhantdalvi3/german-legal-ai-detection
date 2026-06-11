import logging

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from config import CLASSIFIER_THRESHOLDS, RANDOM_SEED
from scripts.models.features import build_tfidf_vectorizer

logger = logging.getLogger(__name__)


def train_logistic_regression(texts_train, y_train, texts_val, y_val,
                               experiment_name: str = "baseline_lr"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        model_params = {
            "max_iter": 1000,
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
        }
        mlflow.log_params({"model_type": "LogisticRegression", **model_params})

        pipeline = Pipeline([
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", LogisticRegression(**model_params)),
        ])
        pipeline.fit(texts_train, y_train)

        _log_metrics(pipeline, texts_val, y_val, "val")
        mlflow.sklearn.log_model(pipeline, "model")

        return pipeline, run.info.run_id


def train_random_forest(texts_train, y_train, texts_val, y_val,
                         experiment_name: str = "baseline_rf"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        model_params = {
            "n_estimators": 200,
            "max_depth": 50,
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        mlflow.log_params({"model_type": "RandomForest", **model_params})

        pipeline = Pipeline([
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", RandomForestClassifier(**model_params)),
        ])
        pipeline.fit(texts_train, y_train)

        _log_metrics(pipeline, texts_val, y_val, "val")
        mlflow.sklearn.log_model(pipeline, "model")

        return pipeline, run.info.run_id


def _log_metrics(pipeline, texts_val, y_val, prefix: str):
    y_prob = pipeline.predict_proba(texts_val)[:, 1]

    for threshold in CLASSIFIER_THRESHOLDS:
        y_t = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_t, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        mlflow.log_metrics({
            f"{prefix}_precision_at_{threshold}": precision,
            f"{prefix}_recall_at_{threshold}": recall,
            f"{prefix}_f1_at_{threshold}": f1,
            f"{prefix}_specificity_at_{threshold}": specificity,
            f"{prefix}_fpr_at_{threshold}": fpr,
        })

    mlflow.log_metrics({
        f"{prefix}_accuracy": pipeline.score(texts_val, y_val),
    })


def train_baseline_pipeline(texts_train, labels_train, texts_val, labels_val):
    logger.info("Training Logistic Regression pipeline...")
    lr_pipeline, lr_run_id = train_logistic_regression(
        texts_train, labels_train, texts_val, labels_val
    )

    logger.info("Training Random Forest pipeline...")
    rf_pipeline, rf_run_id = train_random_forest(
        texts_train, labels_train, texts_val, labels_val
    )

    logger.info("Baseline training complete")
    return {"lr": (lr_pipeline, lr_run_id), "rf": (rf_pipeline, rf_run_id)}
