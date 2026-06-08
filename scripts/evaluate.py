import json
import logging

import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PROCESSED_DIR, CLASSIFIER_THRESHOLDS

logger = logging.getLogger(__name__)


def load_jsonl(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["label"])
    return texts, labels


def evaluate_oneclass(pipeline, texts, labels,
                      experiment_name: str, run_id: str = None):
    mlflow.set_experiment(experiment_name)
    y_pred_raw = pipeline.predict(texts)
    y_scores_raw = pipeline.decision_function(texts)

    y_prob = 1 - (y_scores_raw - y_scores_raw.min()) / (y_scores_raw.max() - y_scores_raw.min() + 1e-10)
    y_pred = np.where(y_pred_raw == 1, 0, 1)

    results = _compute_metrics(y_prob, labels, prefix="test")
    logger.info(f"\nOne-Class {experiment_name} — Test Results:")
    _print_results(results)

    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
    logger.info(f"  Default-threshold Precision (AI): {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Default-threshold Precision: N/A")
    logger.info(f"  Default-threshold FPR:            {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "  Default-threshold FPR: N/A")

    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
        mlflow.log_metrics({f"test_{k}": v for k, v in results.items()})

    return results


def evaluate_baseline(pipeline, texts, labels,
                      experiment_name: str, run_id: str = None):
    mlflow.set_experiment(experiment_name)
    y_prob = pipeline.predict_proba(texts)[:, 1]

    results = _compute_metrics(y_prob, labels, prefix="test")
    logger.info(f"\nBaseline {experiment_name} — Test Results:")
    _print_results(results)

    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
        mlflow.log_metrics({
            f"test_{k}": v for k, v in results.items()
        })
        report = classification_report(
            labels, (y_prob >= 0.5).astype(int),
            target_names=["Human", "AI"],
            output_dict=True,
        )
        mlflow.log_dict(report, "classification_report.json")

    return results


def evaluate_gbert(model, tokenizer, texts, labels,
                   experiment_name: str = "gbert_lora", run_id: str = None):
    from scripts.models.transformer import predict_gbert

    mlflow.set_experiment(experiment_name)
    y_prob = []
    for text in texts:
        _, prob = predict_gbert(model, tokenizer, text)
        y_prob.append(prob)
    y_prob = np.array(y_prob)

    results = _compute_metrics(y_prob, labels, prefix="test")
    logger.info(f"\nGBERT LoRA — Test Results:")
    _print_results(results)

    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
        mlflow.log_metrics({
            f"test_{k}": v for k, v in results.items()
        })

    return results


def evaluate_hard_set(model_predict_fn, texts, labels,
                      threshold: float = 0.9):
    y_prob = model_predict_fn(texts)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

    logger.info(f"\n{'='*50}")
    logger.info(f"HARD SET EVALUATION (threshold={threshold})")
    logger.info(f"{'='*50}")
    logger.info(f"  True Negatives (correctly Human):  {tn}")
    logger.info(f"  False Positives (Human→AI ERROR):  {fp}")
    logger.info(f"  False Negatives (AI→Human error):  {fn}")
    logger.info(f"  True Positives (correctly AI):     {tp}")
    logger.info(f"\n  Precision (AI class): {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "N/A")
    logger.info(f"  False Positive Rate:  {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "N/A")

    return {"fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn)}


def _compute_metrics(y_prob, y_true, prefix: str = ""):
    metrics = {}
    for threshold in CLASSIFIER_THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.update({
            f"{prefix}_precision_at_{threshold}": precision,
            f"{prefix}_recall_at_{threshold}": recall,
            f"{prefix}_f1_at_{threshold}": f1,
            f"{prefix}_specificity_at_{threshold}": specificity,
            f"{prefix}_fpr_at_{threshold}": fpr,
        })
    return metrics


def _print_results(metrics):
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
