import json
import logging

import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

from config import CLASSIFIER_THRESHOLDS

logger = logging.getLogger(__name__)

MAX_FPR = 0.01


def load_jsonl(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["label"])
    return texts, labels


def calibrate_threshold(y_prob, y_true, max_fpr=MAX_FPR):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) == 0:
        return CLASSIFIER_THRESHOLDS[0]
    return float(thresholds[valid[-1]])


def _prob_from_oneclass(pipeline, texts):
    y_pred_raw = pipeline.predict(texts)
    y_scores_raw = pipeline.decision_function(texts)
    y_prob = 1 - (y_scores_raw - y_scores_raw.min()) / (y_scores_raw.max() - y_scores_raw.min() + 1e-10)
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    return y_prob, y_pred


def evaluate_oneclass(pipeline, texts, labels, texts_cal=None, labels_cal=None,
                      experiment_name: str = "", run_id: str = None):
    mlflow.set_experiment(experiment_name)
    cal_suffix = "_cal" if experiment_name else ""

    y_prob, y_pred = _prob_from_oneclass(pipeline, texts)
    results = _compute_metrics(y_prob, labels, prefix="test")

    if texts_cal is not None and labels_cal is not None:
        y_prob_cal, _ = _prob_from_oneclass(pipeline, texts_cal)
        cal_thresh = calibrate_threshold(y_prob_cal, labels_cal)
        y_pred_cal = (y_prob >= cal_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, y_pred_cal, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cal_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        results[f"test_calibrated_threshold"] = cal_thresh
        results[f"test_calibrated_precision"] = prec
        results[f"test_calibrated_recall"] = rec
        results[f"test_calibrated_fpr"] = cal_fpr
        mlflow.log_metrics({
            f"test_calibrated_threshold": cal_thresh,
            f"test_calibrated_precision": prec,
            f"test_calibrated_recall": rec,
            f"test_calibrated_fpr": cal_fpr,
        })

    logger.info(f"\nOne-Class {experiment_name} — Test Results:")
    _print_results(results)

    cm = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    if len(cm) == 4:
        tn, fp, fn, tp = cm
    else:
        tn = fp = fn = tp = 0
    logger.info(f"  Default-threshold Precision (AI): {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Default-threshold Precision: N/A")
    logger.info(f"  Default-threshold FPR:            {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "  Default-threshold FPR: N/A")

    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
        mlflow.log_metrics({f"test_{k}": v for k, v in results.items()})

    return results


def evaluate_baseline(pipeline, texts, labels, texts_cal=None, labels_cal=None,
                      experiment_name: str = "", run_id: str = None):
    mlflow.set_experiment(experiment_name)

    y_prob = pipeline.predict_proba(texts)[:, 1]
    results = _compute_metrics(y_prob, labels, prefix="test")

    if texts_cal is not None and labels_cal is not None:
        y_prob_cal = pipeline.predict_proba(texts_cal)[:, 1]
        cal_thresh = calibrate_threshold(y_prob_cal, labels_cal)
        y_pred_cal = (y_prob >= cal_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, y_pred_cal, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cal_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        results[f"test_calibrated_threshold"] = cal_thresh
        results[f"test_calibrated_precision"] = prec
        results[f"test_calibrated_recall"] = rec
        results[f"test_calibrated_fpr"] = cal_fpr
        mlflow.log_metrics({
            f"test_calibrated_threshold": cal_thresh,
            f"test_calibrated_precision": prec,
            f"test_calibrated_recall": rec,
            f"test_calibrated_fpr": cal_fpr,
        })

    logger.info(f"\nBaseline {experiment_name} — Test Results:")
    _print_results(results)

    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
        mlflow.log_metrics({f"test_{k}": v for k, v in results.items()})
        report = classification_report(
            labels, (y_prob >= 0.5).astype(int),
            target_names=["Human", "AI"],
            output_dict=True,
        )
        mlflow.log_dict(report, "classification_report.json")

    return results


def _compute_metrics(y_prob, y_true, prefix: str = ""):
    metrics = {}
    if len(set(y_true)) > 1:
        metrics[f"{prefix}_auroc"] = float(roc_auc_score(y_true, y_prob))

    for threshold in CLASSIFIER_THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.update({
            f"{prefix}_precision_at_{threshold}": precision,
            f"{prefix}_recall_at_{threshold}": recall,
            f"{prefix}_f1_at_{threshold}": f1,
            f"{prefix}_fpr_at_{threshold}": fpr,
        })
    return metrics


def _print_results(metrics):
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
