import json
import logging
from pathlib import Path

import mlflow
import torch
from transformers import AutoTokenizer

from config import (
    CLASSIFIER_THRESHOLDS,
    DEFAULT_THRESHOLD,
    BERT_MODEL,
    BERT_MAX_LENGTH,
    get_device,
)

logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self, model_type: str = "lr", threshold: float = DEFAULT_THRESHOLD):
        self.model_type = model_type
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.model_type == "gbert":
            self._load_gbert()
        else:
            self._load_sklearn()

    def _load_sklearn(self):
        from mlflow.sklearn import load_model

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(
            f"baseline_{self.model_type}"
        )
        if experiment:
            runs = client.search_runs(
                experiment.experiment_id,
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                self.model = load_model(model_uri)
                logger.info(f"Loaded {self.model_type} pipeline from MLflow run {run_id}")
                return

        logger.warning(f"No MLflow run found for baseline_{self.model_type}")
        raise RuntimeError(f"Model not found: baseline_{self.model_type}")

    def _load_gbert(self):
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification

        if Path("./models/gbert_lora_best").exists():
            model_path = "./models/gbert_lora_best"
        else:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("gbert_lora")
            if not experiment:
                raise RuntimeError("No gbert_lora experiment found")
            runs = client.search_runs(
                experiment.experiment_id,
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if not runs:
                raise RuntimeError("No gbert_lora runs found")
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/model"
            model_path = mlflow.artifacts.download_artifacts(
                model_uri, dst_path="./models/gbert_mlflow"
            )

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if Path(model_path, "tokenizer.json").exists() else BERT_MODEL
        )

        if hasattr(self.model, "peft_config"):
            self.model = PeftModel.from_pretrained(self.model, model_path)

        device = get_device()
        self.model.to(device)
        self.model.eval()
        logger.info(f"Loaded gbert model on {device} from {model_path}")

    def predict(self, text: str) -> dict:
        if self.model_type == "gbert":
            return self._predict_gbert(text)
        return self._predict_sklearn(text)

    def _predict_sklearn(self, text: str) -> dict:
        y_prob = self.model.predict_proba([text])[0, 1]

        probs = {}
        for t in CLASSIFIER_THRESHOLDS:
            probs[f"ai_prob_at_{t}"] = float(y_prob)
            probs[f"label_at_{t}"] = "AI" if y_prob >= t else "Human"

        return {
            "label": "AI" if y_prob >= self.threshold else "Human",
            "confidence": float(y_prob) if y_prob >= self.threshold else float(1 - y_prob),
            "ai_probability": float(y_prob),
            "human_probability": float(1 - y_prob),
            "threshold_used": self.threshold,
            **probs,
        }

    def _predict_gbert(self, text: str) -> dict:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            y_prob = probabilities[0, 1].item()

        probs = {}
        for t in CLASSIFIER_THRESHOLDS:
            probs[f"ai_prob_at_{t}"] = float(y_prob)
            probs[f"label_at_{t}"] = "AI" if y_prob >= t else "Human"

        return {
            "label": "AI" if y_prob >= self.threshold else "Human",
            "confidence": float(y_prob) if y_prob >= self.threshold else float(1 - y_prob),
            "ai_probability": float(y_prob),
            "human_probability": float(1 - y_prob),
            "threshold_used": self.threshold,
            **probs,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="German AI-Text Detector")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument(
        "--model", type=str, default="lr",
        choices=["lr", "rf", "gbert"],
        help="Model type (default: lr)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Classification threshold (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    pipeline = PredictionPipeline(model_type=args.model, threshold=args.threshold)

    if args.text:
        result = pipeline.predict(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            texts = [line.strip() for line in f if line.strip()]
        results = pipeline.predict_batch(texts)
        for text, result in zip(texts, results):
            print(f"{text[:80]:80s} → {result['label']:5s} (conf={result['confidence']:.3f})")

    else:
        print("Interactive mode. Enter text (Ctrl+D to quit):")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    result = pipeline.predict(text)
                    print(
                        f"  → {result['label']} "
                        f"(confidence: {result['confidence']:.4f}, "
                        f"AI prob: {result['ai_probability']:.4f})"
                    )
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
