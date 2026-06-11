import logging

import mlflow
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from config import BERT_MODEL, BERT_MAX_LENGTH, CLASSIFIER_THRESHOLDS, RANDOM_SEED, get_device, supports_fp16

logger = logging.getLogger(__name__)

device = get_device()
logger.info(f"Transformer using device: {device}")


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=BERT_MAX_LENGTH,
    )


def prepare_dataset(texts, labels):
    data = Dataset.from_dict({"text": texts, "label": labels})
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = data.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return data, tokenizer


def create_lora_model(num_labels: int = 2):
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=num_labels)
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL, config=config
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))
    y_prob = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
    metrics = {}

    for threshold in CLASSIFIER_THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
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
f"precision_at_{threshold}": precision,
f"recall_at_{threshold}": recall,
f"f1_at_{threshold}": f1,
f"specificity_at_{threshold}": specificity,
f"fpr_at_{threshold}": fpr,
        })

    return metrics


def train_gbert(texts_train, labels_train, texts_val, labels_val,
                experiment_name: str = "gbert_lora"):
    mlflow.set_experiment(experiment_name)

    logger.info(f"Preparing gbert datasets: {len(texts_train)} train, {len(texts_val)} val")
    logger.info(f"Loading tokenizer: {BERT_MODEL}")
    train_dataset, tokenizer = prepare_dataset(texts_train, labels_train)
    val_dataset, _ = prepare_dataset(texts_val, labels_val)
    logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    logger.info("Creating LoRA model (gbert-base)...")
    model = create_lora_model()

    training_args = TrainingArguments(
        output_dir="./checkpoints/gbert_lora",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="precision_at_0.9",
        greater_is_better=True,
        fp16=supports_fp16(),
        dataloader_num_workers=2,
        report_to="mlflow",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": BERT_MODEL,
            "max_length": BERT_MAX_LENGTH,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "learning_rate": 2e-5,
            "epochs": 3,
            "batch_size": 8,
            "grad_accum": 4,
        })

        logger.info("Starting gbert training (3 epochs, early stopping patience=2)...")
        trainer.train()

        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({
            f"val_{k}": v for k, v in eval_metrics.items()
        })

        logger.info("gbert training complete, saving model...")
        model_path = "./models/gbert_lora_best"
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        mlflow.log_artifacts(model_path, artifact_path="gbert_lora_model")

        logger.info(f"gbert model saved. Run ID: {run.info.run_id}")
        return model, tokenizer, run.info.run_id


def predict_gbert(model, tokenizer, text: str) -> tuple[int, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=BERT_MAX_LENGTH,
    ).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        y_prob = probabilities[0, 1].item()
        y_pred = 1 if y_prob >= 0.5 else 0

    return y_pred, y_prob
