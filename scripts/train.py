import json
import logging

from config import PROCESSED_DIR
from scripts.models.baseline import train_baseline_pipeline
from scripts.models.transformer import train_gbert

logger = logging.getLogger(__name__)


def load_jsonl(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["label"])
    return texts, labels


def train_all():
    train_path = PROCESSED_DIR / "train.jsonl"
    test_path = PROCESSED_DIR / "test.jsonl"

    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}. Run preprocessing first.")
        return

    logger.info("Loading training data...")
    texts, labels = load_jsonl(train_path)

    split = int(len(texts) * 0.9)
    texts_train, labels_train = texts[:split], labels[:split]
    texts_val, labels_val = texts[split:], labels[split:]

    logger.info(f"Train: {len(texts_train)} sentences, Val: {len(texts_val)} sentences")

    logger.info("=" * 50)
    logger.info("Training baseline models...")
    logger.info("=" * 50)
    baseline_results = train_baseline_pipeline(
        texts_train, labels_train, texts_val, labels_val
    )

    logger.info("=" * 50)
    logger.info("Training gbert-base + LoRA...")
    logger.info("=" * 50)
    gbert_result = train_gbert(texts_train, labels_train, texts_val, labels_val)

    logger.info("=" * 50)
    logger.info("All training complete!")
    logger.info("=" * 50)
    logger.info(f"Baseline LR run:   {baseline_results['lr'][1]}")
    logger.info(f"Baseline RF run:   {baseline_results['rf'][1]}")
    logger.info(f"GBERT LoRA run:    {gbert_result[2]}")
    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all()
