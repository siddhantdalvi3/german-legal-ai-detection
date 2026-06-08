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


def train_all(one_class: bool = False):
    train_path = PROCESSED_DIR / "train.jsonl"
    test_path = PROCESSED_DIR / "test.jsonl"

    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}. Run preprocessing first.")
        return

    logger.info("Loading training data...")
    texts, labels = load_jsonl(train_path)

    if one_class:
        human_mask = [l == 0 for l in labels]
        texts_human = [t for t, m in zip(texts, human_mask) if m]
        texts_ai = [t for t, m in zip(texts, human_mask) if not m]
        logger.info(f"One-class mode: {len(texts_human)} human, {len(texts_ai)} AI (AI only for validation)")

        split = int(len(texts_human) * 0.9)
        texts_train = texts_human[:split]
        texts_val_human = texts_human[split:]
        texts_val = texts_val_human + texts_ai
        labels_val = [0] * len(texts_val_human) + [1] * len(texts_ai)

        logger.info(f"Train (human only): {len(texts_train)} sentences, Val (human+AI): {len(texts_val)} sentences")

        from scripts.models.oneclass import train_oneclass_svm, train_isolation_forest

        logger.info("=" * 50)
        logger.info("Training One-Class SVM...")
        logger.info("=" * 50)
        svm_result = train_oneclass_svm(texts_train, texts_val, labels_val)

        logger.info("=" * 50)
        logger.info("Training Isolation Forest...")
        logger.info("=" * 50)
        if_result = train_isolation_forest(texts_train, texts_val, labels_val)

        logger.info("=" * 50)
        logger.info("One-class training complete!")
        logger.info("=" * 50)
        logger.info(f"One-Class SVM run:     {svm_result[1]}")
        logger.info(f"Isolation Forest run:  {if_result[1]}")
        logger.info("=" * 50)
        return

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
