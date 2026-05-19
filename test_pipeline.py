#!/usr/bin/env python3.14
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")

sys.path.insert(0, str(Path(__file__).parent))

TEMP_DIR = Path(tempfile.mkdtemp(prefix="ai_detector_test_"))
logger.info(f"Test directory: {TEMP_DIR}")


def test_00_environment():
    logger.info("=" * 60)
    logger.info("TEST: Environment check")
    logger.info("=" * 60)
    assert sys.version_info.major == 3 and sys.version_info.minor >= 14
    logger.info("  ✓ Python >= 3.14")

    import spacy
    nlp = spacy.load("de_core_news_lg")
    doc = nlp("Die Deutsche Bundesbank wird ermächtigt.")
    assert len(doc) > 0
    logger.info("  ✓ spaCy de_core_news_lg loaded")

    from peft import LoraConfig
    logger.info("  ✓ PEFT (LoRA) available")

    from transformers import AutoModelForSequenceClassification
    logger.info("  ✓ Transformers available")

    import mlflow
    logger.info("  ✓ MLflow available")

    from sklearn.linear_model import LogisticRegression
    logger.info("  ✓ scikit-learn available")

    logger.info("")


def test_01_mining():
    logger.info("=" * 60)
    logger.info("TEST: Mining (download 1 law to temp dir)")
    logger.info("=" * 60)

    import config
    config.DATA_DIR = TEMP_DIR
    config.GESETZE_DIR = TEMP_DIR / "gesetze_im_internet"
    config.PROCESSED_DIR = TEMP_DIR / "processed"

    import importlib
    import utils.mining as um
    importlib.reload(um)
    from utils.mining import save_all_data_item_files, get_module_data

    from scripts.mining import Miner
    miner = Miner()
    miner.mine_gesetze_im_internet(count=1)

    xml_files = list(TEMP_DIR.glob("gesetze_im_internet/**/*.xml"))
    logger.info(f"  XML files found: {len(xml_files)}")
    assert len(xml_files) >= 1, f"No XML files in {TEMP_DIR / 'gesetze_im_internet'}"
    logger.info(f"  ✓ Mining works: {xml_files[0].name}")

    from utils.nlp_utils import extract_text_from_xml
    text = xml_files[0].read_text(encoding="utf-8", errors="replace")
    paragraphs = extract_text_from_xml(text)
    logger.info(f"  Paragraphs extracted: {len(paragraphs)}")
    assert len(paragraphs) >= 1
    logger.info("")


def test_02_preprocessing():
    logger.info("=" * 60)
    logger.info("TEST: Preprocessing pipeline")
    logger.info("=" * 60)

    xml_files = list(TEMP_DIR.glob("gesetze_im_internet/**/*.xml"))
    assert len(xml_files) >= 1
    text = xml_files[0].read_text(encoding="utf-8", errors="replace")

    from utils.nlp_utils import extract_text_from_xml, sentence_segment
    paragraphs = extract_text_from_xml(text)
    assert len(paragraphs) >= 1
    logger.info(f"  Paragraphs: {len(paragraphs)}")

    sentences = sentence_segment(" ".join(paragraphs[:3]))
    logger.info(f"  Sentences from first 3 paragraphs: {len(sentences)}")

    from scripts.preprocessing import deduplicate
    records = [
        {"text": p, "label": 0, "source": "gesetze_test"}
        for p in paragraphs[:10]
    ]
    ai_records = [
        {"text": (
            "Die Voraussetzungen einer wirksamen Willenserklärung im Bürgerlichen Recht "
            "sind in den §§ 116 ff. BGB geregelt. Eine Willenserklärung liegt vor, wenn "
            "der Erklärende seinen Willen bewusst und gewollt nach außen kundtut."
        ), "label": 1, "source": "ai_test"}
    ]
    all_recs = deduplicate(records + ai_records)
    logger.info(f"  After dedup: {len(all_recs)} records")

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(2, 5), analyzer="char", max_features=1000)
    X = vec.fit_transform([r["text"] for r in all_recs])
    logger.info(f"  TF-IDF shape: {X.shape}")
    assert X.shape[0] > 0
    logger.info("")


def test_03_baseline_training():
    logger.info("=" * 60)
    logger.info("TEST: Baseline model training")
    logger.info("=" * 60)

    import mlflow
    mlflow.set_tracking_uri(f"file://{TEMP_DIR}/mlruns")

    texts = [
        "Die Deutsche Bundesbank wird ermächtigt, zum Gedenken an die Deutsche Mark.",
        "Die Gestaltung der Wert- und Bildseite ist mit Ausnahme der Umschrift identisch.",
        "Die 1-DM-Goldmünzen sind gesetzliches Zahlungsmittel bis zum 31. Dezember 2001.",
        "Die Stiftung gibt sich im Einvernehmen mit der Deutschen Bundesbank eine Satzung.",
        "Die Voraussetzungen der Willenserklärung sind im BGB geregelt. Der Erklärende muss seinen Willen bewusst kundtun. Die Rechtsfolgen sind in den §§ 116 ff. BGB normiert.",
        "Die Haftung des Verkäufers für Sachmängel bestimmt sich nach § 437 BGB. Der Käufer kann Nacherfüllung verlangen.",
        "Die Anfechtung eines Verwaltungsakts setzt Rechtswidrigkeit voraus. Die Behörde muss innerhalb eines Jahres zurücknehmen.",
        "Die Verhältnismäßigkeit erfordert eine Güterabwägung. Der Eingriff muss geeignet, erforderlich und angemessen sein.",
    ]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    split = 6
    from scripts.models.baseline import train_logistic_regression
    model, run_id = train_logistic_regression(
        texts[:split], labels[:split], texts[split:], labels[split:]
    )

    y_prob = model.predict_proba(texts[split:])[:, 1]
    logger.info(f"  Predictions: {y_prob.tolist()}")
    assert len(y_prob) == 2
    logger.info("  ✓ Baseline training works")
    logger.info("")


def test_04_inference():
    logger.info("=" * 60)
    logger.info("TEST: Inference pipeline")
    logger.info("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vec = TfidfVectorizer(ngram_range=(2, 5), analyzer="char", max_features=100)
    X = vec.fit_transform([
        "Human text about legal matters and the law. This is written by a person.",
        "AI generated content about legal matters and regulations. This is synthetic text.",
    ])
    model = LogisticRegression(max_iter=100)
    model.fit(X, [0, 1])

    X_test = vec.transform(["Die Deutsche Bundesbank wird ermächtigt, das Gesetz auszuführen."])
    prob = model.predict_proba(X_test)[0, 1]
    label = "AI" if prob >= 0.5 else "Human"
    logger.info(f"  → {label} (AI probability: {prob:.4f})")
    logger.info("  ✓ Inference works")
    logger.info("")


def test_99_cleanup():
    import shutil
    shutil.rmtree(TEMP_DIR)
    logger.info(f"Cleaned up {TEMP_DIR}")


if __name__ == "__main__":
    tests = [
        test_00_environment,
        test_01_mining,
        test_02_preprocessing,
        test_03_baseline_training,
        test_04_inference,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"  ✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1

    logger.info("=" * 60)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if passed == len(tests):
        test_99_cleanup()
    else:
        logger.info(f"Test artifacts preserved at: {TEMP_DIR}")

    sys.exit(1 if failed > 0 else 0)
