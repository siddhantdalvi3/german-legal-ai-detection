#!/usr/bin/env python3.14
import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from utils.mining import logger, setup_logging

LEVELS = {
    "--mine": "Mining: Downloading human text sources...",
    "--generate": "Generation: Creating AI text corpus...",
    "--preprocess": "Preprocessing: Building dataset...",
    "--train": "Training: Training models...",
    "--evaluate": "Evaluation: Evaluating models...",
    "--predict": "Inference: Making predictions...",
    "--serve": "Serving: Starting API server...",
    "--setup": "Setup: Checking environment...",
    "--all": "Full Pipeline: Running all stages...",
}


def stage(func):
    def wrapper(*args, **kwargs):
        logger.info(f"{'='*60}")
        logger.info(f"STAGE: {func.__name__.replace('_', ' ').title()}")
        logger.info(f"{'='*60}")
        return func(*args, **kwargs)
    return wrapper


@stage
def setup_environment():
    logger.info("Checking Python version...")
    v = sys.version_info
    assert v.major == 3 and v.minor >= 12, f"Need Python >= 3.12, got {v.major}.{v.minor}"

    logger.info("Checking spaCy model...")
    try:
        import spacy
        spacy.load("de_core_news_sm")
        logger.info("  ✓ de_core_news_sm loaded")
    except OSError:
        logger.warning("  ✗ de_core_news_sm not found, will download")
        subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"])

    logger.info("Checking Ollama...")
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        logger.info("  ✓ Ollama found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("  ✗ Ollama not found. Some AI generation features may be unavailable.")

    logger.info("Environment check complete!")


@stage
def mine(use_openlegaldata: bool = False, use_rii: bool = False, fobbe_datasets: list[str] | None = None, use_legal_commons: bool = False, use_dip: bool = False, use_gesp: bool = False):
    from scripts.mining import Miner
    miner = Miner()
    miner.mine_gesetze_im_internet()

    if use_openlegaldata:
        from scripts.mining_openlegaldata import mine_openlegaldata
        try:
            mine_openlegaldata()
        except Exception as e:
            logger.warning(f"OpenLegalData mining failed (optional): {e}")

    if use_rii:
        from scripts.mining_rii import mine_rii
        try:
            mine_rii()
        except Exception as e:
            logger.warning(f"RII mining failed: {e}")

    if fobbe_datasets is not None:
        from scripts.mining_fobbe import extract_court_decisions
        try:
            extract_court_decisions(datasets=fobbe_datasets or None)
        except Exception as e:
            logger.warning(f"Fobbe mining failed: {e}")

    if use_legal_commons:
        from scripts.mining_legal_commons import extract_court_decisions
        try:
            extract_court_decisions()
        except Exception as e:
            logger.warning(f"Legal Commons mining failed: {e}")

    if use_dip:
        from scripts.mining_dip_bundestag import mine_dip_bundestag
        try:
            mine_dip_bundestag()
        except Exception as e:
            logger.warning(f"DIP Bundestag mining failed: {e}")

    if use_gesp:
        from scripts.mining_gesp import mine_gesp
        try:
            mine_gesp()
        except Exception as e:
            logger.warning(f"GESP mining failed: {e}")

    logger.info("All mining complete!")


@stage
def generate_ai(models: list[str] | None = None, temps: list[float] | None = None):
    from scripts.generate_ai import generate_ai_corpus
    generate_ai_corpus(models, temps)


@stage
def preprocess(use_openlegaldata: bool = False, use_rii: bool = False, use_fobbe: bool = False, use_legal_commons: bool = False, use_dip: bool = False, use_gesp: bool = False):
    from scripts.preprocessing import build_dataset
    build_dataset(use_openlegaldata=use_openlegaldata, use_rii=use_rii, use_fobbe=use_fobbe, use_legal_commons=use_legal_commons, use_dip=use_dip, use_gesp=use_gesp)


@stage
def train(one_class: bool = False):
    from scripts.train import train_all
    train_all(one_class=one_class)


@stage
def evaluate(one_class: bool = False):
    import json
    from config import PROCESSED_DIR
    from scripts.evaluate import (
        load_jsonl, evaluate_baseline, evaluate_oneclass
    )
    from scripts.models.baseline import train_logistic_regression, train_random_forest

    train_path = PROCESSED_DIR / "train.jsonl"
    test_path = PROCESSED_DIR / "test.jsonl"

    if not test_path.exists():
        logger.error("No test set found. Run --preprocess first.")
        return

    texts_test, labels_test = load_jsonl(test_path)

    if one_class:
        texts_train, labels_train = load_jsonl(train_path)
        human_mask = [l == 0 for l in labels_train]
        texts_train_human = [t for t, m in zip(texts_train, human_mask) if m]
        split = int(len(texts_train_human) * 0.9)
        texts_val_human = texts_train_human[split:]
        texts_val = texts_val_human + [t for t, m in zip(texts_train, human_mask) if not m][:5000]
        labels_val = [0] * len(texts_val_human) + [1] * len([t for t, m in zip(texts_train, human_mask) if not m][:5000])

        from scripts.models.oneclass import train_oneclass_svm, train_isolation_forest

        logger.info("Retraining One-Class SVM for evaluation...")
        svm_model, svm_id = train_oneclass_svm(texts_train_human, texts_val, labels_val)
        evaluate_oneclass(svm_model, texts_test, labels_test,
                          experiment_name="oneclass_svm", run_id=svm_id)

        logger.info("Retraining Isolation Forest for evaluation...")
        if_model, if_id = train_isolation_forest(texts_train_human, texts_val, labels_val)
        evaluate_oneclass(if_model, texts_test, labels_test,
                          experiment_name="oneclass_if", run_id=if_id)

        logger.info("One-class evaluation complete!")
        return

    texts_train, labels_train = load_jsonl(train_path)

    split = int(len(texts_train) * 0.9)
    texts_val, labels_val = texts_train[split:], labels_train[split:]
    texts_train_part, labels_train_part = texts_train[:split], labels_train[:split]

    logger.info("Retraining LR for evaluation...")
    lr_model, lr_id = train_logistic_regression(
        texts_train_part, labels_train_part, texts_val, labels_val
    )
    evaluate_baseline(lr_model, texts_test, labels_test,
                      experiment_name="baseline_lr", run_id=lr_id)

    logger.info("Retraining RF for evaluation...")
    rf_model, rf_id = train_random_forest(
        texts_train_part, labels_train_part, texts_val, labels_val
    )
    evaluate_baseline(rf_model, texts_test, labels_test,
                      experiment_name="baseline_rf", run_id=rf_id)

    logger.info("Evaluation complete!")


@stage
def predict(args):
    from scripts.predict import PredictionPipeline
    pipeline = PredictionPipeline(model_type=args.model, threshold=args.threshold)

    if args.text:
        result = pipeline.predict(args.text)
        import json as j
        print(j.dumps(result, indent=2, ensure_ascii=False))
    else:
        logger.info("Interactive mode. Enter text (Ctrl+D to quit):")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    result = pipeline.predict(text)
                    print(f"  → {result['label']} (confidence: {result['confidence']:.4f})")
        except (EOFError, KeyboardInterrupt):
            print()


@stage
def serve():
    import uvicorn
    from scripts.serve import app
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="German AI-Text Detector Pipeline")
    args = parser.parse_known_args()[0]

    parser.add_argument("--setup", action="store_true", help="Check environment")
    parser.add_argument("--mine", action="store_true", help="Mine human text sources (Gesetze-im-Internet only)")
    parser.add_argument("--openlegaldata", action="store_true", help="Include OpenLegalData (ODbL) in mining/preprocessing")
    parser.add_argument("--rii", action="store_true", help="Include Rechtsprechung-im-Internet (gov open data) in mining/preprocessing")
    parser.add_argument("--fobbe", nargs="*", default=None,
                        help="Include Fobbe datasets (CC0) in mining/preprocessing. "
                             "Specify names: bverwg bpatg bgh_strafsachen, or all if omitted")
    parser.add_argument("--legal-commons", action="store_true",
                        help="Include CUI03/german-commons Legal Commons in mining/preprocessing")
    parser.add_argument("--dip", action="store_true",
                        help="Include DIP Bundestag (Drucksachen + Plenarprotokolle) in mining/preprocessing")
    parser.add_argument("--gesp", action="store_true",
                        help="Include state court decisions via gesp in mining/preprocessing")
    parser.add_argument("--generate", action="store_true", help="Generate AI text corpus")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models for generation (e.g. --models mistral qwen2.5 mlx)")
    parser.add_argument("--temps", nargs="*", type=float, default=None,
                        help="Temperatures to use (e.g. --temps 0.1 0.3 0.7)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available generation models")
    parser.add_argument("--sources", action="store_true",
                        help="Show detailed data source report")
    parser.add_argument("--preprocess", action="store_true", help="Build dataset")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--one-class", action="store_true", help="Train one-class models (OC-SVM, Isolation Forest) on human data only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all models")
    parser.add_argument("--predict", nargs="?", const=True, help="Make predictions")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--text", type=str, help="Text for prediction")
    parser.add_argument("--file", type=str, help="File for batch prediction")
    parser.add_argument("--model", type=str, default="lr",
                        choices=["lr", "rf", "gbert"])
    parser.add_argument("--threshold", type=float, default=0.9)

    args = parser.parse_args()

    if args.sources:
        from scripts.report_sources import report_sources
        report_sources()
        return

    if args.list_models:
        from config import AVAILABLE_MODELS
        logger.info("Available generation models:")
        for key, info in AVAILABLE_MODELS.items():
            logger.info(f"  {key:12s}  {info['type']:6s}  {info['desc']}")
        return

    if args.setup:
        setup_environment()
    if args.mine:
        mine(use_openlegaldata=args.openlegaldata, use_rii=args.rii, fobbe_datasets=args.fobbe, use_legal_commons=args.legal_commons, use_dip=args.dip, use_gesp=args.gesp)
    if args.generate:
        generate_ai(args.models, args.temps)
    if args.preprocess:
        preprocess(use_openlegaldata=args.openlegaldata, use_rii=args.rii, use_fobbe=args.fobbe is not None, use_legal_commons=args.legal_commons, use_dip=args.dip, use_gesp=args.gesp)
    if args.train:
        train(one_class=args.one_class)
    if args.evaluate:
        evaluate(one_class=args.one_class)
    if args.serve:
        serve()

    if args.all:
        logger.info("Starting full pipeline...")
        setup_environment()
        mine(use_openlegaldata=args.openlegaldata, use_rii=args.rii, fobbe_datasets=args.fobbe, use_legal_commons=args.legal_commons, use_dip=args.dip, use_gesp=args.gesp)
        generate_ai(args.models, args.temps)
        preprocess(use_openlegaldata=args.openlegaldata, use_rii=args.rii, use_fobbe=args.fobbe is not None, use_legal_commons=args.legal_commons, use_dip=args.dip, use_gesp=args.gesp)
        train(one_class=args.one_class)
        evaluate(one_class=args.one_class)

    has_input = args.text or args.file
    if args.predict is not None or has_input:
        predict(args)

    if not any([args.setup, args.mine, args.generate, args.preprocess,
                args.train, args.evaluate, args.serve, args.all, args.list_models,
                args.sources,
                args.predict is not None, has_input,
                args.openlegaldata, args.rii, args.fobbe is not None, args.legal_commons,
                args.dip, args.gesp, args.one_class]):
        parser.print_help()


if __name__ == "__main__":
    main()
