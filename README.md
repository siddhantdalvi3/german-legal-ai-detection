# German AI-Text Detector (Law & Public Administration)

Detect AI-generated German legal and administrative texts with a focus on minimizing false positives.

## Quick Start

```bash
# Setup:
uv sync                    # Install Python deps
uv run python main.py --setup       # Check environment

# Pipeline:
uv run python main.py --mine --fobbe --legal-commons --rii
uv run python main.py --generate --models qwen2.5 --temps 0.3 0.7
uv run python main.py --generate --models mistral --temps 0.3 0.7
uv run python main.py --generate --models gemma4 --temps 0.3
uv run python main.py --generate --models gemma4 --temps 0.7
uv run python main.py --preprocess --fobbe --legal-commons --rii
uv run python main.py --train
uv run python main.py --evaluate

# Inference:
uv run python main.py --predict --text "Die Deutsche Bundesbank wird ermächtigt..."

# Interactive mode:
uv run python main.py --predict

# API server:
uv run python main.py --serve
```

## Project Structure

```
├── main.py                    # Pipeline orchestrator
├── config.py                  # Central configuration
├── scripts/
│   ├── mining*.py             # Per-source miners (Gesetze, Fobbe, Legal Commons, RII, DIP, GESP)
│   ├── generate_ai.py         # Ollama generation (checkpointed)
│   ├── preprocessing.py       # Clean, segment, deduplicate
│   ├── presplit_cache.py      # spaCy pre-split for large sources
│   ├── train.py               # Training orchestrator
│   ├── evaluate.py            # Metrics + MLflow
│   ├── predict.py             # CLI inference
│   ├── serve.py               # FastAPI server
│   └── models/
│       ├── oneclass.py        # OneClassSVM + IsolationForest
│       ├── features.py        # TF-IDF + statistical features
│       ├── baseline.py        # Logistic Regression + Random Forest
│       └── transformer.py     # gbert-base + LoRA
├── docs/
│   ├── architecture.md        # Pipeline + MLflow structure
│   ├── dataset.md             # Sources, preprocessing, current state
│   ├── evaluation.md          # Metrics, thresholds, results
│   └── lit_review/            # 24 papers across 3 topics
├── data/                      # Raw and processed data (gitignored)
└── paper/                     # Conference paper (LaTeX)
```

## Dataset

- **Human**: 13.7M sentences from Legal Commons (76.9%), Fobbe (17.5%), Gesetze (5.1%), RII (2.1%)
- **AI (planned)**: ~450K sentences from qwen2.5:7b, mistral, gemma4:12b at temps 0.3 and 0.7
- **AI ratio**: ~3.2% (evaluation only; one-class methods train on human-only data)
- **All data pre-2022** (`MAX_DATE = "2021-12-31"`)

## Requirements

- Python >= 3.12 (tested on 3.14)
- NVIDIA GPU 16GB+ for AI generation (RTX 4080)
- Ollama for AI text generation (`ollama pull qwen2.5:7b mistral gemma4:12b`)
- ~55 GB disk for full dataset + models

## Key Design Decisions

- **False positives must be minimized** — threshold calibrated for ≥0.9, precision-focused
- **One-class training** — OneClassSVM + IsolationForest trained on human data only
- **All generation on CUDA** — single RTX 4080, 4-night schedule (15.5h/night)
- **spaCy sm > lg** — faster, accurate enough for German legal sentence splitting
- **Pre-2022 data** — ensures human data predates widespread LLM use
- **Experiment tracking**: MLflow with local file store (`mlruns/`)
