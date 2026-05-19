# German AI-Text Detector (Law & Public Administration)

Detect AI-generated German legal and administrative texts with a focus on minimizing false positives.

## Quick Start

```bash
# Setup
python3.14 -m venv .venv
source .venv/bin/activate
uv sync

# Run full pipeline
python main.py --all

# Or step by step:
python main.py --mine          # Download human texts
python main.py --generate      # Generate AI texts (Ollama + MLX)
python main.py --preprocess    # Build dataset
python main.py --train         # Train models
python main.py --evaluate      # Evaluate + MLflow logging

# Inference
python main.py --predict --text "Die Deutsche Bundesbank wird ermächtigt..."
python predict.py --model lr --threshold 0.9 --text "..."

# Interactive mode
python predict.py

# API server
python main.py --serve
```

## Project Structure

```
├── main.py                    # Pipeline orchestrator
├── config.py                  # Central configuration
├── scripts/
│   ├── mining.py              # Gesetze-im-Internet miner
│   ├── mining_openlegaldata.py
│   ├── mining_bundestag.py
│   ├── generate_ai.py         # Ollama + MLX generation (checkpointed)
│   ├── preprocessing.py       # Clean, segment, deduplicate
│   ├── train.py               # Training orchestrator
│   ├── evaluate.py            # Metrics + MLflow
│   ├── predict.py             # CLI inference
│   ├── serve.py               # FastAPI server
│   └── models/
│       ├── features.py        # TF-IDF + statistical features
│       ├── baseline.py        # Logistic Regression + Random Forest
│       └── transformer.py     # gbert-base + LoRA
├── utils/
│   ├── mining.py              # Data storage utilities
│   └── nlp_utils.py           # spaCy, cleaning helpers
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── evaluation.md
│   └── lit_review/            # 15-20 paper review
├── data/                      # Raw and processed data
├── models/                    # Symlink to MLflow artifacts
└── mlruns/                    # MLflow experiment data
```

## Data Sources

- **Human**: Gesetze-im-Internet, Open Legal Data, Bundestag protocols (2006-2026)
- **AI**: Ollama (qwen2.5, gemma3, gemma4, mistral, leo-mistral) + MLX (Mistral-7B)
  - Temperature sweep: 0.3, 0.7, 1.0
  - Target: 450K AI sentences from 18 model×temp combinations

## Key Design Decisions

- **Precision-focused**: threshold calibration at 0.5, 0.7, 0.8, 0.9, 0.95
- **Hard set**: 200 curated Grundgesetz/BVerfG excerpts for FP validation
- **Stepwise features**: TF-IDF first, then statistical (perplexity, burstiness, lexical diversity)
- **Models**: Logistic Regression + Random Forest (baselines), gbert-base + LoRA (advanced)
- **Experiment tracking**: MLflow with local file store

## Requirements

- Python >= 3.14
- Apple Silicon (M-series) recommended for MPS GPU acceleration
- Ollama for AI text generation
- ~50 GB disk for full dataset + models
