# German AI-Text Detector (Law & Public Administration)

Detect AI-generated German legal and administrative texts with a focus on minimizing false positives.

## Quick Start

```bash
# One-shot setup (downloads everything ~55 GB, idempotent):
uv run python download_all.py

# Or step by step:
uv sync                                # Install Python deps
python main.py --mine                  # Download human texts
python main.py --generate              # Generate AI texts (all 15 combos)
python main.py --generate --models mistral qwen2.5   # Selective generation
python main.py --preprocess            # Build dataset
python main.py --train                 # Train models (baseline + gbert)
python main.py --evaluate              # Evaluate + MLflow logging

# Inference:
python main.py --predict --text "Die Deutsche Bundesbank wird ermächtigt..."
python main.py --predict --model lr --threshold 0.9 --text "..."

# Interactive mode:
python main.py --predict

# API server:
python main.py --serve
```

## Project Structure

```
├── main.py                    # Pipeline orchestrator
├── config.py                  # Central configuration
├── download_all.py            # Idempotent download of all assets
├── scripts/
│   ├── mining.py              # Gesetze-im-Internet miner
│   ├── mining_openlegaldata.py# OpenLegalData (HF datasets)
│   ├── mining_bundestag.py    # Bundestag protocols (AJAX endpoint)
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
│   ├── mining.py              # Data storage + logging utilities
│   └── nlp_utils.py           # spaCy, XML cleaning, sentence splitting
├── docs/
│   ├── architecture.md        # Pipeline + MLflow structure
│   ├── dataset.md             # Sources, preprocessing, current state
│   ├── evaluation.md          # Metrics, thresholds, status
│   └── lit_review/            # 24 papers across 3 topics
├── data/                      # Raw and processed data
├── models/                    # Symlink to MLflow artifacts
└── mlruns/                    # MLflow experiment data
```

## Data Sources

- **Human**: 
  - Gesetze-im-Internet (6,120 XMLs) ✓
  - Open Legal Data (HF datasets, 16.9 GB, needs HF login + accepted conditions)
- **AI**: 5 models × 3 temperatures (0.3, 0.7, 1.0)
  - Ollama: qwen2.5:7b, gemma3:12b, gemma4, mistral
  - MLX: Mistral-7B-Instruct-v0.3-4bit (Mistral-7B-Instruct-v0.3-4bit)
  - Target: 200 sentences per combo (`SENTENCES_PER_COMBINATION` in config.py)
  - Current: 10 of 15 combos complete

## Key Design Decisions

- **Precision-focused**: threshold calibration sweep at 0.5–0.95
- **Hard set**: 200 curated Grundgesetz/BVerfG excerpts for FP validation
- **Stepwise features**: TF-IDF first, then statistical (perplexity, burstiness, lexical diversity)
- **Models**: Logistic Regression + Random Forest (baselines, sklearn Pipelines), gbert-base + LoRA (advanced)
- **Experiment tracking**: MLflow with local file store (`mlruns/`)
- **MLX**: Python API with module-level model cache (not subprocess), temperature via `make_sampler(temp=...)`
- **Selective generation**: `--models mistral qwen2.5 mlx` to run specific models
- **Leo-mistral**: Dropped — not available on Ollama

## Requirements

- Python >= 3.12 (tested on 3.14, `/opt/homebrew/bin/python3.14`)
- Apple Silicon (M3 Pro recommended for MPS GPU acceleration, 18 GB RAM)
- Ollama for AI text generation (`ollama pull qwen2.5 gemma3 gemma4 mistral`)
- HuggingFace account for OpenLegalData dataset (`hf auth login`)
- ~55 GB disk for full dataset + models

## Optional: Generate More AI Data

Adjust `SENTENCES_PER_COMBINATION` in `config.py` (default: 200). Run specific models:

```bash
python main.py --generate --models mistral qwen2.5 mlx
```

List available models:

```bash
python main.py --list-models
```
