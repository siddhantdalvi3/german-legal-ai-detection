# German AI-Text Detector (Law & Public Administration)

Detect AI-generated German legal and administrative texts with a focus on minimizing false positives.

## Quick Start

```bash
# Setup:
uv sync                                         # Install Python deps
python main.py --setup                          # Check environment

# Pipeline:
python main.py --mine                           # Download Gesetze-im-Internet
python main.py --mine --openlegaldata           # + OpenLegalData (optional, needs HF auth)
python main.py --generate                       # Generate AI texts (all models)
python main.py --generate --models qwen2.5      # Selective generation
python main.py --preprocess                     # Build dataset (Gesetze + AI)
python main.py --train                          # Train baseline models (LR + RF)
python main.py --evaluate                       # Evaluate + MLflow logging

# Inference:
python main.py --predict --text "Die Deutsche Bundesbank wird ermächtigt..."
python main.py --predict --model rf --threshold 0.9 --text "..."

# Interactive mode:
python main.py --predict

# API server:
python main.py --serve
```

## Key Results

| Model | Threshold | Precision | Recall | FPR | Hard Set FP (0 FPs ideal) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9 | **97.2%** | 88.5% | 0.23% | 19/152 |
| **Random Forest** | **0.9** | **99.9%** | 11.9% | **0.00%** | **0/152** |

**RF @ thr=0.9** is the production recommendation for false-positive minimization.

## Project Structure

```
├── main.py                    # Pipeline orchestrator
├── config.py                  # Central configuration
├── download_all.py            # Idempotent download of all assets
├── scripts/
│   ├── mining.py              # Gesetze-im-Internet miner
│   ├── mining_openlegaldata.py# OpenLegalData (HF datasets)
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
│   ├── evaluation.md          # Metrics, thresholds, results
│   └── lit_review/            # 24 papers across 3 topics
├── tests/
│   └── hard_set.jsonl         # 152 human legal sentences for FP validation
├── data/                      # Raw and processed data (gitignored)
└── mlruns/                    # MLflow experiment data (gitignored)
```

## Dataset

- **Human**: 6,119 Gesetze-im-Internet XMLs → 707,714 sentences
- **AI**: 9,783 paragraphs from qwen2.5:7b (6,546), gemma3:12b (1,500), MLX Mistral (1,500), gemma4 (partial) → 66,628 sentences after preprocessing
- **Total**: 774,342 sentences (91.4% Human, 8.6% AI)
- **Average sentence length**: 25 words
- **OpenLegalData**: Optional (`--openlegaldata` flag)

## Requirements

- Python >= 3.12 (tested on 3.14, `/opt/homebrew/bin/python3.14`)
- Apple Silicon (M3 Pro, 18 GB RAM) for MPS GPU acceleration
- Ollama for AI text generation (`ollama pull qwen2.5 gemma3 gemma4`)
- ~55 GB disk for full dataset + models

## Key Design Decisions

- **500 paragraphs per model×temp combo**: Final choice after real generation rates measured (~10.6 s/sent). See `docs/dataset.md` for full sizing analysis.
- **Precision-focused**: threshold calibration sweep at 0.5–0.95.
- **Hard set**: 152 curated human legal sentences for FP validation.
- **Models**: Logistic Regression + Random Forest (sklearn Pipelines), gbert-base + LoRA configured but not viable on MPS (~75 hrs estimated).
- **RF @ thr=0.9 achieves 0 false positives** on the hard set.
- **Experiment tracking**: MLflow with local file store (`mlruns/`).
- **gbert-base tokenizer/model**: Requires `BertTokenizer` directly (not `AutoTokenizer`) due to missing fast tokenizer serialization; requires `BertConfig`/`BertForSequenceClassification` (not `Auto*`) due to missing `model_type` field.
