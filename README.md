# German AI-Text Detector (Law & Public Administration)

Detect AI-generated German legal and administrative texts with a focus on minimizing false positives.

## Installation

### Prerequisites

- **Python 3.12+** (tested on 3.14)
- **uv** (Python package manager) — install via `pip install uv` or `brew install uv`
- **Ollama** (for AI text generation) — https://ollama.com/download
- **~55 GB free disk space** for full dataset + models

### Step 1: Clone & Setup

```bash
git clone https://github.com/siddhantdalvi3/german-legal-ai-detection.git
cd german-legal-ai-detection
uv sync
```

### Step 2: Download spaCy Model

```bash
uv run python -m spacy download de_core_news_sm
```

### Step 3: Pull Ollama Models (for AI Generation)

```bash
ollama pull qwen2.5:7b
ollama pull mistral
ollama pull gemma4:12b
```

### Step 4: Mine Human Text Sources

Downloads ~17 GB of German legal texts from multiple public sources:

```bash
uv run python main.py --mine --fobbe --legal-commons --rii
```

### Step 5: Generate AI Texts

Runs on your available hardware (GPU recommended). Generation time depends on your system:

```bash
uv run python main.py --generate --models qwen2.5 --temps 0.3 0.7
uv run python main.py --generate --models mistral --temps 0.3 0.7
uv run python main.py --generate --models gemma4 --temps 0.3
uv run python main.py --generate --models gemma4 --temps 0.7
```

On a single RTX 4080, this takes ~3 days (15.5h/night). Checkpointing allows stopping and resuming.

### Step 6: Preprocess & Train

```bash
uv run python main.py --preprocess --fobbe --legal-commons --rii
uv run python main.py --train
uv run python main.py --evaluate
```

### Inference

```bash
# Single text
uv run python main.py --predict --text "Die Deutsche Bundesbank wird ermächtigt..."

# Interactive mode
uv run python main.py --predict

# API server
uv run python main.py --serve
```

## Project Structure

```
├── main.py                    # Pipeline orchestrator
├── config.py                  # Central configuration
├── scripts/
│   ├── mining*.py             # Per-source miners
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
├── data/                      # Raw and processed data (gitignored)
└── paper/                     # Conference paper (LaTeX)
```

## Dataset

- **Human**: 13.7M sentences from Legal Commons (76.9%), Fobbe (17.5%), Gesetze (5.1%), RII (2.1%)
- **AI**: ~450K sentences from qwen2.5:7b, mistral, gemma4:12b at temps 0.3 and 0.7
- **AI ratio**: ~3.2% (evaluation only; one-class methods train on human-only data)
- **All data pre-2022** (`MAX_DATE = "2021-12-31"`)

## Requirements

- **Python >= 3.12**
- **Ollama** for AI generation
- **GPU recommended** for generation (any NVIDIA/AMD/macOS GPU with 16GB+ VRAM). CPU fallback works but is significantly slower.
- **~55 GB disk** for full dataset + models

## Hardware Notes

- **AI generation** benefits significantly from GPU acceleration. Tested on RTX 4080 (CUDA), M3 Pro (MPS), and CPU.
- **Training** (Logistic Regression, Random Forest, OneClassSVM) runs on CPU — no GPU needed.
- **spaCy** uses `de_core_news_sm` (small model) for speed; results are comparable to the large model for sentence splitting.

## Key Design Decisions

- **False positives must be minimized** — threshold calibrated for ≥0.9, precision-focused
- **One-class training** — OneClassSVM + IsolationForest trained on human data only
- **spaCy sm > lg** — faster, accurate enough for German legal sentence splitting
- **Pre-2022 data** — ensures human data predates widespread LLM use
- **Experiment tracking**: MLflow with local file store (`mlruns/`)
