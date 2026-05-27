import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

GESETZE_DIR = DATA_DIR / "gesetze_im_internet"
OPENLEGALDATA_DIR = DATA_DIR / "openlegaldata"
AI_GENERATED_DIR = DATA_DIR / "ai_generated"

GESETZE_TOC_URL = "https://www.gesetze-im-internet.de/gii-toc.xml"
OPENLEGALDATA_DUMP_URL = "https://static.openlegaldata.io/dumps/latest/"

AVAILABLE_MODELS = {
    "qwen2.5": {"type": "ollama", "name": "qwen2.5:7b", "desc": "Qwen 2.5 7B"},
    "gemma3": {"type": "ollama", "name": "gemma3:12b", "desc": "Gemma 3 12B"},
    "gemma4": {"type": "ollama", "name": "gemma4", "desc": "Gemma 4 (optional ~15 GB)"},
    "mistral": {"type": "ollama", "name": "mistral", "desc": "Mistral 7B v0.3"},
    "mlx": {
        "type": "mlx",
        "name": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "desc": "MLX Mistral 7B Instruct 4bit",
    },
    "mlx_gemma4": {
        "type": "mlx_vlm",
        "name": "mlx-community/gemma-4-31b-it-4bit",
        "desc": "MLX Gemma 4 31b Instruct 4bit (VLM)",
    },
}

RUN_OPTIONAL = True
OPTIONAL_MODELS = [
    m["name"] for m in AVAILABLE_MODELS.values() if "optional" in m["desc"].lower()
]
OLLAMA_MODELS = [m["name"] for m in AVAILABLE_MODELS.values() if m["type"] == "ollama"]
DEFAULT_GENERATION_MODELS = [m for m in OLLAMA_MODELS if m not in OPTIONAL_MODELS] + (
    OPTIONAL_MODELS if RUN_OPTIONAL else []
)
MLX_MODEL = next(m["name"] for m in AVAILABLE_MODELS.values() if m["type"] == "mlx")
MLX_VLM_MODEL = next(m["name"] for m in AVAILABLE_MODELS.values() if m["type"] == "mlx_vlm")

TEMPERATURES = [0.3, 0.7, 1.0]
SENTENCES_PER_COMBINATION = 500  # 9 combos (3 Ollama models × 3 temps) × 500 = 4,500 AI; total ~444K with Gesetze
AI_TARGET = 4_500

HUMAN_TARGET = 500_000
DATASET_TARGET = 1_000_000

SPACY_MODEL = "de_core_news_lg"
BERT_MODEL = "deepset/gbert-base"
BERT_MAX_LENGTH = 256

CLASSIFIER_THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]
DEFAULT_THRESHOLD = 0.9

RANDOM_SEED = 42
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
N_FOLDS = 5

os.environ["TOKENIZERS_PARALLELISM"] = "false"
