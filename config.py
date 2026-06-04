import os
import platform
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.absolute()


def get_device() -> str:
    """Auto-detect best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_fp16() -> bool:
    """fp16 is beneficial on CUDA; not reliable on MPS."""
    return get_device() == "cuda"


def is_macos() -> bool:
    return platform.system() == "Darwin"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

GESETZE_DIR = DATA_DIR / "gesetze_im_internet"
OPENLEGALDATA_DIR = DATA_DIR / "openlegaldata"
OPENLEGALDATA_HF_DIR = DATA_DIR / "openlegaldata_hf"
RII_DIR = DATA_DIR / "rii"
FOBBE_DIR = DATA_DIR / "fobbe"
LEGAL_COMMONS_DIR = DATA_DIR / "legal_commons"
AI_GENERATED_DIR = DATA_DIR / "ai_generated"

GESETZE_TOC_URL = "https://www.gesetze-im-internet.de/gii-toc.xml"
OPENLEGALDATA_DUMP_URL = "https://static.openlegaldata.io/dumps/latest/"

AVAILABLE_MODELS = {
    "qwen2.5": {"type": "ollama", "name": "qwen2.5:7b", "desc": "Qwen 2.5 7B"},
    "qwen3": {"type": "ollama", "name": "qwen3:14b", "desc": "Qwen 3 14B"},
    "gemma3": {"type": "ollama", "name": "gemma3:12b", "desc": "Gemma 3 12B"},
    "gemma4": {"type": "ollama", "name": "gemma4:12b", "desc": "Gemma 4 12B"},
    "mistral": {"type": "ollama", "name": "mistral", "desc": "Mistral 7B v0.3"},
    "llama3.1": {"type": "ollama", "name": "llama3.1:8b", "desc": "Llama 3.1 8B"},
    "deepseek": {"type": "ollama", "name": "deepseek-r1:7b", "desc": "DeepSeek R1 7B"},
    "phi4": {"type": "ollama", "name": "phi4:14b", "desc": "Phi-4 14B"},
    "mlx": {
        "type": "mlx",
        "name": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "desc": "MLX Mistral 7B Instruct 4bit",
    },
    "mlx_gemma4": {
        "type": "mlx",
        "name": "jedisct1/gemma-4-12B-it-txt-mlx-8bit",
        "desc": "MLX Gemma 4 12B Instruct 8bit (text-only)",
    },
}

OLLAMA_MODELS = [m["name"] for m in AVAILABLE_MODELS.values() if m["type"] == "ollama"]
DEFAULT_GENERATION_MODELS = list(OLLAMA_MODELS)
MLX_MODEL = next(m["name"] for m in AVAILABLE_MODELS.values() if m["type"] == "mlx")
MLX_VLM_MODEL = next((m["name"] for m in AVAILABLE_MODELS.values() if m["type"] == "mlx_vlm"), None)

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
SENTENCES_PER_COMBINATION = 20_000
AI_TARGET = 1_320_000

HUMAN_TARGET = 500_000
DATASET_TARGET = 1_000_000

SPACY_MODEL = "de_core_news_sm"
BERT_MODEL = "deepset/gbert-base"
BERT_MAX_LENGTH = 256

CLASSIFIER_THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]
DEFAULT_THRESHOLD = 0.9

RANDOM_SEED = 42
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
N_FOLDS = 5

os.environ["TOKENIZERS_PARALLELISM"] = "false"
