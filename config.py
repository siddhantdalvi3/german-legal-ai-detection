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

GESETZE_DIR = DATA_DIR / "gesetze_im_internet"
OPENLEGALDATA_DIR = DATA_DIR / "openlegaldata"
RII_DIR = DATA_DIR / "rii"
FOBBE_DIR = DATA_DIR / "fobbe"
LEGAL_COMMONS_DIR = DATA_DIR / "legal_commons"
DIP_DIR = DATA_DIR / "dip_bundestag"
GESP_DIR = DATA_DIR / "gesp"
AI_GENERATED_DIR = DATA_DIR / "ai_generated"

AVAILABLE_MODELS = {
    "qwen2.5": {"type": "ollama", "name": "qwen2.5:7b", "desc": "Qwen 2.5 7B"},
    "qwen3": {"type": "ollama", "name": "qwen3:30b", "desc": "Qwen 3 30B (MoE)"},
    "gemma3": {"type": "ollama", "name": "gemma3:27b", "desc": "Gemma 3 27B"},
    "gemma4": {"type": "ollama", "name": "gemma4:12b", "desc": "Gemma 4 12B"},
    "gemma4-ctx": {"type": "ollama", "name": "gemma4-ctx", "desc": "Gemma 4 12B (ctx=2048)"},
    "mistral": {"type": "ollama", "name": "mistral-small:24b", "desc": "Mistral Small 24B"},
    "deepseek": {"type": "ollama", "name": "deepseek-r1:70b", "desc": "DeepSeek R1 70B"},
    "deepseek-ctx": {"type": "ollama", "name": "deepseek-ctx", "desc": "DeepSeek R1 70B (ctx=2048)"},
    "phi4": {"type": "ollama", "name": "phi4:14b", "desc": "Phi-4 14B"},
    "phi4-ctx": {"type": "ollama", "name": "phi4-ctx", "desc": "Phi-4 14B (ctx=2048)"},
    "steuerllm": {
        "type": "ollama",
        "name": "steuerllm:28b",
        "desc": "Open-SteuerLLM 28B (German tax law specialist)",
        "source": "hf:windprak/open_steuerllm",
    },
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

DEFAULT_GENERATION_MODELS = [m["name"] for m in AVAILABLE_MODELS.values() if m["type"] == "ollama"]

TEMPERATURES = [0.3, 0.7]
SENTENCES_PER_COMBINATION = 10_000

SPACY_MODEL = "de_core_news_sm"
BERT_MODEL = "deepset/gbert-base"
BERT_MAX_LENGTH = 256

CLASSIFIER_THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]
DEFAULT_THRESHOLD = 0.9

MAX_DATE = "2021-12-31"

RANDOM_SEED = 42
TEST_SPLIT = 0.1

os.environ["TOKENIZERS_PARALLELISM"] = "false"
