import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

GESETZE_DIR = DATA_DIR / "gesetze_im_internet"
OPENLEGALDATA_DIR = DATA_DIR / "openlegaldata"
BUNDESTAG_DIR = DATA_DIR / "bundestag"
AI_GENERATED_DIR = DATA_DIR / "ai_generated"

GESETZE_TOC_URL = "https://www.gesetze-im-internet.de/gii-toc.xml"
OPENLEGALDATA_DUMP_URL = "https://static.openlegaldata.io/dumps/latest/"
BUNDESTAG_OPENDATA_URL = "https://www.bundestag.de/services/opendata"

OLLAMA_MODELS = [
    "qwen2.5:7b",
    "gemma3:12b",
    "gemma4",
    "mistral",
]
MLX_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

TEMPERATURES = [0.3, 0.7, 1.0]
SENTENCES_PER_COMBINATION = 25_000
AI_TARGET = 450_000

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
