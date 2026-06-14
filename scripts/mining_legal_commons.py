import logging
from pathlib import Path

from config import LEGAL_COMMONS_DIR

logger = logging.getLogger(__name__)

HF_DATASET = "CUI03/german-commons"
HF_CONFIG = "legal"

# All court-decision splits (skip bundesrecht + eurlex which are laws, not narrative prose)
COURT_SPLITS = [
    "bgh",
    "bgh20",
    "bverfg",
    "bverfgaes",
    "bverwg",
    "bpatg",
    "bag",
    "bfh",
]

CACHE_DIR = LEGAL_COMMONS_DIR / "cache"


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"{split}.jsonl"


def _download_split(split: str):
    """Download a single split and cache it."""
    if _cache_path(split).exists():
        cached = sum(1 for _ in open(_cache_path(split), encoding="utf-8", errors="replace"))
        logger.info(f"  {split}: {cached} docs already cached, skipping")
        return

    logger.info(f"  Downloading {split}...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: uv pip install datasets")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=split, streaming=True)
    with open(_cache_path(split), "w", encoding="utf-8") as f:
        import json
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) >= 100:
                f.write(json.dumps({
                    "id": row.get("id", ""),
                    "source": f"legal_commons_{split}",
                    "text": text,
                }, ensure_ascii=False) + "\n")
                count += 1
    logger.info(f"  {split}: cached {count} docs")


def _load_split(split: str) -> list[str]:
    texts = []
    path = _cache_path(split)
    if path.exists():
        import json
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                texts.append(row["text"])
    return texts


def download_dump(splits: list[str] | None = None):
    """Download specified Legal Commons splits (or all court splits) and cache."""
    names = splits or COURT_SPLITS
    for s in names:
        if s in ("bundesrecht", "eurlex"):
            logger.info(f"  Skipping {s} (laws, not narrative prose)")
            continue
        _download_split(s)


def extract_court_decisions(splits: list[str] | None = None) -> list[str]:
    """Load cached Legal Commons decisions."""
    if not CACHE_DIR.exists():
        logger.info("Legal Commons not cached, downloading...")
        download_dump(splits)

    names = splits or COURT_SPLITS
    all_texts = []
    for s in names:
        if s in ("bundesrecht", "eurlex"):
            continue
        texts = _load_split(s)
        all_texts.extend(texts)
        logger.info(f"  {s}: {len(texts)} docs")
    return all_texts


if __name__ == "__main__":
    extract_court_decisions()
