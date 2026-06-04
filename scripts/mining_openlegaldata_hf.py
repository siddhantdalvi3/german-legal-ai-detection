import logging
from pathlib import Path

from config import OPENLEGALDATA_HF_DIR
from utils.mining import get_module_data

logger = logging.getLogger(__name__)

HF_DATASET = "harshildarji/openlegaldata"
HF_DATASET_CACHE = OPENLEGALDATA_HF_DIR / "hf_dataset"

NARRATIVE_FIELDS = ["tatbestand", "entscheidungsgruende"]


def cache_valid() -> bool:
    if not HF_DATASET_CACHE.exists():
        return False
    try:
        from datasets import load_from_disk
        load_from_disk(str(HF_DATASET_CACHE))
        return True
    except Exception:
        import shutil
        logger.warning("Corrupt cache found, re-downloading")
        shutil.rmtree(HF_DATASET_CACHE)
        return False


def download_dump():
    if cache_valid():
        logger.info("harshildarji/openlegaldata cache found, skipping download")
        return

    logger.info(f"Loading harshildarji/openlegaldata from Hugging Face (MIT license)...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: uv pip install datasets")
        return

    try:
        import tempfile
        OPENLEGALDATA_HF_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OPENLEGALDATA_HF_DIR)) as tmp:
            tmp_path = Path(tmp) / "hf_dataset"
            dataset = load_dataset(HF_DATASET, split="main")
            dataset.save_to_disk(str(tmp_path))
            tmp_path.rename(HF_DATASET_CACHE)
        logger.info(f"Saved {len(dataset)} rows to disk")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")


def extract_court_decisions(limit: int | None = None) -> list[str]:
    download_dump()

    if not cache_valid():
        logger.warning("harshildarji/openlegaldata not available, skipping")
        return []

    logger.info("Loading cached harshildarji/openlegaldata from disk...")
    from datasets import load_from_disk
    dataset = load_from_disk(str(HF_DATASET_CACHE))

    texts = []
    for i, sample in enumerate(dataset):
        if limit and i >= limit:
            break
        parts = []
        for field in NARRATIVE_FIELDS:
            chunks = sample.get(field) or []
            for c in chunks:
                c = c.strip()
                if c:
                    parts.append(c)
        text = " ".join(parts)
        if len(text) >= 100:
            texts.append(text)

    logger.info(f"Extracted {len(texts)} court decisions from harshildarji/openlegaldata")
    return texts


def mine_openlegaldata_hf(limit: int | None = None) -> list[str]:
    return extract_court_decisions(limit)


if __name__ == "__main__":
    mine_openlegaldata_hf()
