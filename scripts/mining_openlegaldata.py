import re
from pathlib import Path

from config import OPENLEGALDATA_DIR
from utils.mining import logger

HF_DATASET = "openlegaldata/court-decisions-germany"
HF_CONFIG = "dump-20260520"


def cache_valid() -> bool:
    if not HF_DATASET_CACHE.exists():
        return False
    try:
        from datasets import load_from_disk
        load_from_disk(str(HF_DATASET_CACHE))
        return True
    except Exception as e:
        logger.warning(f"Cannot load OpenLegalData cache: {e}")
        logger.warning(f"Delete {HF_DATASET_CACHE} manually and re-run to re-download")
        return False


def download_dump():
    if cache_valid():
        logger.info(f"OpenLegalData cache found, skipping download")
        return

    logger.info(
        f"Loading OpenLegalData from Hugging Face: {HF_DATASET} (config: {HF_CONFIG})"
    )
    logger.info(
        "Note: Requires HF login + accepting dataset conditions at "
        "https://huggingface.co/datasets/openlegaldata/court-decisions-germany"
    )

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: uv pip install datasets")
        return

    try:
        import tempfile
        with tempfile.TemporaryDirectory(dir=str(OPENLEGALDATA_DIR)) as tmp:
            tmp_path = Path(tmp) / "hf_dataset"
            dataset = load_dataset(HF_DATASET, HF_CONFIG, split="train")
            dataset.save_to_disk(str(tmp_path))
            tmp_path.rename(HF_DATASET_CACHE)
        logger.info(f"Saved {len(dataset)} rows to disk")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info(
            "Make sure you:\n"
            "  1. Created a HF account at https://huggingface.co\n"
            "  2. Run: hf auth login\n"
            "  3. Accepted conditions at https://huggingface.co/datasets/openlegaldata/court-decisions-germany"
        )


HF_DATASET_CACHE = OPENLEGALDATA_DIR / "hf_dataset"


def extract_court_decisions(limit: int | None = None) -> list[str]:
    download_dump()

    if not cache_valid():
        logger.warning("OpenLegalData not available, skipping")
        return []

    logger.info("Loading cached OpenLegalData from disk...")
    from datasets import load_from_disk
    dataset = load_from_disk(str(HF_DATASET_CACHE))

    texts = []
    for i, sample in enumerate(dataset):
        if limit and i >= limit:
            break
        content = sample.get("content", "") or ""
        clean = re.sub(r"<[^>]+>", " ", content)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) >= 100:
            texts.append(clean)

    logger.info(f"Extracted {len(texts)} court decisions from OpenLegalData")
    return texts


def mine_openlegaldata(limit: int | None = None) -> list[str]:
    return extract_court_decisions(limit)


if __name__ == "__main__":
    mine_openlegaldata()
