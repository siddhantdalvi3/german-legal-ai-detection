import re
from pathlib import Path

from config import OPENLEGALDATA_DIR
from utils.mining import logger

HF_DATASET = "openlegaldata/court-decisions-germany"
HF_CONFIG = "dump-20260520-10k"  # ~10k rows, ~50 MB


def download_dump():
    if HF_DATASET_CACHE.exists():
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
        dataset = load_dataset(HF_DATASET, HF_CONFIG, split="train", trust_remote_code=True)
        dataset.save_to_disk(str(OPENLEGALDATA_DIR / "hf_dataset"))
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
    if not HF_DATASET_CACHE.exists():
        download_dump()

    if not HF_DATASET_CACHE.exists():
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
