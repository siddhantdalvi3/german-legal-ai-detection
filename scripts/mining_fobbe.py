import json
import logging
import zipfile
import io
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

from config import FOBBE_DIR

logger = logging.getLogger(__name__)

# Datasets: name -> (concept_or_record_id, txt_filename_pattern)
# Using concept DOIs where possible so they resolve to latest version
FOBBE_DATASETS = {
    "bverwg": {
        "record_id": 10809039,
        "name": "Bundesverwaltungsgericht (CE-BVerwG)",
        "about": "27,200 decisions, 2002-2024",
    },
    "bpatg": {
        "concept_doi": "10.5281/zenodo.3954850",
        "name": "Bundespatentgericht (CE-BPatG)",
        "about": "30,700 decisions, technical/patent law",
    },
    "bgh_strafsachen": {
        "record_id": 4540377,
        "name": "BGH Strafsachen 20. Jhd.",
        "about": "36,000+ criminal decisions, 1950-1999",
    },
}

ZENODO_API = "https://zenodo.org/api/records"


def _resolve_txt_url(dataset: dict) -> str | None:
    """Resolve the TXT dataset ZIP URL for a Fobbe dataset."""
    if "record_id" in dataset:
        url = f"{ZENODO_API}/{dataset['record_id']}"
    elif "concept_doi" in dataset:
        concept_id = dataset["concept_doi"].split(".")[-1]
        url = f"{ZENODO_API}/{concept_id}"
    else:
        return None

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # If concept DOI redirected to a list of versions, take the latest
        if isinstance(data, list):
            data = data[0]

        for f in data.get("files", []):
            if "_TXT_" in f["key"] and f["key"].endswith(".zip"):
                return f["links"]["self"]
        return None
    except Exception as e:
        logger.warning(f"Failed to resolve TXT URL for {dataset.get('name', url)}: {e}")
        return None


def _download_and_extract(dataset: dict) -> list[str]:
    """Download TXT zip for a dataset and extract all decision texts."""
    txt_url = _resolve_txt_url(dataset)
    if not txt_url:
        logger.warning(f"No TXT dataset found for {dataset['name']}")
        return []

    name_key = dataset.get("name", "unknown")
    logger.info(f"Downloading {name_key} TXT dataset...")

    try:
        resp = requests.get(txt_url, timeout=600, stream=True)
        resp.raise_for_status()
        content = resp.content
    except Exception as e:
        logger.error(f"Failed to download {name_key}: {e}")
        return []

    logger.info(f"Extracting {name_key}...")
    texts = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            txt_files = [n for n in z.namelist() if n.endswith(".txt")]
            logger.info(f"  Found {len(txt_files)} TXT files in archive")
            for tf in txt_files:
                try:
                    text = z.read(tf).decode("utf-8", errors="replace").strip()
                    # Remove multiple blank lines
                    text = re.sub(r"\n{3,}", "\n\n", text)
                    if len(text) >= 100:
                        texts.append(text)
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"Failed to extract {name_key}: {e}")

    logger.info(f"  Extracted {len(texts)} valid decisions from {name_key}")
    return texts


def _cache_path(name: str) -> Path:
    return FOBBE_DIR / f"{name}.jsonl"


def _cache_exists(name: str) -> bool:
    return _cache_path(name).exists()


def _save_cache(name: str, texts: list[str]):
    FOBBE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"source": name, "text": t}, ensure_ascii=False) + "\n")
    logger.info(f"Cached {len(texts)} decisions to {path}")


def _load_cache(name: str) -> list[str]:
    path = _cache_path(name)
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
    return texts


def download_datasets(datasets: list[str] | None = None):
    """Download specified Fobbe datasets (or all) and cache them."""
    names = datasets or list(FOBBE_DATASETS.keys())
    for name in names:
        if name not in FOBBE_DATASETS:
            logger.warning(f"Unknown Fobbe dataset: {name}")
            continue
        if _cache_exists(name):
            cached = sum(1 for _ in open(_cache_path(name)))
            logger.info(f"{FOBBE_DATASETS[name]['name']} already cached ({cached} decisions)")
            continue

        dataset = FOBBE_DATASETS[name]
        texts = _download_and_extract(dataset)
        if texts:
            _save_cache(name, texts)
        else:
            logger.warning(f"No texts extracted for {dataset['name']}")


def extract_court_decisions(datasets: list[str] | None = None) -> list[str]:
    """Load cached Fobbe decisions."""
    names = datasets or list(FOBBE_DATASETS.keys())
    all_texts = []
    for name in names:
        if name not in FOBBE_DATASETS:
            continue
        if _cache_exists(name):
            texts = _load_cache(name)
            all_texts.extend(texts)
            logger.info(f"Loaded {len(texts)} decisions from {FOBBE_DATASETS[name]['name']}")
        else:
            logger.info(f"{FOBBE_DATASETS[name]['name']} not cached yet, downloading...")
            dataset = FOBBE_DATASETS[name]
            texts = _download_and_extract(dataset)
            if texts:
                _save_cache(name, texts)
                all_texts.extend(texts)
    return all_texts


def mine_fobbe(datasets: list[str] | None = None) -> list[str]:
    return extract_court_decisions(datasets)


if __name__ == "__main__":
    mine_fobbe()
