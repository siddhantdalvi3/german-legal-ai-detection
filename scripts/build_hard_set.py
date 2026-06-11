"""
Build the 200-sentence hard set from sources NOT in training data.
Uses German Wikipedia legal articles — stable API, CC-licensed text,
and clearly distinct from Gesetze-im-Internet / OpenLegalData corpus.

Output: data/processed/hard_set.jsonl  (200 sentences, label=0)
"""

import json
import time
from pathlib import Path

import requests

from config import PROJECT_ROOT
from utils.mining import logger
from utils.nlp_utils import sentence_segment

HARD_SET_PATH = PROJECT_ROOT / "tests" / "hard_set.jsonl"
TARGET = 200

WIKI_API = "https://de.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "GermanAITextDetector/0.2 (project; academic) Python/3.14"
}

CATEGORIES = [
    "Kategorie:Recht_(Deutschland)",
    "Kategorie:Verwaltungsrecht_(Deutschland)",
    "Kategorie:Verfassungsrecht_(Deutschland)",
    "Kategorie:Strafrecht_(Deutschland)",
    "Kategorie:Zivilrecht_(Deutschland)",
    "Kategorie:Öffentliches_Recht",
    "Kategorie:Verwaltung_(Deutschland)",
    "Kategorie:Rechtsgeschichte_(Deutschland)",
    "Kategorie:Steuerrecht_(Deutschland)",
    "Kategorie:Arbeitsrecht_(Deutschland)",
]

BATCH_SIZE = 50


def get_category_pages(category: str, limit: int = 20) -> list[str]:
    pages = []
    cmcontinue = None
    retries = 3
    while len(pages) < limit:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "50",
            "cmtype": "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        for attempt in range(retries):
            resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break

        data = resp.json()

        for m in data.get("query", {}).get("categorymembers", []):
            title = m.get("title", "")
            if title and "Liste" not in title:
                pages.append(title)

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break

        time.sleep(1)

    return pages[:limit]


def batch_extract(titles: list[str]) -> dict[str, str]:
    """Fetch extracts for up to 50 titles in one API call."""
    result = {}
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i : i + BATCH_SIZE]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "prop": "extracts",
            "explaintext": True,
            "exlimit": BATCH_SIZE,
        }
        retries = 3
        for attempt in range(retries):
            resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=60)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        for page_id, page_data in data.get("query", {}).get("pages", {}).items():
            if page_id != "-1":
                title = page_data.get("title", "")
                result[title] = page_data.get("extract", "")
        time.sleep(1)
    return result


def collect_sentences() -> list[str]:
    sentences = []

    logger.info(f"Fetching pages from {len(CATEGORIES)} legal categories...")
    all_pages = []
    for cat in CATEGORIES:
        pages = get_category_pages(cat, limit=15)
        logger.info(f"  {cat.split('_')[0].split(':')[-1]:30s} → {len(pages)} pages")
        all_pages.extend(pages)
        time.sleep(1)

    all_pages = list(dict.fromkeys(all_pages))
    logger.info(f"Total unique pages: {len(all_pages)}")

    logger.info(f"Fetching extracts in batches of {BATCH_SIZE}...")
    texts = batch_extract(all_pages)

    for title, text in texts.items():
        if len(sentences) >= TARGET:
            break
        if len(text) < 200:
            continue

        segs = sentence_segment(text)
        for s in segs:
            s = s.strip()
            if 50 <= len(s) <= 500:
                sentences.append(s)

        logger.info(
            f"  {title[:50]:50s} → {len(segs):3d} sentences "
            f"(total: {len(sentences)})"
        )

    return sentences[:TARGET]


def load_existing() -> set[str]:
    existing = set()
    if HARD_SET_PATH.exists():
        for line in open(HARD_SET_PATH):
            try:
                rec = json.loads(line)
                existing.add(rec.get("text", ""))
            except json.JSONDecodeError:
                continue
    return existing


def build_hard_set():
    HARD_SET_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_texts = load_existing()
    if len(existing_texts) >= TARGET:
        logger.info(f"Hard set already exists ({len(existing_texts)} sentences), skipping")
        return
    logger.info(f"Hard set has {len(existing_texts)}/{TARGET}, adding more...")

    sentences = list(existing_texts)
    new_sentences = collect_sentences()
    for s in new_sentences:
        if len(sentences) >= TARGET:
            break
        if s not in existing_texts:
            sentences.append(s)

    if len(sentences) < TARGET:
        logger.warning(
            f"Only got {len(sentences)}/{TARGET} sentences. "
            f"Run again or add more categories."
        )

    with open(HARD_SET_PATH, "w", encoding="utf-8") as f:
        for s in sentences:
            record = {"text": s, "label": 0, "source": "hard_set"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"Hard set saved: {HARD_SET_PATH} ({len(sentences)} sentences)"
    )


if __name__ == "__main__":
    build_hard_set()
