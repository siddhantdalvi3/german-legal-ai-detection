import json
import logging
import time
from pathlib import Path

import requests

from config import DATA_DIR, MAX_DATE

logger = logging.getLogger(__name__)

DIP_API_BASE = "https://search.dip.bundestag.de/api/v1"
DIP_API_KEY = "R2BZaee.DjdCyihKZMf8AOjtScubP2EVydegzjmBIQ"
DIP_DIR = DATA_DIR / "dip_bundestag"
DELAY = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GermanLegalTextResearchBot; contact@siddhantdalvi.com)",
    "Accept": "application/json",
}


def _api_get(endpoint: str, params: dict | None = None) -> dict | None:
    url = f"{DIP_API_BASE}/{endpoint}"
    p = params or {}
    p["apikey"] = DIP_API_KEY
    try:
        resp = requests.get(url, params=p, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"API error: {e}")
        return None


def _append_jsonl(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in open(path))


def mine_endpoint(endpoint: str, doc_type: str, limit: int | None = None):
    text_path = DIP_DIR / f"{doc_type}_text.jsonl"
    DIP_DIR.mkdir(parents=True, exist_ok=True)

    total_before = _count_lines(text_path)
    if total_before > 0:
        logger.info(f"  {doc_type} existing: {total_before} texts, skipping")
        logger.info(f"  Delete {text_path} to re-download from scratch")
        return

    logger.info(f"  Mining {doc_type} (single-threaded, {DELAY}s delay)...")
    cursor = None
    saved = 0
    num_found = None

    while True:
        params = {"f.datum.end": MAX_DATE, "format": "json"}
        if cursor:
            params["cursor"] = cursor

        data = _api_get(endpoint, params)
        if data is None:
            logger.warning(f"  API unavailable, retrying in 30s...")
            time.sleep(30)
            continue

        docs = data.get("documents", [])
        for d in docs:
            text = d.get("text", "")
            if text and len(text) >= 100:
                row = {
                    "id": d.get("id"),
                    "dokumentnummer": d.get("dokumentnummer"),
                    "datum": d.get("datum"),
                    "titel": d.get("titel"),
                    "text": text,
                }
                _append_jsonl(text_path, row)
                saved += 1

        cursor = data.get("cursor")
        if not cursor or not docs:
            break

        if saved > 0 and saved % 500 == 0:
            num_found = data.get("numFound", num_found) or 0
            logger.info(f"  {doc_type}: {saved} / {num_found}")

        if limit and saved >= limit:
            break

        time.sleep(DELAY)

    total = _count_lines(text_path)
    logger.info(f"  {doc_type}: {total} texts cached ({saved} new)")


def mine_dip_bundestag(limit: int | None = None):
    logger.info("=" * 50)
    logger.info("Mining DIP Bundestag (Drucksachen + Plenarprotokolle)...")
    logger.info("=" * 50)
    mine_endpoint("drucksache-text", "drucksache", limit=limit)
    mine_endpoint("plenarprotokoll-text", "plenarprotokoll", limit=limit)
    logger.info("DIP Bundestag mining complete!")


if __name__ == "__main__":
    mine_dip_bundestag(limit=100)
