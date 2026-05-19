import logging
import sqlite3
from pathlib import Path

import requests
from tqdm import tqdm

from config import OPENLEGALDATA_DIR
from utils.mining import logger

OPENLEGALDATA_DUMP_URL = "https://static.openlegaldata.io/dumps/latest/data.db.gz"

DUMP_PATH = OPENLEGALDATA_DIR / "data.db"


def download_dump():
    if DUMP_PATH.exists():
        logger.info(f"OpenLegalData dump already exists: {DUMP_PATH}")
        return

    OPENLEGALDATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading OpenLegalData dump from {OPENLEGALDATA_DUMP_URL} ...")

    resp = requests.get(OPENLEGALDATA_DUMP_URL, stream=True, timeout=300)
    resp.raise_for_status()

    import gzip
    with open(DUMP_PATH, "wb") as f:
        for chunk in tqdm(resp.iter_content(chunk_size=8192), desc="Downloading OLD"):
            f.write(chunk)

    logger.info(f"OpenLegalData dump saved: {DUMP_PATH}")


def extract_court_decisions(limit: int | None = None) -> list[str]:
    if not DUMP_PATH.exists():
        download_dump()

    texts = []
    conn = sqlite3.connect(str(DUMP_PATH))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT text FROM cases WHERE language='de' AND text IS NOT NULL "
        "AND length(text) > 100 ORDER BY date DESC"
    )
    rows = cursor.fetchmany(limit) if limit else cursor.fetchall()

    for (text,) in tqdm(rows, desc="Extracting OLD decisions"):
        import re
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) >= 100:
            texts.append(clean)

    conn.close()
    logger.info(f"Extracted {len(texts)} court decisions from OpenLegalData")
    return texts


def mine_openlegaldata(limit: int | None = None) -> list[str]:
    return extract_court_decisions(limit)


if __name__ == "__main__":
    mine_openlegaldata()
