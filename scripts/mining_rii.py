import asyncio
import json
import logging
from pathlib import Path

from config import RII_DIR

logger = logging.getLogger(__name__)

RII_CACHE = RII_DIR / "judgements.jsonl"
NARRATIVE_FIELDS = ["tatbestand", "entscheidungsgruende"]
FALLBACK_FIELDS = ["gruende", "tenor"]

# How many to download per batch (a few 1000s is already a lot of text)
DEFAULT_LIMIT = 5000


def _get_text(judgement, field: str) -> str | None:
    """Extract text content from a judgement section field."""
    section = getattr(judgement, field, None)
    if section is None:
        return None
    content = getattr(section, "content", None)
    if content and isinstance(content, str) and len(content.strip()) >= 50:
        return content.strip()
    return None


def _iter_cache() -> list[dict]:
    """Read cached judgements from disk (doknr -> text blob)."""
    if not RII_CACHE.exists():
        return []
    results = []
    with open(RII_CACHE, encoding="utf-8", errors="replace") as f:
        for line in f:
            row = json.loads(line)
            results.append(row)
    return results


def _append_cache(judgement) -> None:
    """Append a single judgement to the cache file."""
    RII_DIR.mkdir(parents=True, exist_ok=True)
    text = _extract_text(judgement)
    if not text:
        return
    row = {
        "doknr": getattr(judgement, "doknr", ""),
        "aktenzeichen": getattr(judgement, "aktenzeichen", ""),
        "gertyp": getattr(judgement, "gertyp", ""),
        "text": text,
    }
    with open(RII_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_text(judgement) -> str | None:
    """Extract the best narrative text from a judgement."""
    for field in NARRATIVE_FIELDS:
        text = _get_text(judgement, field)
        if text:
            return text
    for field in FALLBACK_FIELDS:
        text = _get_text(judgement, field)
        if text:
            return text
    return None


def download_dump(limit: int = DEFAULT_LIMIT):
    """Download judgements from RII and cache to disk. Skips if already cached."""
    if RII_CACHE.exists():
        cached = sum(1 for _ in open(RII_CACHE, encoding="utf-8", errors="replace"))
        logger.info(f"RII cache found: {cached} judgements, skipping download")
        return

    logger.info(f"Downloading up to {limit} judgements from Rechtsprechung-im-Internet...")
    logger.info("This may take a while (one HTTP request per judgement)")

    try:
        from germanlegaltexts.GermanJudgementDownloader import GermanJudgementDownloader
    except ImportError:
        logger.error("Install germanlegaltexts: uv pip install germanlegaltexts")
        return

    downloader = GermanJudgementDownloader()
    RII_DIR.mkdir(parents=True, exist_ok=True)

    async def _download():
        count = 0
        async for j in downloader.iter_first_n_judgements(limit, max_per_second=5.0):
            _append_cache(j)
            count += 1
            if count % 100 == 0:
                logger.info(f"  Downloaded {count}/{limit} judgements")
        logger.info(f"Download complete: {count} judgements cached")

    asyncio.run(_download())
    cached = sum(1 for _ in open(RII_CACHE, encoding="utf-8", errors="replace"))
    logger.info(f"Total cached: {cached} judgements with narrative text")


def extract_court_decisions(limit: int | None = None) -> list[str]:
    if not RII_CACHE.exists():
        logger.warning("RII cache not found. Run download first or it will download now.")
        download_dump()

    records = _iter_cache()
    if limit:
        records = records[:limit]

    texts = [r["text"] for r in records]
    logger.info(f"Extracted {len(texts)} decisions from RII cache")
    return texts


def mine_rii(limit: int | None = None) -> list[str]:
    """Mine court decisions from Rechtsprechung-im-Internet.de."""
    return extract_court_decisions(limit)


if __name__ == "__main__":
    mine_rii()
