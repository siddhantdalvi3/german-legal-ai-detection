import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import BUNDESTAG_DIR
from utils.mining import logger

# Section IDs for each legislative period with structured XML protocols
# Each maps to {section_id: legislative_period_label}
SECTION_IDS = {
    "1058442": "21",  # 21st period (2025–2029)
    "866354":  "20",  # 20th period (2021–2025)
    "543410":  "19",  # 19th period (2017–2021)
}


def fetch_protocols_for_section(section_id: str, label: str) -> list[dict]:
    protocols = []
    offset = 0
    while True:
        url = (
            f"https://www.bundestag.de/ajax/filterlist/de/services/opendata/"
            f"{section_id}-{section_id}?noFilterSet=true&offset={offset}"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        links = re.findall(
            r'href="(https://www\.bundestag\.de/resource/blob/\d+/[^"]+\.xml)"',
            resp.text,
        )
        for link in links:
            protocols.append({"url": link, "period": label})

        hits_match = re.search(r'data-nextoffset="(\d+)"', resp.text)
        if hits_match:
            offset = int(hits_match.group(1))
        else:
            break

        if not links:
            break

        time.sleep(0.5)

    return protocols


def get_protocol_list() -> list[dict]:
    all_protocols = []
    for section_id, label in SECTION_IDS.items():
        logger.info(f"Fetching protocol list for period {label} (section {section_id})...")
        protocols = fetch_protocols_for_section(section_id, label)
        logger.info(f"  Found {len(protocols)} protocols")
        all_protocols.extend(protocols)
    return all_protocols


def download_protocols(limit: int | None = None):
    BUNDESTAG_DIR.mkdir(parents=True, exist_ok=True)
    protocols = get_protocol_list()
    logger.info(f"Total Bundestag protocols found: {len(protocols)}")

    downloaded = 0
    skipped = 0
    for p in tqdm(protocols[:limit], desc="Downloading protocols"):
        period_dir = BUNDESTAG_DIR / p["period"]
        period_dir.mkdir(exist_ok=True)

        filename = p["url"].split("/")[-1]
        dest = period_dir / filename

        if dest.exists():
            skipped += 1
            continue

        try:
            resp = requests.get(p["url"], timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            downloaded += 1
        except Exception as e:
            logger.warning(f"Failed to download {p['url']}: {e}")

    logger.info(f"Downloaded {downloaded} new protocols ({skipped} already existed)")


def parse_protocol_text(xml_path: Path) -> list[str]:
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    speeches = re.findall(r"<rede[^>]*>(.*?)</rede>", text, re.DOTALL)
    paragraphs = []
    for speech in speeches:
        clean = re.sub(r"<[^>]+>", " ", speech)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) >= 100:
            paragraphs.append(clean)
    return paragraphs


def extract_all_speeches(limit_protocols: int | None = None) -> list[str]:
    download_protocols(limit_protocols)
    all_speeches = []

    for period_dir in sorted(BUNDESTAG_DIR.iterdir()):
        if not period_dir.is_dir():
            continue
        for xml_file in period_dir.glob("*.xml"):
            try:
                speeches = parse_protocol_text(xml_file)
                all_speeches.extend(speeches)
            except Exception as e:
                logger.warning(f"Failed to parse {xml_file.name}: {e}")

    logger.info(f"Extracted {len(all_speeches)} speeches from Bundestag protocols")
    return all_speeches


def mine_bundestag(limit: int | None = None) -> list[str]:
    return extract_all_speeches(limit)


if __name__ == "__main__":
    mine_bundestag()
