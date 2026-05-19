import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import requests
from tqdm import tqdm

from config import BUNDESTAG_DIR
from utils.mining import logger

BUNDESTAG_OPENDATA_URL = "https://www.bundestag.de/services/opendata"

MIN_YEAR = 2006
MAX_YEAR = 2026


def get_protocol_list() -> list[dict]:
    url = f"{BUNDESTAG_OPENDATA_URL}/index.xml"
    resp = requests.get(url, timeout=60)
    tree = ElementTree.fromstring(resp.content)

    protocols = []
    for item in tree.findall(".//item"):
        link = item.findtext("link", "")
        title = item.findtext("title", "")
        if "plenarprotokoll" in link.lower():
            match = re.search(r"(\d{4})", title)
            if match:
                year = int(match.group(1))
                if MIN_YEAR <= year <= MAX_YEAR:
                    protocols.append({"title": title, "url": link, "year": year})
    return protocols


def download_protocols(limit: int | None = None):
    BUNDESTAG_DIR.mkdir(parents=True, exist_ok=True)
    protocols = get_protocol_list()
    logger.info(f"Found {len(protocols)} Bundestag protocols ({MIN_YEAR}-{MAX_YEAR})")

    downloaded = 0
    for p in tqdm(protocols[:limit], desc="Downloading protocols"):
        year_dir = BUNDESTAG_DIR / str(p["year"])
        year_dir.mkdir(exist_ok=True)

        filename = p["url"].split("/")[-1].replace(".xml", ".zip")
        dest = year_dir / filename

        if dest.exists():
            continue

        try:
            resp = requests.get(p["url"], timeout=120)
            if "xml" in resp.headers.get("Content-Type", ""):
                dest = dest.with_suffix(".xml")
                dest.write_bytes(resp.content)
            else:
                dest.write_bytes(resp.content)
                if zipfile.is_zipfile(dest):
                    with zipfile.ZipFile(dest) as zf:
                        zf.extractall(year_dir)
            downloaded += 1
        except Exception as e:
            logger.warning(f"Failed to download {p['title']}: {e}")

    logger.info(f"Downloaded {downloaded} Bundestag protocols")


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

    for year_dir in sorted(BUNDESTAG_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for xml_file in year_dir.glob("*.xml"):
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
