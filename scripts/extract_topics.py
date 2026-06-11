import json
import random
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from config import (
    DATA_DIR,
    DIP_DIR,
    GESP_DIR,
    GESETZE_DIR,
    RANDOM_SEED,
)
from utils.mining import logger

TOPICS_DIR = DATA_DIR / "topics"

random.seed(RANDOM_SEED)


def extract_gesetze_topics(output_path: Path) -> list[dict]:
    cache_path = GESETZE_DIR / "topics.jsonl"
    if not cache_path.exists():
        return []
    topics = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            topics.append({"topic": d["heading"], "source": "gesetze"})
    logger.info(f"  Gesetze: {len(topics)} topics")
    with open(output_path, "w") as f:
        for t in topics:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return topics


def extract_dip_topics(output_path: Path) -> list[dict]:
    topics = []
    for name in ("drucksache_text",):
        cache = DIP_DIR / f"{name}.jsonl"
        if not cache.exists():
            continue
        with open(cache) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    titel = d.get("titel", "").strip().replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                    if len(titel) >= 15:
                        topics.append({"topic": titel, "source": "dip"})
                except json.JSONDecodeError:
                    continue
    logger.info(f"  DIP: {len(topics)} topics")
    with open(output_path, "w") as f:
        for t in topics:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return topics


class _GespHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_titel = False
        self._in_p = False
        self._depth = 0
        self._parts = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div" and attrs_dict.get("class") == "docLayoutTitel":
            self._in_titel = True
        if self._in_titel and tag == "p":
            self._in_p = True
            self._parts = []

    def handle_data(self, data):
        if self._in_p:
            self._parts.append(data.strip())

    def handle_entityref(self, name):
        if self._in_p:
            import html
            self._parts.append(html.entities.name2codepoint.get(name, ""))

    def handle_endtag(self, tag):
        if self._in_titel and tag == "div":
            self._in_titel = False
        if self._in_p and tag == "p":
            self._in_p = False

    def get_topic(self) -> str | None:
        text = " ".join(p for p in self._parts if p).strip()
        if len(text) >= 10:
            return text
        return None


def _extract_gesp_html_topic(filepath: Path) -> str | None:
    try:
        content = filepath.read_text("utf-8", errors="replace")
        parser = _GespHtmlParser()
        parser.feed(content)
        return parser.get_topic()
    except Exception:
        return None


def extract_gesp_html_topics(output_path: Path) -> list[dict]:
    topics = []
    raw_dir = GESP_DIR / "raw"
    if not raw_dir.exists():
        logger.info("  GESP raw dir not found, skipping")
        return topics

    for state_dir in sorted(raw_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        inner = state_dir / state_dir.name
        if not inner.exists():
            continue
        for fpath in sorted(inner.iterdir()):
            if fpath.suffix.lower() not in (".html", ".htm"):
                continue
            topic = _extract_gesp_html_topic(fpath)
            if topic:
                topics.append({"topic": topic, "source": "gesp"})

    logger.info(f"  GESP HTML: {len(topics)} topics from HTML files")
    with open(output_path, "w") as f:
        for t in topics:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return topics


def extract_gesp_xml_topics(output_path: Path) -> list[dict]:
    topics = []
    raw_dir = GESP_DIR / "raw"
    if not raw_dir.exists():
        return topics

    for state_dir in sorted(raw_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        inner = state_dir / state_dir.name
        if not inner.exists():
            continue
        for fpath in sorted(inner.iterdir()):
            if fpath.suffix.lower() not in (".xml",):
                continue
            try:
                content = fpath.read_text("utf-8", errors="replace")
                root = ElementTree.fromstring(content)
                schlagworte = [kw.text for kw in root.iter("schlagwort") if kw.text]
                if not schlagworte:
                    continue
                topic = " / ".join(schlagworte)
                if len(topic) >= 10:
                    topics.append({"topic": topic, "source": "gesp"})
            except Exception:
                continue

    logger.info(f"  GESP XML: {len(topics)} topics from XML files")
    with open(output_path, "w") as f:
        for t in topics:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return topics


def combine_topics(all_topics: list[dict], output_path: Path):
    random.shuffle(all_topics)
    with open(output_path, "w") as f:
        for t in all_topics:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    source_counts = {}
    for t in all_topics:
        source_counts[t["source"]] = source_counts.get(t["source"], 0) + 1
    logger.info(f"Combined: {len(all_topics)} topics total")
    for src, cnt in sorted(source_counts.items()):
        logger.info(f"  {src}: {cnt} ({cnt/len(all_topics)*100:.1f}%)")


def main():
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Extracting topics from all sources...")
    logger.info("=" * 50)

    all_topics = []

    logger.info("Gesetze...")
    all_topics.extend(extract_gesetze_topics(TOPICS_DIR / "gesetze.jsonl"))

    logger.info("DIP...")
    all_topics.extend(extract_dip_topics(TOPICS_DIR / "dip.jsonl"))

    logger.info("GESP HTML...")
    all_topics.extend(extract_gesp_html_topics(TOPICS_DIR / "gesp_html.jsonl"))

    logger.info("GESP XML...")
    all_topics.extend(extract_gesp_xml_topics(TOPICS_DIR / "gesp_xml.jsonl"))

    logger.info("Combining all topics...")
    combine_topics(all_topics, TOPICS_DIR / "all_topics.jsonl")
    logger.info("Done!")


if __name__ == "__main__":
    main()
