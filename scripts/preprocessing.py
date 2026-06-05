import json
import random
from pathlib import Path

from tqdm import tqdm

from config import (
    DATA_DIR,
    GESETZE_DIR,
    OPENLEGALDATA_DIR,
    RII_DIR,
    FOBBE_DIR,
    LEGAL_COMMONS_DIR,
    AI_GENERATED_DIR,
    PROCESSED_DIR,
    TEST_SPLIT,
    RANDOM_SEED,
)
from utils.mining import logger, find_xml_files
from utils.nlp_utils import (
    extract_text_from_xml,
    sentence_segment,
    is_boilerplate,
)

random.seed(RANDOM_SEED)


def extract_human_gesetze() -> list[dict]:
    texts = []
    for xml_file in find_xml_files(GESETZE_DIR):
        try:
            content = xml_file.read_text(encoding="utf-8", errors="replace")
            paragraphs = extract_text_from_xml(content)
            for p in paragraphs:
                if not is_boilerplate(p):
                    texts.append({"text": p, "label": 0, "source": "gesetze"})
        except Exception as e:
            logger.warning(f"Error parsing {xml_file}: {e}")
    return texts


def extract_human_openlegaldata() -> list[dict]:
    db_path = OPENLEGALDATA_DIR / "data.db"
    texts = []
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT text FROM cases WHERE language='de' AND text IS NOT NULL "
                "AND length(text) > 100 ORDER BY date DESC LIMIT 50000"
            )
            for (text,) in cursor.fetchall():
                import re
                clean = re.sub(r"<[^>]+>", " ", text)
                clean = re.sub(r"\s+", " ", clean).strip()
                if len(clean) >= 100:
                    texts.append({"text": clean, "label": 0, "source": "old"})
        except sqlite3.OperationalError:
            logger.warning("OpenLegalData DB has no 'cases' table or different schema")
        conn.close()
    return texts

def extract_human_rii(limit: int | None = None) -> list[dict]:
    cache_path = RII_DIR / "judgements.jsonl"
    texts = []
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                if limit and len(texts) >= limit:
                    break
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    if len(text) >= 100:
                        texts.append({"text": text, "label": 0, "source": "rii"})
                except json.JSONDecodeError:
                    continue
    return texts


def extract_human_fobbe() -> list[dict]:
    texts = []
    cache_dir = FOBBE_DIR / "cache_sentences"
    if not cache_dir.exists():
        logger.warning("Fobbe cache_sentences not found, run presplit_cache.py first")
        return texts
    for cache_file in sorted(cache_dir.glob("*.jsonl")):
        with open(cache_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    source = row.get("source", f"fobbe_{cache_file.stem}")
                    if len(text) >= 20:
                        texts.append({"text": text, "label": 0, "source": source, "pre_split": True})
                except json.JSONDecodeError:
                    continue
    return texts


def extract_human_legal_commons() -> list[dict]:
    texts = []
    cache_dir = LEGAL_COMMONS_DIR / "cache_sentences"
    if not cache_dir.exists():
        logger.warning("Legal Commons cache_sentences not found, run presplit_cache.py first")
        return texts
    for cache_file in sorted(cache_dir.glob("*.jsonl")):
        with open(cache_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    source = row.get("source", "legal_commons")
                    if len(text) >= 20:
                        texts.append({"text": text, "label": 0, "source": source, "pre_split": True})
                except json.JSONDecodeError:
                    continue
    return texts


def extract_ai_texts() -> list[dict]:
    texts = []
    for jsonl_file in AI_GENERATED_DIR.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    response = data.get("response", "")
                    texts.append({
                        "text": response,
                        "label": 1,
                        "source": f"ai_{data.get('model','unknown')}",
                        "model": data.get("model", ""),
                        "temperature": data.get("temperature", 0),
                    })
                except json.JSONDecodeError:
                    continue
    return texts


def deduplicate(records: list[dict]) -> list[dict]:
    seen_texts = set()
    unique = []
    for rec in records:
        text = rec["text"].strip().lower()
        if text not in seen_texts:
            seen_texts.add(text)
            unique.append(rec)
    return unique


def sentence_split_records(records: list[dict], batch_size: int = 256) -> list[dict]:
    already_split = []
    to_split = []
    for r in records:
        if r.pop("pre_split", False):
            already_split.append(r)
        else:
            to_split.append(r)
    if not to_split:
        return already_split
    logger.info(f"Sentence splitting {len(to_split)} records with spaCy...")
    from utils.nlp_utils import get_nlp
    nlp = get_nlp()
    texts_with_records = [(r["text"], r) for r in to_split]
    split_records = []
    from tqdm import tqdm
    for doc, record in tqdm(
        nlp.pipe(texts_with_records, batch_size=batch_size, as_tuples=True),
        total=len(to_split),
        desc="Sentence splitting",
    ):
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s) >= 20:
                split_records.append({**record, "text": s})
    return split_records + already_split


def build_dataset(use_openlegaldata: bool = False, use_rii: bool = False, use_fobbe: bool = False, use_legal_commons: bool = False):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    human_records = []
    logger.info("Extracting human texts from Gesetze...")
    human_records.extend(extract_human_gesetze())
    logger.info(f"  Gesetze: {len(human_records)} paragraphs")

    if use_openlegaldata:
        logger.info("Extracting human texts from OpenLegalData...")
        human_records.extend(extract_human_openlegaldata())
        logger.info(f"  OpenLegalData: {len(human_records)} paragraphs cumulative")

    if use_rii:
        logger.info("Extracting human texts from Rechtsprechung-im-Internet...")
        human_records.extend(extract_human_rii())
        logger.info(f"  RII: {len(human_records)} paragraphs cumulative")

    if use_fobbe:
        logger.info("Extracting human texts from Fobbe datasets...")
        human_records.extend(extract_human_fobbe())
        logger.info(f"  Fobbe: {len(human_records)} paragraphs cumulative")

    if use_legal_commons:
        logger.info("Extracting human texts from Legal Commons...")
        human_records.extend(extract_human_legal_commons())
        logger.info(f"  Legal Commons: {len(human_records)} paragraphs cumulative")

    logger.info(f"Total human paragraphs: {len(human_records)}")

    logger.info("Extracting AI texts...")
    ai_records = extract_ai_texts()
    logger.info(f"Total AI responses: {len(ai_records)}")

    all_records = human_records + ai_records
    logger.info(f"Total records before dedup: {len(all_records)}")

    all_records = deduplicate(all_records)
    logger.info(f"Total records after dedup: {len(all_records)}")

    all_records = sentence_split_records(all_records)
    logger.info(f"Total sentences after splitting: {len(all_records)}")

    random.shuffle(all_records)

    split_idx = int(len(all_records) * TEST_SPLIT)
    train_records = all_records[split_idx:]
    test_records = all_records[:split_idx]

    def write_jsonl(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_path = PROCESSED_DIR / "train.jsonl"
    test_path = PROCESSED_DIR / "test.jsonl"

    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)

    label_counts = {0: 0, 1: 0}
    for rec in all_records:
        label_counts[rec["label"]] = label_counts.get(rec["label"], 0) + 1

    logger.info("=" * 50)
    logger.info("Dataset complete:")
    logger.info(f"  Train: {len(train_records)} sentences -> {train_path}")
    logger.info(f"  Test:  {len(test_records)} sentences -> {test_path}")
    logger.info(f"  Human (0): {label_counts.get(0, 0)}")
    logger.info(f"  AI (1):   {label_counts.get(1, 0)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    build_dataset()
