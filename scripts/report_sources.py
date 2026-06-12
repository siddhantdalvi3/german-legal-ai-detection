import json
from collections import defaultdict
from pathlib import Path

from config import (
    AI_GENERATED_DIR,
    DATA_DIR,
    DIP_DIR,
    FOBBE_DIR,
    GESETZE_DIR,
    GESP_DIR,
    LEGAL_COMMONS_DIR,
    OPENLEGALDATA_DIR,
    PROCESSED_DIR,
    RII_DIR,
)
from utils.mining import logger, find_xml_files


def _fmt(n: int) -> str:
    return f"{n:,}"


def _fmt_size(path) -> str:
    if not path.exists():
        return "—"
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _fmt_size_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                for i, _ in enumerate(f, 1):
                    pass
            return i
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 0


def _source_name(raw: str) -> str:
    mapping = {
        "gesetze": "Gesetze-im-Internet",
        "old": "OpenLegalData",
        "rii": "Rechtsprechung-im-Internet",
        "fobbe": "Fobbe",
        "legal_commons": "Legal Commons",
        "dip": "DIP Bundestag",
        "gesp": "GESP State Courts",
        "ai": "AI Generated",
    }
    for prefix, name in mapping.items():
        if raw.startswith(prefix):
            return name
    return raw


def _processed_source_breakdown(path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not path.exists():
        return counts
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
                src = row.get("source", "unknown")
                counts[src] += 1
            except json.JSONDecodeError:
                continue
    return counts


def report_sources():
    print()
    print("=" * 72)
    print("  DATA SOURCE REPORT")
    print("=" * 72)
    print(f"  Data root: {DATA_DIR}")
    print(f"  Total size: {_fmt_size(DATA_DIR)}")
    print()

    # ── Raw sources ──
    print("  ── RAW SOURCES ──")
    print()

    # 1. Gesetze-im-Internet
    gesetze_files = list(find_xml_files(GESETZE_DIR))
    gesetze_size = sum(f.stat().st_size for f in gesetze_files)
    topics_file = GESETZE_DIR / "topics.jsonl"
    print(f"  Gesetze-im-Internet")
    print(f"    Location: {GESETZE_DIR}")
    print(f"    XML files: {_fmt(len(gesetze_files))} ({_fmt_size_bytes(gesetze_size)})")
    if topics_file.exists():
        print(f"    Topic headings: {_fmt(_count_lines(topics_file))}")

    # 2. OpenLegalData
    old_hf = OPENLEGALDATA_DIR / "hf_dataset"
    old_db = OPENLEGALDATA_DIR / "data.db"
    if old_db.exists():
        print(f"  OpenLegalData")
        print(f"    Location: {OPENLEGALDATA_DIR}")
        print(f"    SQLite DB: {_fmt_size_bytes(old_db.stat().st_size)}")
    elif old_hf.exists():
        print(f"  OpenLegalData")
        print(f"    Location: {OPENLEGALDATA_DIR}")
        print(f"    HF Dataset: {_fmt_size(old_hf)}")
    else:
        print(f"  OpenLegalData")
        print(f"    Location: {OPENLEGALDATA_DIR}")
        print(f"    Status: not downloaded")

    # 3. RII
    rii_file = RII_DIR / "judgements.jsonl"
    if rii_file.exists():
        rii_lines = _count_lines(rii_file)
        rii_size = rii_file.stat().st_size
        print(f"  Rechtsprechung-im-Internet")
        print(f"    Location: {RII_DIR}")
        print(f"    Judgements: {_fmt(rii_lines)} ({_fmt_size_bytes(rii_size)})")

    # 4. Fobbe
    fobbe_cache = FOBBE_DIR / "cache_sentences"
    if fobbe_cache.exists():
        fobbe_files = list(fobbe_cache.glob("*.jsonl"))
        fobbe_lines = sum(_count_lines(f) for f in fobbe_files)
        fobbe_size = sum(f.stat().st_size for f in fobbe_files)
        print(f"  Fobbe (CC0 Court Decisions)")
        print(f"    Location: {FOBBE_DIR}")
        print(f"    Cache files: {_fmt(len(fobbe_files))} ({_fmt_size_bytes(fobbe_size)})")
        print(f"    Total sentences: {_fmt(fobbe_lines)}")
        for f in sorted(fobbe_files):
            print(f"      {f.name}: {_fmt(_count_lines(f))} sentences ({_fmt_size_bytes(f.stat().st_size)})")

    # 5. Legal Commons
    lc_cache = LEGAL_COMMONS_DIR / "cache_sentences"
    if lc_cache.exists():
        lc_files = list(lc_cache.glob("*.jsonl"))
        lc_lines = sum(_count_lines(f) for f in lc_files)
        lc_size = sum(f.stat().st_size for f in lc_files)
        print(f"  Legal Commons (CUI03/german-legal-commons)")
        print(f"    Location: {LEGAL_COMMONS_DIR}")
        print(f"    Cache files: {_fmt(len(lc_files))} ({_fmt_size_bytes(lc_size)})")
        print(f"    Total sentences: {_fmt(lc_lines)}")
        for f in sorted(lc_files):
            print(f"      {f.name}: {_fmt(_count_lines(f))} sentences ({_fmt_size_bytes(f.stat().st_size)})")

    # 6. DIP Bundestag
    dip_files = list(DIP_DIR.glob("*.jsonl"))
    if dip_files:
        dip_size = sum(f.stat().st_size for f in dip_files)
        dip_lines = sum(_count_lines(f) for f in dip_files)
        print(f"  DIP Bundestag")
        print(f"    Location: {DIP_DIR}")
        print(f"    Files: {_fmt(len(dip_files))} ({_fmt_size_bytes(dip_size)})")
        print(f"    Total documents: {_fmt(dip_lines)}")
        for f in sorted(dip_files):
            print(f"      {f.name}: {_fmt(_count_lines(f))} docs ({_fmt_size_bytes(f.stat().st_size)})")

    # 7. GESP State Courts
    gesp_raw = GESP_DIR / "raw"
    if gesp_raw.exists():
        state_dirs = sorted([d for d in gesp_raw.iterdir() if d.is_dir()])
        html_files = list(gesp_raw.rglob("*.html")) + list(gesp_raw.rglob("*.xhtml"))
        gesp_size = sum(f.stat().st_size for f in html_files)
        print(f"  GESP State Courts")
        print(f"    Location: {GESP_DIR}")
        print(f"    Total files: {_fmt(len(html_files))} ({_fmt_size_bytes(gesp_size)})")
        for sd in state_dirs:
            sf = list(sd.rglob("*.html")) + list(sd.rglob("*.xhtml"))
            ss = sum(f.stat().st_size for f in sf)
            print(f"      {sd.name}: {_fmt(len(sf))} files ({_fmt_size_bytes(ss)})")
        gesp_texts = GESP_DIR / "texts.jsonl"
        if gesp_texts.exists():
            print(f"    Extracted texts: {_fmt(_count_lines(gesp_texts))}")

    # 8. AI Generated
    ai_files = list(AI_GENERATED_DIR.glob("*.jsonl"))
    if ai_files:
        ai_total_lines = sum(_count_lines(f) for f in ai_files)
        ai_size = sum(f.stat().st_size for f in ai_files)
        print(f"  AI Generated")
        print(f"    Location: {AI_GENERATED_DIR}")
        print(f"    Files: {_fmt(len(ai_files))} ({_fmt_size_bytes(ai_size)})")
        print(f"    Total responses: {_fmt(ai_total_lines)}")
        for f in sorted(ai_files):
            print(f"      {f.name}: {_fmt(_count_lines(f))} responses ({_fmt_size_bytes(f.stat().st_size)})")
    else:
        print(f"  AI Generated")
        print(f"    Location: {AI_GENERATED_DIR}")
        print(f"    Status: empty (no AI data yet)")

    # ── Processed dataset ──
    print()
    print("  ── PROCESSED DATASET ──")
    print()

    train_file = PROCESSED_DIR / "train.jsonl"
    test_file = PROCESSED_DIR / "test.jsonl"

    if not train_file.exists():
        print("  No processed dataset. Run --preprocess first.")
        return

    train_lines = _count_lines(train_file)
    test_lines = _count_lines(test_file)
    train_size = train_file.stat().st_size
    test_size = test_file.stat().st_size

    print(f"  Train: {_fmt(train_lines):>10} sentences  ({_fmt_size_bytes(train_size)})")
    print(f"  Test:  {_fmt(test_lines):>10} sentences  ({_fmt_size_bytes(test_size)})")
    print(f"  Total: {_fmt(train_lines + test_lines):>10} sentences")

    train_sources = _processed_source_breakdown(train_file)
    test_sources = _processed_source_breakdown(test_file)
    all_sources = sorted(set(list(train_sources.keys()) + list(test_sources.keys())))

    # Aggregate by source group
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for src, count in train_sources.items():
        group = _source_name(src)
        grouped[group]["train"] += count
    for src, count in test_sources.items():
        group = _source_name(src)
        grouped[group]["test"] += count

    print()
    print(f"  {'Source':<30} {'Train':>10} {'Test':>10} {'Total':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
    for group in sorted(grouped.keys()):
        tr = grouped[group]["train"]
        te = grouped[group]["test"]
        total = tr + te
        frac = total / (train_lines + test_lines) * 100
        print(f"  {group:<30} {_fmt(tr):>10} {_fmt(te):>10} {_fmt(total):>10}  ({frac:.1f}%)")

    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'TOTAL':<30} {_fmt(train_lines):>10} {_fmt(test_lines):>10} {_fmt(train_lines + test_lines):>10}")

    # Label counts
    label_counts = {0: 0, 1: 0}
    for path in (train_file, test_file):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    label_counts[row["label"]] += 1
                except json.JSONDecodeError:
                    continue

    print()
    print(f"  Labels:")
    print(f"    Human (0): {_fmt(label_counts.get(0, 0)):>12}")
    print(f"    AI    (1): {_fmt(label_counts.get(1, 0)):>12}")
    print(f"    AI ratio:  {(label_counts.get(1, 0) / max(label_counts.get(0, 0) + label_counts.get(1, 0), 1) * 100):.2f}%")
    print("=" * 72)
    print()
