import json
import logging
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DATA_DIR

logger = logging.getLogger(__name__)

GESP_DIR = DATA_DIR / "gesp"
STATE_COURT_TYPES = ["ag", "arbg", "fg", "lag", "lg", "lsg", "olg", "ovg", "sg", "verfgh"]

# Each state has its own court database server; safe to run in parallel
STATES = [
    ("bw", "Baden-Württemberg"),
    ("by", "Bavaria"),
    ("be", "Berlin"),
    ("bb", "Brandenburg"),
    ("hb", "Bremen"),
    ("hh", "Hamburg"),
    ("he", "Hesse"),
    ("mv", "Mecklenburg-Vorpommern"),
    ("ni", "Lower Saxony"),
    ("nw", "North Rhine-Westphalia"),
    ("rp", "Rhineland-Palatinate"),
    ("sl", "Saarland"),
    ("sn", "Saxony"),
    ("st", "Saxony-Anhalt"),
    ("sh", "Schleswig-Holstein"),
    ("th", "Thuringia"),
]
MAX_WORKERS = 4


def _check_gesp_installed() -> bool:
    try:
        subprocess.run([sys.executable, "-m", "gesp", "--help"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run_gesp_state(state_code: str, state_name: str) -> int:
    state_dir = GESP_DIR / "raw" / state_code
    state_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(state_dir.rglob("*.html"))) + len(list(state_dir.rglob("*.xhtml")))
    if existing > 0:
        logger.info(f"  [{state_code}] {state_name}: {existing} files exist, skipping")
        return existing

    courts_str = ",".join(STATE_COURT_TYPES)
    cmd = [
        sys.executable, "-m", "gesp",
        "-s", state_code,
        "-c", courts_str,
        "-p", str(state_dir),
        "-w", "0.5",
    ]
    logger.info(f"  [{state_code}] Mining {state_name}...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.warning(f"  [{state_code}] gesp returned {result.returncode}")
        return 0

    found = len(list(state_dir.rglob("*.html"))) + len(list(state_dir.rglob("*.xhtml")))
    logger.info(f"  [{state_code}] {state_name}: {found} files")
    return found


def run_gesp():
    text_cache = GESP_DIR / "texts.jsonl"
    if text_cache.exists():
        count = sum(1 for _ in open(text_cache))
        logger.info(f"GESP text cache: {count} entries, skipping download")
        return

    if not _check_gesp_installed():
        logger.info("Installing gesp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gesp"], check=True)

    GESP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Mining {len(STATES)} states in parallel ({MAX_WORKERS} workers)...")
    total = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for code, name in STATES:
            fut = pool.submit(_run_gesp_state, code, name)
            futures[fut] = (code, name)

        for fut in as_completed(futures):
            code, name = futures[fut]
            try:
                n = fut.result()
                total += n
                logger.info(f"  [{code}] {name}: {n} files")
            except Exception as e:
                logger.error(f"  [{code}] Failed: {e}")

    logger.info(f"GESP download complete: ~{total} files")


def extract_texts():
    text_cache = GESP_DIR / "texts.jsonl"
    if text_cache.exists():
        count = sum(1 for _ in open(text_cache))
        logger.info(f"GESP text cache: {count} entries")
        return

    raw_dir = GESP_DIR / "raw"
    if not raw_dir.exists():
        logger.warning("GESP raw HTML not found. Run run_gesp() first.")
        return

    logger.info("Extracting texts from GESP HTML files...")
    GESP_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    html_files = list(raw_dir.rglob("*.html")) + list(raw_dir.rglob("*.xhtml"))
    logger.info(f"Found {len(html_files)} HTML/XHTML files")

    for i, html_path in enumerate(html_files):
        try:
            text = html_path.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 100:
                continue

            rel = html_path.relative_to(raw_dir)
            rel_str = str(rel)
            state = rel.parts[0] if len(rel.parts) > 0 else "unknown"
            court = rel.parts[1] if len(rel.parts) > 1 else "unknown"

            row = {
                "source": f"gesp_{state}_{court}",
                "state": state,
                "court": court,
                "file": rel_str,
                "text": text,
            }
            _append_jsonl(text_cache, row)
            saved += 1

            if (i + 1) % 2000 == 0:
                logger.info(f"  Extracted {i+1}/{len(html_files)} ({saved} saved)")

        except Exception as e:
            pass

    logger.info(f"Extracted {saved} texts to {text_cache}")


def _append_jsonl(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mine_gesp():
    logger.info("=" * 50)
    logger.info("Mining state court decisions via gesp...")
    logger.info("=" * 50)
    run_gesp()
    extract_texts()
    logger.info("GESP mining complete!")


if __name__ == "__main__":
    mine_gesp()
