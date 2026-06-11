import json
import logging
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DATA_DIR
from utils.mining import append_jsonl

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
MAX_WORKERS = 8
MIN_FILES_PER_STATE = 2_000
MAX_FILES_PER_STATE = 10_000
UNLIMITED_MAX = 1_000_000
MONITOR_INTERVAL = 10


def _check_gesp_installed() -> bool:
    try:
        subprocess.run([sys.executable, "-m", "gesp", "--help"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _count_state_files(state_dir: Path) -> int:
    return len(list(state_dir.rglob("*.html"))) + len(list(state_dir.rglob("*.xhtml")))


def _run_gesp_state(state_code: str, state_name: str, all_files: bool = False) -> int:
    state_dir = GESP_DIR / "raw" / state_code
    state_dir.mkdir(parents=True, exist_ok=True)
    cap = UNLIMITED_MAX if all_files else MAX_FILES_PER_STATE

    existing = _count_state_files(state_dir)
    if not all_files and existing >= MIN_FILES_PER_STATE:
        logger.info(f"  [{state_code}] {state_name}: {existing} files (>= {MIN_FILES_PER_STATE}), skipping")
        return existing
    if existing > 0:
        logger.info(f"  [{state_code}] {state_name}: {existing} files, continuing (target {cap})")
    else:
        logger.info(f"  [{state_code}] Mining {state_name} (cap {cap} files)...")

    courts_str = ",".join(STATE_COURT_TYPES)
    cmd = [
        sys.executable, "-m", "gesp",
        "-s", state_code,
        "-c", courts_str,
        "-p", str(state_dir),
        "-w", "0.5",
    ]

    process = subprocess.Popen(cmd, stdout=None, stderr=None)

    try:
        while process.poll() is None:
            time.sleep(MONITOR_INTERVAL)
            current = _count_state_files(state_dir)
            logger.info(f"  [{state_code}] {current} files so far...")
            if current >= cap:
                logger.info(f"  [{state_code}] Reached {cap} files, stopping...")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                break
    except KeyboardInterrupt:
        logger.warning(f"  [{state_code}] Interrupted, stopping...")
        process.terminate()
        process.wait()
        raise

    process.wait()

    found = _count_state_files(state_dir)
    logger.info(f"  [{state_code}] {state_name}: {found} files")
    return found


def run_gesp(all_files: bool = False):
    text_cache = GESP_DIR / "texts.jsonl"
    if text_cache.exists():
        count = sum(1 for _ in open(text_cache))
        logger.info(f"GESP text cache: {count} entries, skipping download")
        return

    if not _check_gesp_installed():
        logger.info("Installing gesp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gesp"], check=True)

    GESP_DIR.mkdir(parents=True, exist_ok=True)

    cap_str = "unlimited" if all_files else str(MAX_FILES_PER_STATE)
    states_queue = list(STATES)
    logger.info(f"Mining {len(states_queue)} states ({MAX_WORKERS} workers, cap {cap_str}/state)...")
    total = 0
    queue_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        def mine_next():
            nonlocal total
            while True:
                with queue_lock:
                    if not states_queue:
                        break
                    code, name = states_queue.pop(0)
                try:
                    n = _run_gesp_state(code, name, all_files=all_files)
                    with queue_lock:
                        total += n
                except Exception as e:
                    logger.error(f"  [{code}] Failed: {e}")

        num_workers = min(MAX_WORKERS, len(states_queue))
        futures = [pool.submit(mine_next) for _ in range(num_workers)]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Worker failed: {e}")

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
            append_jsonl(text_cache, row)
            saved += 1

            if (i + 1) % 2000 == 0:
                logger.info(f"  Extracted {i+1}/{len(html_files)} ({saved} saved)")

        except Exception as e:
            pass

    logger.info(f"Extracted {saved} texts to {text_cache}")


def mine_gesp(all_files: bool = False):
    logger.info("=" * 50)
    logger.info("Mining state court decisions via gesp...")
    logger.info("=" * 50)
    run_gesp(all_files=all_files)
    extract_texts()
    logger.info("GESP mining complete!")


if __name__ == "__main__":
    mine_gesp()
