import json
import logging
import os
import zipfile
from pathlib import Path

import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
))
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

_LOGGING_CONFIGURED = False


def setup_logging(level: int = logging.INFO):
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root_handler = colorlog.StreamHandler()
        root_handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(root_handler)
    _LOGGING_CONFIGURED = True

from config import DATA_DIR


def load_index():
    index_file = DATA_DIR / "index.json"
    if not index_file.exists():
        return {}
    try:
        with open(index_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_index(index: dict):
    index_file = DATA_DIR / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


def get_module_path(module_name: str) -> Path:
    module_path = DATA_DIR / module_name
    module_path.mkdir(parents=True, exist_ok=True)
    return module_path


def get_module_data(module_name: str) -> dict[int, str]:
    get_module_path(module_name)
    index = load_index()
    return index.get(module_name, {})


def save_all_data_item_files(
    module_name: str, data_items: list[dict]
):
    module_path = get_module_path(module_name)
    index = load_index()
    module_index = index.get(module_name, {})

    for item in data_items:
        item_id = item["id"]
        title = item["title"]
        content = item["file"]

        dir_path = module_path / str(item_id)
        dir_path.mkdir(parents=True, exist_ok=True)

        zip_path = dir_path / "data.zip"
        zip_path.write_bytes(content)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dir_path)
        except zipfile.BadZipFile:
            logger.warning(f"Bad zip for item {item_id}, saving raw file")
            (dir_path / "data.xml").write_bytes(content)

        module_index[str(item_id)] = title
        logger.info(f"Saved [{module_name}] item {item_id}: {title[:60]}")

    index[module_name] = module_index
    save_index(index)


def find_xml_files(module_path: Path):
    for subdir in sorted(module_path.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() == ".xml":
                    yield f
