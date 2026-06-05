#!/usr/bin/env python3
"""
Download all project assets (idempotent — skips anything already cached).

Usage:
    uv run python download_all.py             # everything (~55 GB)
    uv run python download_all.py --deps      # Python deps + spaCy model
    uv run python download_all.py --models    # AI generation models (Ollama)
    uv run python download_all.py --data      # all human text sources
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def run(cmd: list[str], desc: str, timeout: int | None = None) -> bool:
    print(f"  [{desc}]")
    try:
        subprocess.run(cmd, timeout=timeout, check=True)
        print(f"  OK")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  FAIL: {e}")
        return False


def _spacy_installed(model: str = "de_core_news_sm") -> bool:
    r = subprocess.run(
        [VENV_PYTHON, "-c", f"import spacy; spacy.load('{model}')"],
        capture_output=True,
    )
    return r.returncode == 0


def _ollama_model_pulled(model: str) -> bool:
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return model in r.stdout


def download_deps():
    print("\n=== Python Dependencies (~2 GB with torch) ===")
    venv = PROJECT_ROOT / ".venv"
    if (venv / "bin" / "python").exists():
        r = subprocess.run(
            [str(venv / "bin" / "python"), "-c", "import mlx_lm"],
            capture_output=True,
        )
        if r.returncode == 0:
            print("  .venv + deps already set up, skipping")
        else:
            run(["uv", "sync"], "uv sync (pyproject.toml deps)")
    else:
        run(["uv", "venv"], "create .venv")
        run(["uv", "sync"], "uv sync (pyproject.toml deps)")

    print("\n=== spaCy Model (12 MB) ===")
    if _spacy_installed():
        print("  de_core_news_sm already installed, skipping")
    else:
        run(
            [VENV_PYTHON, "-m", "spacy", "download", "de_core_news_sm"],
            "spacy de_core_news_sm",
            timeout=300,
        )


def download_models():
    print("\n=== Ollama Models (~40 GB total) ===")
    models = [
        ("qwen2.5:7b", "4.7 GB"),
        ("gemma3:12b", "8 GB"),
        ("gemma4", "15 GB"),
        ("mistral", "4.1 GB"),
        ("llama3.1:8b", "4.7 GB"),
        ("llama3.3:70b", "43 GB"),
        ("mixtral:8x7b", "26 GB"),
        ("gemma4:12b", "15 GB"),
    ]
    for model, size in models:
        if _ollama_model_pulled(model):
            print(f"  {model} already pulled, skipping")
        else:
            run(["ollama", "pull", model], f"ollama pull {model} ({size})", timeout=3600)


def download_data():
    print("\n=== Human Text Sources ===")
    print("  (All sources skip if already cached — idempotent)")

    print("\n--- Gesetze-im-Internet + Fobbe + Legal Commons ---")
    run(
        [VENV_PYTHON, "main.py", "--mine", "--openlegaldata", "--fobbe", "bverwg", "bpatg", "bgh_strafsachen", "--legal-commons"],
        "mine Gesetze, OpenLegalData, Fobbe, Legal Commons",
        timeout=3600,
    )

    print("\n--- Rechtsprechung-im-Internet ---")
    r = subprocess.run(
        [VENV_PYTHON, "-c", "import germanlegaltexts"],
        capture_output=True,
    )
    if r.returncode != 0:
        print("  Installing germanlegaltexts for RII downloader...")
        run([VENV_PYTHON, "-m", "pip", "install", "germanlegaltexts"], "pip install germanlegaltexts")
    run(
        [VENV_PYTHON, "main.py", "--mine", "--rii"],
        "mine RII (~5000 judgements)",
        timeout=3600,
    )

    print()
    print("  Data sources summary:")
    run([VENV_PYTHON, "main.py", "--list-models"], "list available data sources")

    print("\n--- OpenLegalData (official HF, ODbL) ---")
    print("  NOTE: Requires HF login + accepting dataset terms:")
    print("    huggingface-cli login")
    print("    Accept: https://huggingface.co/datasets/openlegaldata/court-decisions-germany")
    print("  Then re-run: python main.py --mine --openlegaldata")


def main():
    parser = argparse.ArgumentParser(
        description="Download all project assets (idempotent)"
    )
    parser.add_argument("--deps", action="store_true", help="Python deps + spaCy")
    parser.add_argument("--models", action="store_true", help="Ollama models (~40 GB)")
    parser.add_argument("--data", action="store_true", help="Human text sources (~17 GB)")
    args = parser.parse_args()

    any_specific = args.deps or args.models or args.data

    if not any_specific:
        print("Total download estimate: ~55 GB")
        print("  Python deps + spaCy:  ~2.5 GB")
        print("  Ollama models:        ~40 GB")
        print("  Human data:           ~17 GB  (idempotent, skips cached)")
        print()

    if not any_specific or args.deps:
        download_deps()
    if not any_specific or args.models:
        download_models()
    if not any_specific or args.data:
        download_data()

    print("\n=== All done! ===")


if __name__ == "__main__":
    main()
