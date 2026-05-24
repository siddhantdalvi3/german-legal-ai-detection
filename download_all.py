#!/usr/bin/env python3.14
"""
Download all dependencies, models, and data for the German AI-text detector.
Idempotent: skips anything already downloaded.

Usage:
    uv run python download_all.py            # full download (~55 GB)
    uv run python download_all.py --deps     # only Python deps + spaCy
    uv run python download_all.py --models   # only AI models
    uv run python download_all.py --data     # only human data (mine)
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()


def run(cmd: list[str], desc: str, timeout: int | None = None,
        silent: bool = False) -> bool:
    print(f"  [{desc}]")
    std = subprocess.DEVNULL if silent else None
    try:
        subprocess.run(cmd, timeout=timeout, check=True,
                       stdout=std, stderr=std)
        print(f"  ✓")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  ✗ {e}")
        return False


def _spacy_installed() -> bool:
    result = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-c",
         "import spacy; spacy.load('de_core_news_lg')"],
        capture_output=True,
    )
    return result.returncode == 0


def _ollama_model_pulled(model: str) -> bool:
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True
    )
    return model in result.stdout


def _mlx_cached() -> bool:
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        return False
    for d in hf_cache.iterdir():
        if "mistral" in d.name.lower() and "4bit" in d.name.lower():
            return True
    return False


def download_deps():
    print("\n=== Python Dependencies (~2 GB with torch) ===")
    venv = PROJECT_ROOT / ".venv"
    if (venv / "bin" / "python").exists():
        print("  .venv exists, skipping")
    else:
        run(["uv", "venv"], "create .venv")

    # lazy check: if mlx-lm is importable, deps are installed
    result = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "import mlx_lm"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("  uv sync already done, skipping")
    else:
        run(["uv", "sync"], "uv sync (pyproject.toml deps)")

    print("\n=== spaCy Model (568 MB) ===")
    if _spacy_installed():
        print("  de_core_news_lg already installed, skipping")
    else:
        run(
            [str(venv / "bin" / "python"), "-m", "spacy", "download", "de_core_news_lg"],
            "spacy de_core_news_lg",
            timeout=600,
        )


def download_models():
    print("\n=== Ollama Models (~32 GB total) ===")
    ollama_models = [
        ("qwen2.5:7b", "4.7 GB"),
        ("gemma3:12b", "8 GB"),
        ("gemma4", "15 GB"),
        ("mistral", "4.1 GB"),
    ]
    for model, size in ollama_models:
        if _ollama_model_pulled(model):
            print(f"  {model} already pulled, skipping")
        else:
            run(["ollama", "pull", model], f"ollama pull {model} ({size})",
                timeout=1800)

    print("\n=== MLX Model (3.8 GB) ===")
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if _mlx_cached():
        print("  mlx-community/Mistral-7B-Instruct-v0.3-4bit already cached, skipping")
    else:
        run(["uv", "sync"], "uv sync (ensure mlx-lm installed)")
        run(
            [venv_python, "-c",
             "from mlx_lm import load; load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')"],
            "MLX Mistral 7B 4bit",
            timeout=600,
        )


def download_data():
    print("\n=== Human Data Mining ===")
    print("  (Gesetze ~50 MB + Bundestag ~320 MB + OpenLegalData ~16.9 GB)")
    print("  Each source skips if already downloaded.")
    print("  OpenLegalData requires HF login + accepted conditions:")
    print("    hf auth login")
    print("    https://huggingface.co/datasets/openlegaldata/court-decisions-germany")
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    run([venv_python, "main.py", "--mine"], "mine all sources",
        timeout=7200)


def main():
    parser = argparse.ArgumentParser(
        description="Download all project assets (idempotent)"
    )
    parser.add_argument("--deps", action="store_true",
                        help="Python deps + spaCy (~2.5 GB)")
    parser.add_argument("--models", action="store_true",
                        help="Ollama + MLX models (~36 GB)")
    parser.add_argument("--data", action="store_true",
                        help="Human data via mining (~17 GB)")
    args = parser.parse_args()

    any_specific = args.deps or args.models or args.data

    if not any_specific:
        print("Total download estimate: ~55 GB")
        print("  Python deps + spaCy:  ~2.5 GB")
        print("  Ollama + MLX models:  ~36 GB")
        print("  Human data:           ~17 GB")
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
