#!/usr/bin/env python3
"""One-click setup + mine + generate for CUDA systems."""
import subprocess
import sys
from pathlib import Path

VENV_DIR = Path(".venv")
PROJECT_DIR = Path(__file__).parent


def in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def create_venv_and_restart():
    print("=" * 60)
    print("STEP 0: Creating virtual environment...")
    print("=" * 60)
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    python = VENV_DIR / ("Scripts" if sys.platform == "win32" else "bin") / "python.exe" if sys.platform == "win32" else VENV_DIR / "bin" / "python"
    print(f"Re-running inside venv: {python}")
    subprocess.run([str(python), __file__] + sys.argv[1:])
    sys.exit()


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def prompt(msg: str, default: str = "") -> str:
    if default:
        val = input(f"{msg} [{default}]: ").strip()
        return val or default
    return input(f"{msg}: ").strip()


def main():
    if not in_venv():
        create_venv_and_restart()
        return

    here = PROJECT_DIR
    os = sys.platform

    # ── Setup ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Installing dependencies")
    print("=" * 60)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], cwd=here)
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=here)
    run([sys.executable, "-m", "pip", "install", "germanlegaltexts", "huggingface-hub"], cwd=here)
    run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], cwd=here)

    # ── HF Token ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: HuggingFace authentication (for OpenLegalData)")
    print("=" * 60)
    token = prompt("Enter your HuggingFace token (or press Enter to skip OpenLegalData)")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("You also need to accept the dataset terms at:")
        print("  https://huggingface.co/datasets/openlegaldata/court-decisions-germany")
        input("Press Enter after accepting the terms...")
        use_openlegaldata = True
    else:
        print("Skipping OpenLegalData.")
        use_openlegaldata = False

    # ── Mining ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Mining all human text sources")
    print("=" * 60)
    print("This downloads ~37 GB of legal text data. May take 30-60 min.")
    input("Press Enter to start mining...")
    mine_cmd = [sys.executable, "main.py", "--mine", "--fobbe", "--legal-commons", "--rii"]
    if use_openlegaldata:
        mine_cmd.append("--openlegaldata")
    run(mine_cmd, cwd=here)

    # ── Generation ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: AI text generation")
    print("=" * 60)
    print("Available models:")
    print("  qwen2.5     mistral      deepseek    llama3.1")
    print("  gemma3      phi4         qwen3       gemma4")
    print()
    models = prompt("Models to generate (space-separated, e.g. 'qwen2.5 mistral')")
    temps = prompt("Temperatures (space-separated, e.g. '0.1 0.3 0.7')", "0.1 0.3 0.5 0.7 0.9 1.0")
    if not models:
        print("No models specified. Skipping generation.")
        print("\nDone! To generate later, run:")
        print("  uv run python main.py --generate --models ... --temps ...")
        return

    gen_cmd = [sys.executable, "main.py", "--generate", "--models", *models.split()]
    gen_cmd.extend(["--temps", *temps.split()])
    run(gen_cmd, cwd=here)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print("Copy all files from data/ai_generated/*.jsonl back to the main Mac.")
    print()


if __name__ == "__main__":
    main()
