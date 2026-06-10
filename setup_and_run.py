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

    # ── Setup ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Installing dependencies")
    print("=" * 60)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], cwd=here)
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=here)
    run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], cwd=here)

    # ── Models ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Pulling Ollama models")
    print("=" * 60)
    for model in ["qwen2.5:7b", "mistral", "gemma4:12b"]:
        run(["ollama", "pull", model], cwd=here)
    print()

    # ── Mining ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Mining all human text sources")
    print("=" * 60)
    print("This downloads ~17 GB of legal text data. May take 30-60 min.")
    input("Press Enter to start mining...")
    run([sys.executable, "main.py", "--mine", "--fobbe", "--legal-commons", "--rii"], cwd=here)

    # ── Generation ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: AI text generation (4-night schedule)")
    print("=" * 60)
    print("Time window: 4pm – 7:30am (15.5h/night)")
    print()
    print("Night 1: qwen2.5 (temps 0.3, 0.7)  ~16h")
    print("Night 2: mistral  (temps 0.3, 0.7)  ~16h")
    print("Night 3: gemma4   (temp  0.3)       ~12h")
    print("Night 4: gemma4   (temp  0.7)       ~12h")
    print()

    night = prompt("Which night? (1/2/3/4)", "1")
    night_cmds = {
        "1": ["--models", "qwen2.5", "--temps", "0.3", "0.7"],
        "2": ["--models", "mistral", "--temps", "0.3", "0.7"],
        "3": ["--models", "gemma4", "--temps", "0.3"],
        "4": ["--models", "gemma4", "--temps", "0.7"],
    }
    if night in night_cmds:
        gen_cmd = [sys.executable, "main.py", "--generate"] + night_cmds[night]
        run(gen_cmd, cwd=here)
        print(f"\nNight {night} done!")
    else:
        print(f"Invalid night: {night}")

    print("\nAfter all nights complete, copy data/ai_generated/*.jsonl back to the main Mac.")
    print()


if __name__ == "__main__":
    main()
