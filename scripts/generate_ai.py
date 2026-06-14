import json
import logging
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from tqdm import tqdm

from config import (
    AI_GENERATED_DIR,
    AVAILABLE_MODELS,
    DEFAULT_GENERATION_MODELS,
    TEMPERATURES,
    SENTENCES_PER_COMBINATION,
    is_macos,
)
from scripts.extract_topics import TOPICS_DIR
from utils.mining import logger

OLLAMA_API = "http://localhost:11434/api/generate"

PROMPT_TEMPLATES: dict[str, str] = {
    "gesetze": (
        "Du bist Jurist im Bundesministerium der Justiz. "
        "Schreibe einen formellen juristischen Text auf Deutsch über folgendes Thema: {topic}. "
        "Verwende präzise juristische Fachsprache und einen offiziellen Amtsstil. "
        "Der Text soll 5-10 Sätze lang sein. "
        "Beziehe dich konkret auf die einschlägigen Gesetze und Paragraphen."
    ),
    "dip": (
        "Du bist wissenschaftlicher Mitarbeiter im Deutschen Bundestag. "
        "Verfasse einen sachlichen parlamentarischen Text auf Deutsch zu folgendem Gegenstand: {topic}. "
        "Der Text soll 5-10 Sätze lang sein. "
        "Konzentriere dich auf die politische und rechtliche Einordnung des Themas."
    ),
    "gesp": (
        "Du bist Richter an einem deutschen Gericht. "
        "Verfasse einen Abschnitt einer gerichtlichen Entscheidung auf Deutsch zu folgendem Sachverhalt: {topic}. "
        "Verwende die typische Sprache und Struktur deutscher Urteile. "
        "Der Text soll 5-10 Sätze lang sein."
    ),
}

DEFAULT_PROMPT_TEMPLATE = PROMPT_TEMPLATES["gesetze"]

TOPICS_CACHE = None


def check_ollama_running() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def start_ollama():
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if check_ollama_running():
            logger.info("Ollama server started")
            return
        time.sleep(1)
    raise RuntimeError("Failed to start Ollama server")


def _generate_synthetic_topics() -> list[dict]:
    parts = [
        "Vertragsschluss", "Willenserklärung", "Anfechtung", "Rücktritt", "Schadensersatz",
        "Sachmängelhaftung", "Kündigung", "Verjährung", "Vollmacht", "Bereicherung",
        "Unerlaubte Handlung", "Eigentumsübertragung", "Pfandrecht", "Bürgschaft",
        "Ermessen", "Verhältnismäßigkeit", "Rechtssicherheit", "Vertrauensschutz",
        "Rechtsbehelf", "Widerspruch", "Klagebefugnis", "Beweislast",
        "Zeugenvernehmung", "Beschlagnahme", "Durchsuchung", "Rechtsmittel",
        "Berufung", "Revision", "Amtshaftung", "Gleichbehandlung",
    ]
    areas = [
        "Bürgerliches Recht", "Strafrecht", "Öffentliches Recht", "Verwaltungsrecht",
        "Steuerrecht", "Arbeitsrecht", "Mietrecht", "Familienrecht", "Erbrecht",
        "Gesellschaftsrecht", "Wettbewerbsrecht", "Verfassungsrecht", "Sozialrecht",
        "Europarecht", "Baurecht", "Umweltrecht", "Versicherungsrecht",
    ]
    topics = []
    for p in parts:
        for a in areas:
            topics.append({"topic": f"Die {p} im {a}", "source": "synthetic"})
            topics.append({"topic": f"Die Rechtsfolgen von {p} im {a}", "source": "synthetic"})
            topics.append({"topic": f"Die Voraussetzungen der {p} im {a}", "source": "synthetic"})
            topics.append({"topic": f"Die Bedeutung von {p} für das {a}", "source": "synthetic"})
    for p in parts:
        topics.append({"topic": f"Die rechtlichen Grundlagen der {p}", "source": "synthetic"})
        topics.append({"topic": f"Aktuelle Entwicklungen bei der {p}", "source": "synthetic"})
        topics.append({"topic": f"Die {p} in der Rechtsprechung", "source": "synthetic"})
        topics.append({"topic": f"Voraussetzungen und Grenzen der {p}", "source": "synthetic"})
    laws = ["BGB", "StGB", "VwVfG", "AO", "GG", "ZPO", "StPO", "HGB", "GmbHG", "AktG"]
    for p in parts:
        for l in laws:
            par = random.randint(1, 500)
            topics.append({"topic": f"§ {par} {l}: Die {p}", "source": "synthetic"})
            topics.append({"topic": f"Die Anwendung von § {par} {l} auf die {p}", "source": "synthetic"})
    random.shuffle(topics)
    return topics[:50000]
    global TOPICS_CACHE
    if TOPICS_CACHE is not None:
        return TOPICS_CACHE

    combined_path = TOPICS_DIR / "all_topics.jsonl"
    if combined_path.exists():
        lines = [l.strip() for l in combined_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        TOPICS_CACHE = [json.loads(l) for l in lines]
        logger.info(f"Loaded {len(TOPICS_CACHE):,} topics from combined pool ({TOPICS_DIR / 'all_topics.jsonl'})")
        return TOPICS_CACHE

    logger.info("Combined topics not found. Generating synthetic topic pool...")
    TOPICS_CACHE = _generate_synthetic_topics()
    logger.info(f"Generated {len(TOPICS_CACHE):,} synthetic topics for generation")
    return TOPICS_CACHE


def get_topic_entry(topics: list[dict], idx: int) -> dict:
    if not topics:
        return {"topic": "Rechtliche Grundlagen und aktuelle Entwicklungen", "source": "gesetze"}
    return topics[idx % len(topics)]


def ollama_generate(model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": 256,
        "num_ctx": 2048,
        "keep_alive": "5m",
        "stream": False,
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["response"].strip()


_mlx_cache: dict[str, tuple] = {}


def mlx_generate(model_name: str, prompt: str, temperature: float) -> str:
    global _mlx_cache

    if model_name not in _mlx_cache:
        logger.info(f"Loading MLX model: {model_name}")
        from mlx_lm import load
        model, tokenizer = load(model_name)
        _mlx_cache[model_name] = (model, tokenizer)
        logger.info(f"MLX model loaded: {model_name}")
    else:
        model, tokenizer = _mlx_cache[model_name]

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    response = generate(
        model, tokenizer,
        prompt=prompt,
        sampler=make_sampler(temp=temperature),
        max_tokens=500,
        verbose=False,
    )
    return response.strip()

def get_checkpoint_path(model_key: str, temperature: float) -> Path:
    temp_str = str(temperature).replace(".", "_")
    return AI_GENERATED_DIR / f"{model_key}__temp_{temp_str}.jsonl"


def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
    return count


def generate_ai_corpus(models: list[str] | None = None, temps: list[float] | None = None, device: int | None = None, count: int | None = None):
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        logger.info(f"Pinned to GPU {device} (CUDA_VISIBLE_DEVICES={device})")

    AI_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = DEFAULT_GENERATION_MODELS
    else:
        from config import AVAILABLE_MODELS
        resolved = []
        for key in models:
            if key not in AVAILABLE_MODELS:
                logger.warning(f"Unknown model key '{key}'. Available: {list(AVAILABLE_MODELS)}")
                continue
            info = AVAILABLE_MODELS[key]
            mtype = info["type"]
            mname = info["name"]
            if mtype == "ollama":
                resolved.append(mname)
            elif mtype == "mlx":
                resolved.append(f"mlx:{mname}")
            elif mtype == "mlx_vlm":
                resolved.append(f"mlx_vlm:{mname}")
        if not resolved:
            logger.error("No valid models selected. Aborting.")
            return
        models = resolved

    has_mlx = any(m.startswith("mlx:") for m in models)
    if has_mlx and not is_macos():
        logger.warning("MLX models are only supported on macOS. Skipping MLX models.")
        models = [m for m in models if not m.startswith("mlx:")]
        if not models:
            logger.error("No models remaining after removing MLX. Aborting.")
            return

    logger.info(f"Models to run: {models}")

    if not check_ollama_running():
        logger.info("Starting Ollama server...")
        start_ollama()

    for model in [m for m in models if not m.startswith("mlx:")]:
        logger.info(f"Pulling model: {model}")
        subprocess.run(["ollama", "pull", model], capture_output=True, timeout=600)

    active_temps = temps if temps is not None else TEMPERATURES
    logger.info(f"Temperatures: {active_temps}")

    topics = load_topics()
    total_sentences = 0
    target = count if count is not None else SENTENCES_PER_COMBINATION
    if count is not None:
        logger.info(f"Overriding target count: {target} per combination")

    for model in models:
        # Determine concurrency based on model size
        if model.startswith("mlx:"):
            concurrency = 1  # MLX models are loaded per-process; avoid OOM
        elif "deepseek-r1:70b" in model:
            concurrency = 1  # 70B: ~43GB, fill one H100
        elif any(m in model for m in ("steuerllm", "gemma3:27b", "qwen3:30b", "mistral-small:24b")):
            concurrency = 2  # 24-30B: ~14-20GB, 2 per H100
        elif any(m in model for m in ("gemma4", "phi4", "mlx_vlm")):
            concurrency = 3  # 12-14B: ~8-10GB, 3-4 per H100
        else:
            concurrency = 4  # small models: qwen2.5:7b, mistral:7b

        for temp in active_temps:
            model_key = model.replace("/", "_").replace(":", "_")
            ckpt_path = get_checkpoint_path(model_key, temp)
            done_count = load_checkpoint(ckpt_path)
            sentences_in_batch = done_count

            logger.info(
                f"[{model_key} | temp={temp}] "
                f"Already have {sentences_in_batch} / {target} sentences"
            )

            if sentences_in_batch >= target:
                total_sentences += sentences_in_batch
                continue

            remaining = target - sentences_in_batch
            pbar = tqdm(total=target, initial=sentences_in_batch, desc=f"{model_key} t={temp}")
            topic_offset = sentences_in_batch

            with open(ckpt_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {}
                while sentences_in_batch < target:
                    # Submit up to concurrency prompts at once
                    while len(futures) < concurrency and sentences_in_batch + len(futures) < target:
                        idx = topic_offset + sentences_in_batch + len(futures)
                        entry = get_topic_entry(topics, idx)
                        topic = entry["topic"]
                        source = entry.get("source", "gesetze")
                        template = PROMPT_TEMPLATES.get(source, DEFAULT_PROMPT_TEMPLATE)
                        prompt = template.format(topic=topic)

                        if model.startswith("mlx:"):
                            mlx_name = model[4:]  # strip "mlx:" prefix
                            future = pool.submit(mlx_generate, mlx_name, prompt, temp)
                        else:
                            future = pool.submit(ollama_generate, model, prompt, temp)
                        futures[future] = (topic, source)

                    # Collect completed results
                    done_futures = {f for f in futures if f.done()}
                    for future in done_futures:
                        topic, source = futures.pop(future)
                        try:
                            response = future.result()
                            f.write(json.dumps({
                                "topic": topic,
                                "topic_source": source,
                                "model": model_key,
                                "temperature": temp,
                                "response": response,
                            }, ensure_ascii=False) + "\n")
                            f.flush()
                            sentences_in_batch += 1
                            total_sentences += 1
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Generation failed: {e}")

                    if not done_futures:
                        time.sleep(0.1)

            pbar.close()
            logger.info(
                f"Finished {model_key} temp={temp}: {sentences_in_batch} sentences"
            )

    logger.info(f"AI generation complete. Total responses: {total_sentences}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--temps", nargs="*", type=float, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    generate_ai_corpus(models=args.models, temps=args.temps, device=args.device, count=args.count)
