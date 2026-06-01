import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    AI_GENERATED_DIR,
    AVAILABLE_MODELS,
    DEFAULT_GENERATION_MODELS,
    OLLAMA_MODELS,
    MLX_MODEL,
    MLX_VLM_MODEL,
    TEMPERATURES,
    SENTENCES_PER_COMBINATION,
    is_macos,
)
from utils.mining import logger

OLLAMA_API = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = (
    "Du bist Jurist im Bundesministerium. "
    "Schreibe einen formellen juristischen Text auf Deutsch über folgendes Thema: {topic}. "
    "Verwende präzise juristische Fachsprache und einen offiziellen Amtsstil. "
    "Der Text soll 5-10 Sätze lang sein."
)

TOPICS = [
    "Die Voraussetzungen einer wirksamen Willenserklärung im Bürgerlichen Recht",
    "Die Haftung des Verkäufers für Sachmängel nach § 437 BGB",
    "Die Anfechtung einer Verwaltungsakt nach § 48 VwVfG",
    "Die Grundsätze der Verhältnismäßigkeit im öffentlichen Recht",
    "Die Rechtsprechung des Bundesverfassungsgerichts zur Meinungsfreiheit",
    "Die Anforderungen an eine ordnungsgemäße Klageerhebung vor dem Verwaltungsgericht",
    "Die Voraussetzungen des polizeilichen Notstandes nach allgemeinem Gefahrenabwehrrecht",
    "Die Rechtsfolgen einer nichtigen Ehe nach § 1313 BGB",
    "Die Vergabe öffentlicher Aufträge nach dem Vergaberecht",
    "Die Haftung des Staates für Amtspflichtverletzungen nach § 839 BGB",
    "Die Auslegung von Steuergesetzen nach der Abgabenordnung",
    "Die Voraussetzungen der einstweiligen Anordnung nach § 123 VwGO",
    "Die Rechtsstellung des Betriebsrats nach dem Betriebsverfassungsgesetz",
    "Die Voraussetzungen der Pfändung von Arbeitseinkommen",
    "Die Wirksamkeit von Allgemeinen Geschäftsbedingungen im Rechtsverkehr",
    "Die Haftung des GmbH-Geschäftsführers bei Insolvenzverschleppung",
    "Die Anforderungen an einen wirksamen Arbeitsvertrag nach deutschem Recht",
    "Die Voraussetzungen der Enteignung nach Art. 14 GG",
    "Die Rechtsmittel gegen einen belastenden Verwaltungsakt",
    "Die Grundsätze der europäischen Datenschutzgrundverordnung",
    "Die Auswirkungen des § 242 StGB auf die tägliche Polizeiarbeit",
    "Die Anforderungen an eine wirksame Kündigung eines Mietverhältnisses",
    "Die Rechtsprechung zur Störerhaftung im Internetrecht",
    "Die Regeln zur Geschwindigkeitsüberschreitung im Straßenverkehr",
    "Die Voraussetzungen der Unterlassungsklage im Wettbewerbsrecht",
    "Die Berechnung des Pflichtteils nach dem Bürgerlichen Gesetzbuch",
    "Die Anforderungen an einen Bauantrag nach der Musterbauordnung",
    "Die Durchführung einer Hauptverhandlung im Strafprozess",
    "Die Grundlagen der Sozialversicherungspflicht von Arbeitnehmern",
    "Die Rechtsfolgen einer fehlerhaften Bilanzierung im Handelsrecht",
    "Die Voraussetzungen der Prozesskostenhilfe im Zivilprozess",
    "Die Haftung des Frachtführers nach dem Handelsgesetzbuch",
    "Die Beihilfefähigkeit von Aufwendungen im öffentlichen Dienst",
    "Die Anerkennung ausländischer Entscheidungen im Familienrecht",
    "Die Voraussetzungen des Besitzschutzes nach § 861 BGB",
]


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


def ollama_generate(model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": 256,
        "keep_alive": "5m",
        "stream": False,
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


_mlx_model = None
_mlx_tokenizer = None


def mlx_generate(prompt: str, temperature: float) -> str:
    global _mlx_model, _mlx_tokenizer

    if _mlx_model is None:
        logger.info(f"Loading MLX model: {MLX_MODEL}")
        from mlx_lm import load
        _mlx_model, _mlx_tokenizer = load(MLX_MODEL)
        logger.info("MLX model loaded")

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    response = generate(
        _mlx_model, _mlx_tokenizer,
        prompt=prompt,
        sampler=make_sampler(temp=temperature),
        max_tokens=500,
        verbose=False,
    )
    return response.strip()


_mlx_vlm_model = None
_mlx_vlm_processor = None


def mlx_vlm_generate(prompt: str, temperature: float) -> str:
    global _mlx_vlm_model, _mlx_vlm_processor

    if _mlx_vlm_model is None:
        logger.info(f"Loading MLX VLM model: {MLX_VLM_MODEL}")
        from mlx_vlm import load
        _mlx_vlm_model, _mlx_vlm_processor = load(MLX_VLM_MODEL)
        logger.info("MLX VLM model loaded")

    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.sample_utils import make_sampler
    from mlx_vlm.utils import prepare_inputs

    response = vlm_generate(
        _mlx_vlm_model, _mlx_vlm_processor,
        prompt=prompt,
        image=None,
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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
    return count


def generate_ai_corpus(models: list[str] | None = None):
    AI_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = DEFAULT_GENERATION_MODELS
    else:
        all_ollama = OLLAMA_MODELS
        ollama_selected = [m for m in all_ollama if any(k in m for k in models)]
        mlx_selected = [f"mlx:{MLX_MODEL}"] if "mlx" in models else []
        mlx_vlm_selected = [f"mlx_vlm:{MLX_VLM_MODEL}"] if "mlx_gemma4" in models else []
        if not ollama_selected and not mlx_selected and not mlx_vlm_selected and models:
            logger.warning(f"No models matched: {models}. Available: {list(AVAILABLE_MODELS)}")
            return
        models = ollama_selected + mlx_selected + mlx_vlm_selected

    has_mlx = any(m.startswith("mlx:") for m in models)
    has_mlx_vlm = any(m.startswith("mlx_vlm:") for m in models)
    if (has_mlx or has_mlx_vlm) and not is_macos():
        logger.warning("MLX models are only supported on macOS. Skipping MLX models.")
        models = [m for m in models if not m.startswith("mlx") and not m.startswith("mlx_vlm")]
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

    total_sentences = 0
    target = SENTENCES_PER_COMBINATION

    for model in models:
        # Determine concurrency based on model size
        if any(m in model for m in ("gemma4", "qwen3:14b", "mlx_vlm")):
            concurrency = 2
        elif any(m in model for m in ("gemma3:12b", "mlx:")):
            concurrency = 3
        else:
            concurrency = 4  # small models: qwen2.5:7b, mistral

        for temp in TEMPERATURES:
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

            with open(ckpt_path, "a") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {}
                while sentences_in_batch < target:
                    # Submit up to concurrency prompts at once
                    while len(futures) < concurrency and sentences_in_batch + len(futures) < target:
                        idx = topic_offset + sentences_in_batch + len(futures)
                        topic = TOPICS[idx % len(TOPICS)]
                        prompt = PROMPT_TEMPLATE.format(topic=topic)

                        if model.startswith("mlx_vlm:"):
                            future = pool.submit(mlx_vlm_generate, prompt, temp)
                        elif model.startswith("mlx:"):
                            future = pool.submit(mlx_generate, prompt, temp)
                        else:
                            future = pool.submit(ollama_generate, model, prompt, temp)
                        futures[future] = topic

                    # Collect completed results
                    done_futures = {f for f in futures if f.done()}
                    for future in done_futures:
                        topic = futures.pop(future)
                        try:
                            response = future.result()
                            f.write(json.dumps({
                                "topic": topic,
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
    generate_ai_corpus()
