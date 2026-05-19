import json
import logging
import subprocess
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    AI_GENERATED_DIR,
    OLLAMA_MODELS,
    MLX_MODEL,
    TEMPERATURES,
    SENTENCES_PER_COMBINATION,
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
        "stream": False,
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def mlx_generate(prompt: str, temperature: float) -> str:
    result = subprocess.run(
        [
            "mlx_lm.generate",
            "--model", MLX_MODEL,
            "--prompt", prompt,
            "--temp", str(temperature),
            "--max-tokens", "500",
        ],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout.strip()


def get_checkpoint_path(model_key: str, temperature: float) -> Path:
    temp_str = str(temperature).replace(".", "_")
    return AI_GENERATED_DIR / f"{model_key}__temp_{temp_str}.jsonl"


def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    prompts_done = set()
    with open(path) as f:
        for line in f:
            try:
                data = json.loads(line)
                prompts_done.add(data.get("topic", ""))
            except json.JSONDecodeError:
                continue
    return prompts_done


def generate_ai_corpus():
    AI_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if not check_ollama_running():
        logger.info("Starting Ollama server...")
        start_ollama()

    for model in OLLAMA_MODELS:
        logger.info(f"Pulling model: {model}")
        subprocess.run(["ollama", "pull", model], capture_output=True, timeout=600)

    total_sentences = 0
    target = SENTENCES_PER_COMBINATION

    for model in OLLAMA_MODELS + [f"mlx:{MLX_MODEL}"]:
        for temp in TEMPERATURES:
            model_key = model.replace("/", "_").replace(":", "_")
            ckpt_path = get_checkpoint_path(model_key, temp)
            done_topics = load_checkpoint(ckpt_path)
            sentences_in_batch = len(done_topics)

            logger.info(
                f"[{model_key} | temp={temp}] "
                f"Already have {sentences_in_batch} / {target} sentences"
            )

            if sentences_in_batch >= target:
                total_sentences += sentences_in_batch
                continue

            pbar = tqdm(
                total=target,
                initial=sentences_in_batch,
                desc=f"{model_key} t={temp}",
            )
            topic_idx = 0

            with open(ckpt_path, "a") as f:
                while sentences_in_batch < target:
                    topic = TOPICS[topic_idx % len(TOPICS)]
                    topic_idx += 1

                    if topic in done_topics:
                        continue

                    prompt = PROMPT_TEMPLATE.format(topic=topic)

                    try:
                        if model.startswith("mlx:"):
                            response = mlx_generate(prompt, temp)
                        else:
                            response = ollama_generate(model, prompt, temp)

                        record = {
                            "topic": topic,
                            "model": model_key,
                            "temperature": temp,
                            "response": response,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        sentences_in_batch += 1
                        total_sentences += 1
                        done_topics.add(topic)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Generation failed: {e}")
                        time.sleep(5)

            pbar.close()
            logger.info(
                f"Finished {model_key} temp={temp}: {sentences_in_batch} sentences"
            )

    logger.info(f"AI generation complete. Total responses: {total_sentences}")


if __name__ == "__main__":
    generate_ai_corpus()
