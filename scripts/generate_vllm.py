import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_vllm")

from config import AI_GENERATED_DIR, TEMPERATURES
from scripts.generate_ai import _generate_synthetic_topics, TOPICS_DIR


PROMPT_TEMPLATE = (
    "Sie sind ein deutscher Jurist. Schreiben Sie einen deutschen Rechtstext "
    "zum folgenden Thema. Antworten Sie nur mit dem Text selbst, ohne "
    "Einleitung oder Erklärung.\n\n"
    "Thema: {topic}\n\n"
    "Text:"
)


def load_topics():
    combined = TOPICS_DIR / "all_topics.jsonl"
    if combined.exists():
        lines = [l.strip() for l in combined.read_text(encoding="utf-8").splitlines() if l.strip()]
        topics = [json.loads(l) for l in lines]
        logger.info(f"Loaded {len(topics):,} topics from {combined}")
        return topics
    topics = _generate_synthetic_topics()
    logger.info(f"Using {len(topics):,} synthetic topics")
    return topics


def get_checkpoint_path(model_key: str, temperature: float) -> Path:
    temp_str = str(temperature).replace(".", "_")
    return AI_GENERATED_DIR / f"{model_key}__temp_{temp_str}.jsonl"


def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


HF_MODEL_KEYS = {
    "gemma4": "google/gemma-4-12b-it",
    "phi4": "microsoft/phi-4",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-Small-Instruct-24B-Base-2501",
}


def main():
    parser = argparse.ArgumentParser(description="Generate AI legal text using vLLM")
    parser.add_argument("--models", nargs="+", default=["gemma4"], choices=list(HF_MODEL_KEYS),
                        help="Models to generate with")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of texts per model/temp combination")
    parser.add_argument("--temps", nargs="+", type=float, default=TEMPERATURES,
                        help="Temperature values")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for gated models")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    from vllm import LLM, SamplingParams

    topics = load_topics()
    AI_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        hf_name = HF_MODEL_KEYS[model_key]
        logger.info(f"Loading {model_key} ({hf_name})...")
        t0 = time.time()
        llm = LLM(
            model=hf_name,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            trust_remote_code=True,
        )
        logger.info(f"Model loaded in {time.time()-t0:.1f}s")

        for temp in args.temps:
            ckpt_path = get_checkpoint_path(model_key, temp)
            done = load_checkpoint(ckpt_path)
            remaining = args.count - done

            if remaining <= 0:
                logger.info(f"[{model_key} | t={temp}] Already done ({done}/{args.count})")
                continue

            logger.info(f"[{model_key} | t={temp}] Generating {remaining} texts "
                       f"(checkpoint had {done})")

            batch_size = 64
            params = SamplingParams(
                temperature=temp,
                max_tokens=256,
                top_p=0.95,
            )

            from tqdm import tqdm
            pbar = tqdm(total=args.count, initial=done, desc=f"{model_key} t={temp}")

            with open(ckpt_path, "a", encoding="utf-8") as f:
                for batch_start in range(0, remaining, batch_size):
                    batch_end = min(batch_start + batch_size, remaining)
                    batch_indices = range(done + batch_start, done + batch_end)

                    prompts = []
                    batch_topics = []
                    for idx in batch_indices:
                        entry = topics[idx % len(topics)]
                        topic = entry["topic"]
                        source = entry.get("source", "synthetic")
                        prompt = PROMPT_TEMPLATE.format(topic=topic)
                        prompts.append(prompt)
                        batch_topics.append((topic, source))

                    outputs = llm.generate(prompts, params)
                    for output, (topic, source) in zip(outputs, batch_topics):
                        response = output.outputs[0].text.strip()
                        f.write(json.dumps({
                            "topic": topic,
                            "topic_source": source,
                            "model": model_key,
                            "temperature": temp,
                            "response": response,
                        }, ensure_ascii=False) + "\n")
                        f.flush()
                        pbar.update(1)

            pbar.close()

        del llm

    logger.info("Done!")


if __name__ == "__main__":
    main()
