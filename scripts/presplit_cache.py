"""Pre-sentence-split cached whole-document JSONL files using spaCy parser.

Processes files in parallel (one process per file) using multiprocessing.
Each worker loads spaCy independently for true parallelism.
"""
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import spacy
from tqdm import tqdm

logger = logging.getLogger(__name__)

BATCH_SIZE = 64
N_WORKERS = 4


def _process_file(args):
    infile, output_dir, source_prefix = args
    name = infile.stem
    outfile = output_dir / f"{name}.jsonl"
    if outfile.exists():
        cnt = sum(1 for _ in open(outfile, encoding="utf-8", errors="replace"))
        return name, 0, cnt, "skipped"

    nlp = spacy.load("de_core_news_sm")

    with open(infile, encoding="utf-8", errors="replace") as f:
        docs = [json.loads(line) for line in f]

    texts = [(d["text"], d) for d in docs if len(d.get("text", "")) >= 100]
    count = 0

    with open(outfile, "w", encoding="utf-8") as f_out:
        for doc, record in nlp.pipe(texts, batch_size=BATCH_SIZE, as_tuples=True):
            source = record.get("source", f"{source_prefix}_{name}")
            for sent in doc.sents:
                text = sent.text.strip()
                if len(text) >= 20:
                    f_out.write(json.dumps({
                        "text": text,
                        "source": source,
                    }, ensure_ascii=False) + "\n")
                    count += 1

    return name, len(texts), count, "done"


def presplit_dir(input_dir: Path, output_dir: Path, source_prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    infiles = sorted(input_dir.glob("*.jsonl"))
    if not infiles:
        logger.info(f"  No files found in {input_dir}")
        return

    args_list = [(f, output_dir, source_prefix) for f in infiles]

    with Pool(N_WORKERS) as pool:
        for name, ndocs, nsents, status in pool.imap_unordered(_process_file, args_list):
            if status == "skipped":
                logger.info(f"  {name}: {nsents} sentences cached, skipping")
            else:
                logger.info(f"  {name}: {nsents} sentences from {ndocs} docs")


def main():
    base = Path("data")
    logger.info("Pre-splitting Legal Commons with spaCy (parser)...")
    presplit_dir(base / "legal_commons" / "cache",
                  base / "legal_commons" / "cache_sentences",
                  "legal_commons")
    logger.info("Pre-splitting Fobbe with spaCy (parser)...")
    presplit_dir(base / "fobbe",
                  base / "fobbe" / "cache_sentences",
                  "fobbe")
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
