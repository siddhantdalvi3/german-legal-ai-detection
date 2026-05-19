import logging
import re

import spacy
from spacy.lang.de import German

from config import SPACY_MODEL

logger = logging.getLogger(__name__)

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL)
            logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
        except OSError:
            logger.warning(f"Model {SPACY_MODEL} not found, using blank German")
            _nlp = German()
    return _nlp


BOILERPLATE_PATTERNS = [
    r"\+\+\+.*?\+\+\+",
    r"\(BGl\.\s*I\s*S\.\s*\d+\)",
    r"\(BGBl\.\s*I\s*\d+,\s*\d+\)",
    r"Fundstelle:\s*BGBl\.\s*[^)]+\)",
    r"^\s*§\s*\d+\s*$",
    r"^\s*Abs\.\s*\d+\s*$",
    r"\d+\.\s+\d+\.\s+\d+\.\s+",
]

_boilerplate_re = re.compile("|".join(BOILERPLATE_PATTERNS))


def is_boilerplate(text: str) -> bool:
    return bool(_boilerplate_re.search(text))


def clean_legal_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_xml(xml_content: str) -> list[str]:
    paragraphs = re.findall(r"<P>(.*?)</P>", xml_content, re.DOTALL)
    return [clean_legal_text(p) for p in paragraphs if p.strip()]


def sentence_segment(text: str) -> list[str]:
    nlp = get_nlp()
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) >= 20:
            sentences.append(s)
    return sentences


def is_short(s: str, min_len: int = 20) -> bool:
    return len(s) < min_len
