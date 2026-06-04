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


# Common German abbreviations that end with a period
_ABBREVIATIONS = {
    "Abs", "Art", "Az", "Bd", "Bf", "BGB", "BMJ", "BVerfG", "BVerwG", "BGH",
    "bzw", "ca", "ders", "dies", "dgl", "d.h", "d.i", "Dr", "etc", "e.V",
    "eG", "ff", "gem", "ggf", "ggfs", "ggü", "h.M", "Hrsg", "i.d",
    "i.e", "i.S", "i.V", "i.W", "jew", "lit", "m.a", "m.E", "m.w",
    "Nr", "o.g", "Prof", "Rspr", "S", "s.", "sog", "st.", "str",
    "u.a", "u.ä", "u.U", "usw", "v.a", "v.Chr", "vgl", "z.B", "z.T",
    "z.Z", "Abs", "Anm", "Aufl", "Bd", "Bez", "Bl", "bzw", "d.h",
    "ds", "einschl", "Erg", "etc", "Ev", "f", "ff", "ggf", "h.M",
    "Hrsg", "i.d.R", "i.S.d", "i.S.v", "i.V.m", "i.Ü", "i.e",
    "Jg", "Jur", "Kap", "lit", "m.a.W", "m.E", "m.w.N", "Mitgl",
    "Nachw", "Nr", "o.g", "Prof", "Rspr", "S", "s.", "s.o", "s.u",
    "sog", "Sp", "str", "Tab", "Tel", "Tsd", "u.a", "u.ä", "u.U",
    "u.v.m", "usw", "v.a", "v.Chr", "v.H", "vgl", "z.B", "z.T",
    "z.Z", "zzgl",
}

_ABBR_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\.(?=\s|$)",
    re.IGNORECASE,
)

# Characters that can end a sentence in German legal text
_SENTENCE_END = re.compile(r"([.!?])(\s+)([A-Z0-9\"'(„«])")

# Pre-compiled abbreviation suffix check
_ABBR_SET = {a.lower() for a in _ABBREVIATIONS}


def _is_abbreviation(word: str) -> bool:
    """Check if a word ending with period is a known abbreviation."""
    key = word.rstrip(".").lower()
    if key in _ABBR_SET:
        return True
    # Single capital letter (like "S.", "A.")
    if len(key) == 1 and key.isalpha():
        return True
    # Ordinal number (like "1.", "2.")
    if key.isdigit():
        return True
    return False


def fast_sentence_split(text: str) -> list[str]:
    """Fast regex-based sentence splitter for German legal texts.

    ~100-1000x faster than spaCy, good enough for court decisions.
    """
    if len(text) < 20:
        return []

    text = text.strip()
    if not text:
        return []

    # Find all potential sentence boundaries
    candidates = []
    for m in _SENTENCE_END.finditer(text):
        candidates.append((m.start(), m.end(), m.group(1), m.end(1)))

    if not candidates:
        if len(text) >= 20:
            return [text]
        return []

    sentences = []
    pos = 0
    for start, end, punct, content_end in candidates:
        # Check abbreviation
        if punct == ".":
            word_start = text.rfind(" ", pos, start)
            if word_start == -1:
                word_start = pos
            else:
                word_start += 1
            word_before = text[word_start:start]
            if _is_abbreviation(word_before):
                continue

        sentence = text[pos:content_end].strip()
        if len(sentence) >= 20:
            sentences.append(sentence)
        pos = content_end + (end - content_end)

    # Last segment
    tail = text[pos:].strip()
    if len(tail) >= 20:
        sentences.append(tail)

    return sentences


def batch_sentence_split(texts: list[str], batch_size: int = 256, min_len: int = 20) -> list[list[str]]:
    """Split multiple texts using spaCy's parser in batches.

    Returns a list of sentence lists (one per input text).
    """
    nlp = get_nlp()
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= min_len]
        results.append(sents)
    return results


def is_short(s: str, min_len: int = 20) -> bool:
    return len(s) < min_len
