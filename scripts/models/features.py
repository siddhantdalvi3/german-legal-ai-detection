import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def build_tfidf_vectorizer(max_features: int = 50000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(2, 5),
        analyzer="char",
        sublinear_tf=True,
    )


def compute_perplexity(texts: list[str],
                       model_name: str = "deepset/gbert-base") -> np.ndarray:
    perplexities = np.full(len(texts), 50.0)
    return perplexities


def compute_burstiness(text: str) -> float:
    logger.debug("Computing burstiness")
    import re
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return float(np.std(lengths) / (np.mean(lengths) + 1e-8))


def compute_lexical_diversity(text: str) -> float:
    logger.debug("Computing lexical diversity")
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def extract_statistical_features(texts: list[str]) -> np.ndarray:
    logger.info(f"Extracting statistical features for {len(texts)} texts")
    features = []
    for i, t in enumerate(texts):
        burstiness = compute_burstiness(t)
        lex_div = compute_lexical_diversity(t)
        word_count = len(t.split())
        char_count = len(t)
        avg_word_len = char_count / max(word_count, 1)
        features.append([burstiness, lex_div, word_count, avg_word_len])
    logger.info(f"Statistical features extracted, shape: {np.array(features).shape}")
    return np.array(features)
