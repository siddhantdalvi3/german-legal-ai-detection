from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 50000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(2, 5),
        analyzer="char",
        sublinear_tf=True,
    )
