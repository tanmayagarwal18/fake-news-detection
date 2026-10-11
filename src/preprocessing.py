"""
preprocessing.py
----------------
spaCy-based NLP preprocessing pipeline for Fake News Detection.

Pipeline:
    Lowercase
    -> Remove URLs / HTML
    -> Tokenization using spaCy
    -> Stopword removal
    -> Punctuation removal
    -> Lemmatization
    -> Return cleaned text

spaCy is used instead of NLTK because it provides a reliable,
production-quality tokenization and lemmatization pipeline.
"""

import re

import spacy

from utils import get_logger

logger = get_logger("preprocessing")



MODEL_NAME = "en_core_web_sm"

logger.info(f"Loading spaCy model: {MODEL_NAME}")

nlp = spacy.load(
    MODEL_NAME,
    disable=["parser", "ner"]
)

logger.info("spaCy model loaded successfully")



URL_RE = re.compile(
    r"http\S+|www\.\S+",
    re.IGNORECASE
)

HTML_RE = re.compile(
    r"<.*?>",
    re.DOTALL
)



def clean_text(text: str) -> str:
    """
    Clean and lemmatize one news article.

    Used mainly for testing/debugging.
    For the complete dataset, clean_series() is preferred
    because it uses spaCy's efficient nlp.pipe().
    """

    if not isinstance(text, str):
        return ""

    if not text.strip():
        return ""

    text = text.lower()

    text = URL_RE.sub(" ", text)

    text = HTML_RE.sub(" ", text)

    doc = nlp(text)

    tokens = []

    for token in doc:

        if token.is_space:
            continue

        if token.is_punct:
            continue

        if token.is_stop:
            continue

        if not token.is_alpha:
            continue

        
        lemma = token.lemma_.lower().strip()

        
        if len(lemma) <= 2:
            continue

        tokens.append(lemma)

    return " ".join(tokens)


# ============================================================
# CLEAN COMPLETE SERIES
# ============================================================

def clean_series(series):
    """
    Efficiently preprocess an entire pandas Series using
    spaCy's nlp.pipe().

    This is much faster than calling clean_text() separately
    for every article.
    """

    texts = (
        series
        .fillna("")
        .astype(str)
        .tolist()
    )

    cleaned = []

    logger.info(
        f"Starting spaCy preprocessing for {len(texts):,} documents..."
    )

    for doc in nlp.pipe(
        texts,
        batch_size=32
    ):

        tokens = []

        for token in doc:

            if token.is_space:
                continue

            if token.is_punct:
                continue

            if token.is_stop:
                continue

            if not token.is_alpha:
                continue

            lemma = token.lemma_.lower().strip()

            if len(lemma) <= 2:
                continue

            tokens.append(lemma)

        cleaned.append(" ".join(tokens))

    logger.info(
        "spaCy preprocessing completed successfully."
    )

    return cleaned


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample = """
    BREAKING!!! Scientists were studying fake news stories.
    Journalists were reporting their findings.
    Visit https://example.com/news
    """

    print("\nOriginal:")
    print(sample)

    print("\nCleaned:")

    print(
        clean_text(sample)
    )