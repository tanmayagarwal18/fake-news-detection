"""
eda.py
------
Fast Exploratory Data Analysis for the Fake News Detection project.

Run from Jupyter:
    from eda import run_eda

    report = run_eda(
        csv_name="news.csv",
        text_col="text",
        title_col="title",
        label_col="label"
    )

Purpose:
    - Dataset statistics
    - Missing values
    - Duplicate values
    - Fake vs Real distribution
    - Article length distribution
    - Title length distribution
    - Most frequent words for each class

IMPORTANT:
    EDA uses lightweight text cleaning for the top-word plots.
    The actual ML pipeline still uses the proper NLP preprocessing
    implemented in preprocessing.py.
"""

import os
import sys
import re
from collections import Counter

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DATA_RAW_DIR, RESULTS_DIR, ensure_dirs, get_logger

logger = get_logger("eda")


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(csv_name: str = "news.csv") -> pd.DataFrame:
    """
    Load the raw dataset from data/raw/.
    """

    path = os.path.join(DATA_RAW_DIR, csv_name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset:\n{path}\n\n"
            f"Place your CSV inside data/raw/."
        )

    logger.info(f"Loading dataset: {path}")

    df = pd.read_csv(path)

    logger.info(
        f"Dataset loaded successfully: {len(df):,} rows"
    )

    return df


# ============================================================
# 2. BASIC DATASET REPORT
# ============================================================

def basic_report(
    df: pd.DataFrame,
    text_col: str,
    label_col: str
) -> dict:

    report = {
        "n_rows": int(len(df)),
        "n_missing_text": int(df[text_col].isna().sum()),
        "n_missing_label": int(df[label_col].isna().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {
            str(k): int(v)
            for k, v in df[label_col].value_counts().items()
        }
    }

    logger.info(f"Dataset report: {report}")

    return report


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str,
    out_path: str
):

    counts = df[label_col].value_counts()

    plt.figure(figsize=(6, 4))

    counts.plot(
        kind="bar",
        color=["#d62728", "#2ca02c"]
    )

    plt.title("Fake vs Real News Count")
    plt.xlabel("Label")
    plt.ylabel("Number of Articles")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150
    )

    plt.close()

    logger.info(f"Saved: {out_path}")


# ============================================================
# 4. ARTICLE / TITLE LENGTH DISTRIBUTION
# ============================================================

def plot_length_distribution(
    df: pd.DataFrame,
    col: str,
    title: str,
    out_path: str
):

    # Lightweight operation.
    # No NLP preprocessing is performed here.

    lengths = (
        df[col]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    plt.figure(figsize=(7, 4))

    plt.hist(
        lengths,
        bins=40
    )

    plt.title(title)
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150
    )

    plt.close()

    logger.info(f"Saved: {out_path}")


# ============================================================
# 5. LIGHTWEIGHT TEXT CLEANING FOR EDA
# ============================================================

def lightweight_clean(text: str) -> list:

    """
    Fast cleaning ONLY for EDA word-frequency analysis.

    This is deliberately NOT the project's final NLP preprocessing.

    Steps:
        - lowercase
        - remove URLs
        - remove HTML
        - remove punctuation
        - remove very short tokens
        - remove common English stopwords
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep alphabetic characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = text.split()

    # Basic stopword list.
    # This is only for visualization.
    stopwords = {
        "the", "a", "an", "and", "or", "but",
        "if", "then", "than", "of", "to", "in",
        "on", "for", "with", "at", "by", "from",
        "is", "are", "was", "were", "be", "been",
        "being", "this", "that", "these", "those",
        "it", "its", "as", "into", "about", "after",
        "before", "over", "under", "again", "further",
        "can", "could", "should", "would", "will",
        "may", "might", "must", "do", "does", "did",
        "have", "has", "had", "having",
        "i", "you", "he", "she", "we", "they",
        "them", "their", "his", "her", "our", "your",
        "my", "me", "who", "what", "which", "when",
        "where", "why", "how"
    }

    tokens = [
        token
        for token in tokens
        if len(token) > 2
        and token not in stopwords
    ]

    return tokens


# ============================================================
# 6. TOP WORDS
# ============================================================

def plot_top_words(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_value,
    title: str,
    out_path: str,
    top_n: int = 20
):

    logger.info(
        f"Calculating top words for class: {label_value}"
    )

    subset = df[
        df[label_col] == label_value
    ][text_col].fillna("").astype(str)

    counter = Counter()

    # Process articles one by one.
    # Much faster than running spaCy/NLTK preprocessing
    # over the complete dataset.

    for text in subset:
        counter.update(
            lightweight_clean(text)
        )

    common = counter.most_common(top_n)

    if not common:

        logger.warning(
            f"No words found for label={label_value}"
        )

        return

    words = [x[0] for x in common]
    counts = [x[1] for x in common]

    plt.figure(figsize=(8, 6))

    plt.barh(
        words[::-1],
        counts[::-1],
        color="#9467bd"
    )

    plt.title(title)
    plt.xlabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150
    )

    plt.close()

    logger.info(f"Saved: {out_path}")


# ============================================================
# 7. MAIN EDA FUNCTION
# ============================================================

def run_eda(
    csv_name: str = "news.csv",
    text_col: str = "text",
    title_col: str = "title",
    label_col: str = "label"
):

    ensure_dirs()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset(csv_name)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        text_col,
        label_col
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise KeyError(
            f"Expected columns {missing} not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    report = basic_report(
        df,
        text_col,
        label_col
    )

    # Save JSON summary
    from utils import save_json

    save_json(
        report,
        os.path.join(
            RESULTS_DIR,
            "eda_summary.json"
        )
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    plot_class_distribution(
        df,
        label_col,
        os.path.join(
            RESULTS_DIR,
            "class_distribution.png"
        )
    )

    # --------------------------------------------------------
    # Article length
    # --------------------------------------------------------

    plot_length_distribution(
        df,
        text_col,
        "Distribution of Article Length",
        os.path.join(
            RESULTS_DIR,
            "article_length_distribution.png"
        )
    )

    # --------------------------------------------------------
    # Title length
    # --------------------------------------------------------

    if title_col in df.columns:

        plot_length_distribution(
            df,
            title_col,
            "Distribution of Title Length",
            os.path.join(
                RESULTS_DIR,
                "title_length_distribution.png"
            )
        )

    # --------------------------------------------------------
    # Top words
    # --------------------------------------------------------

    labels = df[
        label_col
    ].dropna().unique().tolist()

    if len(labels) == 2:

        for label in labels:

            safe_label = str(label)

            plot_top_words(
                df,
                text_col,
                label_col,
                label,
                f"Most Frequent Words: {safe_label}",
                os.path.join(
                    RESULTS_DIR,
                    f"top_words_{safe_label}.png"
                )
            )

    else:

        logger.warning(
            f"Expected 2 classes, found {len(labels)}: {labels}"
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    logger.info(
        "EDA complete. Figures and eda_summary.json "
        "saved to results/."
    )

    return report


# ============================================================
# 8. COMMAND LINE SUPPORT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run EDA on fake news dataset."
    )

    parser.add_argument(
        "--file",
        default="news.csv"
    )

    parser.add_argument(
        "--text-col",
        default="text"
    )

    parser.add_argument(
        "--title-col",
        default="title"
    )

    parser.add_argument(
        "--label-col",
        default="label"
    )

    args = parser.parse_args()

    run_eda(
        csv_name=args.file,
        text_col=args.text_col,
        title_col=args.title_col,
        label_col=args.label_col
    )