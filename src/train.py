"""
train.py
--------
Complete Fake News Detection training pipeline.

Pipeline:

Dataset
   ↓
Remove missing/duplicate rows
   ↓
Combine title + article text
   ↓
spaCy preprocessing
   ↓
Train/Test Split
   ↓
TF-IDF
   ↓
Three ML Models
   ├── Logistic Regression
   ├── Multinomial Naive Bayes
   └── Linear SVM
   ↓
Evaluation
   ↓
Model Comparison
   ↓
Best Model
   ↓
Save Model + TF-IDF Vectorizer

The trained model can later be used by predict.py.
"""

import os
import sys
import argparse

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from utils import (
    DATA_RAW_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    get_logger,
    save_json
)

from preprocessing import clean_series


logger = get_logger("train")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    csv_name: str
) -> pd.DataFrame:

    path = os.path.join(
        DATA_RAW_DIR,
        csv_name
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    logger.info(
        f"Loading dataset: {path}"
    )

    df = pd.read_csv(path)

    logger.info(
        f"Dataset loaded: {len(df):,} rows"
    )

    return df


# ============================================================
# COMBINE TITLE + TEXT
# ============================================================

def combine_title_text(
    df,
    title_col,
    text_col
):

    if title_col in df.columns:

        return (
            df[title_col]
            .fillna("")
            .astype(str)
            + " "
            + df[text_col]
            .fillna("")
            .astype(str)
        )

    return (
        df[text_col]
        .fillna("")
        .astype(str)
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df,
    title_col,
    text_col,
    label_col
):

    df = df.copy()

    # Remove missing values
    df = df.dropna(
        subset=[
            text_col,
            label_col
        ]
    )

    # Remove duplicate articles
    df = df.drop_duplicates(
        subset=[text_col]
    )

    logger.info(
        f"Rows after cleaning duplicates/missing values: "
        f"{len(df):,}"
    )

    # Combine title + article text
    combined = combine_title_text(
        df,
        title_col,
        text_col
    )

    # ========================================================
    # SPACY PREPROCESSING
    # ========================================================

    logger.info(
        "Starting spaCy NLP preprocessing..."
    )

    cleaned = clean_series(
        combined
    )

    cleaned = pd.Series(
        cleaned,
        index=df.index
    )

    # Remove documents that became empty
    mask = cleaned.str.len() > 0

    cleaned = cleaned[mask]

    labels = df.loc[
        mask,
        label_col
    ]

    logger.info(
        f"Documents remaining after NLP preprocessing: "
        f"{len(cleaned):,}"
    )

    return (
        cleaned.reset_index(drop=True),
        labels.reset_index(drop=True)
    )


# ============================================================
# MODELS
# ============================================================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Multinomial Naive Bayes": MultinomialNB(),

        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=42),
            method="sigmoid",
            cv=5
        ),
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "f1_score":
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
    }

    logger.info(
        f"{name}: {metrics}"
    )

    return (
        metrics,
        predictions
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_test,
    predictions,
    labels,
    model_name,
    output_path
):

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    plt.figure(
        figsize=(5, 4)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.colorbar()

    tick_marks = range(
        len(labels)
    )

    plt.xticks(
        tick_marks,
        labels
    )

    plt.yticks(
        tick_marks,
        labels
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    # Add numbers
    for i in range(len(labels)):

        for j in range(len(labels)):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    logger.info(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main(
    csv_name,
    title_col,
    text_col,
    label_col
):

    ensure_dirs()

    # ========================================================
    # LOAD
    # ========================================================

    df = load_dataset(
        csv_name
    )

    # ========================================================
    # CHECK COLUMNS
    # ========================================================

    required = [
        text_col,
        label_col
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise KeyError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ========================================================
    # PREPARE
    # ========================================================

    X_text, y = prepare_data(
        df,
        title_col,
        text_col,
        label_col
    )

    labels = sorted(
        y.unique().tolist()
    )

    logger.info(
        f"Classes: {labels}"
    )

    logger.info(
        f"Final training dataset: "
        f"{len(X_text):,} documents"
    )

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train_text, X_test_text, y_train, y_test = (
        train_test_split(
            X_text,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    logger.info(
        f"Training samples: {len(X_train_text):,}"
    )

    logger.info(
        f"Testing samples: {len(X_test_text):,}"
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    logger.info(
        "Creating TF-IDF features..."
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        
    )

    X_train = vectorizer.fit_transform(
        X_train_text
    )

    X_test = vectorizer.transform(
        X_test_text
    )

    logger.info(
        f"TF-IDF training matrix shape: "
        f"{X_train.shape}"
    )

    logger.info(
        f"TF-IDF testing matrix shape: "
        f"{X_test.shape}"
    )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    models = get_models()

    results = {}

    trained_models = {}

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for name, model in models.items():

        logger.info(
            f"========================================"
        )

        logger.info(
            f"Training {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        metrics, predictions = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results[name] = metrics

        trained_models[name] = model

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )

        report_path = os.path.join(
            RESULTS_DIR,
            f"classification_report_{safe_name}.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                classification_report(
                    y_test,
                    predictions,
                    zero_division=0
                )
            )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        plot_confusion_matrix(
            y_test,
            predictions,
            labels,
            name,
            os.path.join(
                RESULTS_DIR,
                f"confusion_matrix_{safe_name}.png"
            )
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        results
    ).T

    comparison = comparison.round(
        4
    )

    logger.info(
        "\nMODEL COMPARISON:\n"
        + str(comparison)
    )

    comparison.to_csv(
        os.path.join(
            RESULTS_DIR,
            "model_comparison.csv"
        )
    )

    save_json(
        results,
        os.path.join(
            RESULTS_DIR,
            "model_comparison.json"
        )
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = (
        comparison[
            "f1_score"
        ]
        .idxmax()
    )

    best_model = trained_models[
        best_model_name
    ]

    logger.info(
        f"Best model based on F1-score: "
        f"{best_model_name}"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    joblib.dump(
        best_model,
        os.path.join(
            MODELS_DIR,
            "best_model.joblib"
        )
    )

    # Save TF-IDF
    joblib.dump(
        vectorizer,
        os.path.join(
            MODELS_DIR,
            "tfidf_vectorizer.joblib"
        )
    )

    # Save metadata
    metadata = {

        "best_model":
            best_model_name,

        "labels":
            [
                str(label)
                for label in labels
            ],

        "training_samples":
            int(len(X_train_text)),

        "testing_samples":
            int(len(X_test_text)),

        "tfidf_features":
            int(X_train.shape[1]),

        "preprocessing":
            "spaCy tokenization, stopword removal, lemmatization",

        "tfidf":
            "unigrams + bigrams, max_features=5000"
    }

    save_json(
        metadata,
        os.path.join(
            MODELS_DIR,
            "model_metadata.json"
        )
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "TRAINING COMPLETED SUCCESSFULLY!"
    )

    logger.info(
        f"Best model: {best_model_name}"
    )

    logger.info(
        "Model saved to models/best_model.joblib"
    )

    logger.info(
        "Vectorizer saved to models/"
        "tfidf_vectorizer.joblib"
    )

    return (
        comparison,
        best_model_name
    )


# ============================================================
# COMMAND LINE SUPPORT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Fake News Detection models."
    )

    parser.add_argument(
        "--file",
        default="news.csv"
    )

    parser.add_argument(
        "--title-col",
        default="title"
    )

    parser.add_argument(
        "--text-col",
        default="text"
    )

    parser.add_argument(
        "--label-col",
        default="label"
    )

    args = parser.parse_args()

    main(
        args.file,
        args.title_col,
        args.text_col,
        args.label_col
    )