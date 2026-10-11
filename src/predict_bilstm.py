"""
predict_bilstm.py
-----------------
Loads the trained BiLSTM artifacts and predicts whether a news article
is FAKE or REAL.

Required artifacts:

    models/bilstm_artifacts/
        ├── bilstm_model.keras
        ├── tokenizer.pkl
        └── metadata.json

Usage:

    python src/predict_bilstm.py "Some headline or article text"
"""

import os
import sys
import pickle

# ---------------------------------------------------------
# Allow imports from src/
# ---------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import MODELS_DIR, load_json, get_logger
from text_light_clean import light_clean


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------
logger = get_logger("predict_bilstm")


# ---------------------------------------------------------
# IMPORTANT:
# Your actual folder is models/bilstm_artifacts/
# ---------------------------------------------------------
BILSTM_DIR = os.path.join(
    MODELS_DIR,
    "bilstm_artifacts"
)


# ---------------------------------------------------------
# Cached model objects
# ---------------------------------------------------------
_MODEL = None
_TOKENIZER = None
_METADATA = None


# ---------------------------------------------------------
# Load BiLSTM artifacts
# ---------------------------------------------------------
def _load_artifacts():

    global _MODEL
    global _TOKENIZER
    global _METADATA

    # Load only once
    if _MODEL is None:

        model_path = os.path.join(
            BILSTM_DIR,
            "bilstm_model.keras"
        )

        tokenizer_path = os.path.join(
            BILSTM_DIR,
            "tokenizer.pkl"
        )

        metadata_path = os.path.join(
            BILSTM_DIR,
            "metadata.json"
        )

        # -------------------------------------------------
        # Check that all required files exist
        # -------------------------------------------------
        for path in [
            model_path,
            tokenizer_path,
            metadata_path
        ]:

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"\nMissing BiLSTM artifact:\n{path}\n\n"
                    f"Expected folder:\n{BILSTM_DIR}\n\n"
                    f"Make sure these files exist:\n"
                    f"  bilstm_model.keras\n"
                    f"  tokenizer.pkl\n"
                    f"  metadata.json"
                )

        # -------------------------------------------------
        # Import TensorFlow only when needed
        # -------------------------------------------------
        import tensorflow as tf

        # -------------------------------------------------
        # Load trained model
        # -------------------------------------------------
        _MODEL = tf.keras.models.load_model(
            model_path
        )

        # -------------------------------------------------
        # Load tokenizer
        # -------------------------------------------------
        with open(
            tokenizer_path,
            "rb"
        ) as f:

            _TOKENIZER = pickle.load(f)

        # -------------------------------------------------
        # Load metadata
        # -------------------------------------------------
        _METADATA = load_json(
            metadata_path
        )

        logger.info(
            f"Loaded BiLSTM model "
            f"(max_len={_METADATA['max_len']})"
        )

    return (
        _MODEL,
        _TOKENIZER,
        _METADATA
    )


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_news_bilstm(text: str) -> dict:

    """
    Predict FAKE/REAL for a single news article.

    Returns:

    {
        "label": "FAKE" or "REAL",
        "confidence": percentage,
        "is_calibrated": False,
        "model_used": "BiLSTM"
    }
    """

    # -----------------------------------------------------
    # Load TensorFlow padding utility
    # -----------------------------------------------------
    from tensorflow.keras.preprocessing.sequence import (
        pad_sequences
    )

    # -----------------------------------------------------
    # Load model, tokenizer and metadata
    # -----------------------------------------------------
    model, tokenizer, metadata = _load_artifacts()

    # -----------------------------------------------------
    # Clean input text
    # -----------------------------------------------------
    cleaned = light_clean(text)

    # -----------------------------------------------------
    # Handle empty input
    # -----------------------------------------------------
    if cleaned == "":

        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "is_calibrated": False,
            "model_used": "BiLSTM",
            "note": (
                "Input text was empty after cleaning; "
                "cannot classify."
            )
        }

    # -----------------------------------------------------
    # Convert text to integer sequence
    # -----------------------------------------------------
    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )

    # -----------------------------------------------------
    # Pad sequence to MAX_LEN = 512
    # -----------------------------------------------------
    padded = pad_sequences(
        sequence,
        maxlen=metadata["max_len"],
        padding="post",
        truncating="post"
    )

    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------
    prediction = model.predict(
        padded,
        verbose=0
    )

    # Convert prediction to Python float
    probability_positive = float(
        prediction[0][0]
    )

    # -----------------------------------------------------
    # Determine positive and negative labels
    # -----------------------------------------------------
    positive_label = metadata["pos_label"]

    classes = metadata["classes"]

    negative_label = next(
        (
            c for c in classes
            if c != positive_label
        ),
        classes[0]
    )

    # -----------------------------------------------------
    # Classification threshold = 0.5
    # -----------------------------------------------------
    if probability_positive >= 0.5:

        predicted_label = str(
            positive_label
        ).upper()

        confidence = probability_positive

    else:

        predicted_label = str(
            negative_label
        ).upper()

        confidence = 1 - probability_positive

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------
    return {

        "label": predicted_label,

        "confidence": round(
            confidence * 100,
            4
        ),

        # The model output is NOT calibrated
        "is_calibrated": False,

        "model_used": "BiLSTM"
    }


# ---------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            'Usage: python src/predict_bilstm.py '
            '"news text here"'
        )

        sys.exit(1)

    # Combine all command-line arguments
    input_text = " ".join(
        sys.argv[1:]
    )

    # Make prediction
    result = predict_news_bilstm(
        input_text
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------
    print("\n==============================")
    print("BiLSTM Fake News Prediction")
    print("==============================")

    print(
        f"Prediction: {result['label']}"
    )

    print(
        "Confidence "
        "(model probability estimate, "
        f"not guaranteed calibrated): "
        f"{result['confidence']}%"
    )

    print(
        f"Model: {result['model_used']}"
    )

    print("==============================\n")