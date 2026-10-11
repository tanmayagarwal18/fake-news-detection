
"""
predict_bilstm.py
-----------------
Loads BiLSTM artifacts from Hugging Face and predicts whether
a news article is FAKE or REAL.
"""

import os
import sys
import pickle
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_json, get_logger
from text_light_clean import light_clean

logger = get_logger("predict_bilstm")

# Public Hugging Face model repository
HF_REPO_ID = "tanmayagarwal2005/fake-news-bilstm"

_MODEL = None
_TOKENIZER = None
_METADATA = None


def _load_artifacts():
    """Download and load the model artifacts once."""

    global _MODEL, _TOKENIZER, _METADATA

    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _METADATA

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is missing. Add "
            "'huggingface_hub' to requirements.txt."
        ) from exc

    # Download files into the Hugging Face cache.
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="bilstm_model.keras",
    )

    tokenizer_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="tokenizer.pkl",
    )

    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="metadata.json",
    )

    # Import TensorFlow only when the model is needed.
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    metadata = load_json(metadata_path)

    # Validate the metadata needed by prediction.
    required_keys = {"max_len", "pos_label", "classes"}
    missing_keys = required_keys - set(metadata.keys())

    if missing_keys:
        raise ValueError(
            f"BiLSTM metadata is missing keys: {missing_keys}"
        )

    _MODEL = model
    _TOKENIZER = tokenizer
    _METADATA = metadata

    logger.info(
        "Loaded BiLSTM model "
        f"(max_len={metadata['max_len']})"
    )

    return _MODEL, _TOKENIZER, _METADATA


def predict_news_bilstm(text: str) -> dict:
    """Predict FAKE/REAL for one article."""

    cleaned = light_clean(text)

    if cleaned == "":
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "is_calibrated": False,
            "model_used": "BiLSTM",
            "note": (
                "Input text was empty after cleaning; "
                "cannot classify."
            ),
        }

    model, tokenizer, metadata = _load_artifacts()

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=metadata["max_len"],
        padding="post",
        truncating="post",
    )

    prediction = model.predict(padded, verbose=0)
    probability_positive = float(prediction[0][0])

    positive_label = metadata["pos_label"]
    classes = metadata["classes"]

    negative_label = next(
        (c for c in classes if c != positive_label),
        classes[0],
    )

    if probability_positive >= 0.5:
        predicted_label = str(positive_label).upper()
        confidence = probability_positive
    else:
        predicted_label = str(negative_label).upper()
        confidence = 1.0 - probability_positive

    return {
        "label": predicted_label,
        "confidence": round(confidence * 100, 4),
        "is_calibrated": False,
        "model_used": "BiLSTM",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python src/predict_bilstm.py "news text here"'
        )
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    result = predict_news_bilstm(input_text)

    print("\n==============================")
    print("BiLSTM Fake News Prediction")
    print("==============================")
    print(f"Prediction: {result['label']}")
    print(
        "Confidence (not guaranteed calibrated): "
        f"{result['confidence']}%"
    )
    print(f"Model: {result['model_used']}")
    print("==============================\n")
