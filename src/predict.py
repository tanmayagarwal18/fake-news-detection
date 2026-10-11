"""
predict.py
----------
Loads the saved TF-IDF vectorizer + best model (produced by train.py)
and exposes a single reusable function:

    predict_news(text) -> {"label": "FAKE"/"REAL", "confidence": float, "is_calibrated": bool}

Run directly for a quick command-line check:
    python src/predict.py "Some headline or article text here"
"""

import os
import sys

import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import MODELS_DIR, load_json, get_logger
from preprocessing import clean_text

logger = get_logger("predict")

_MODEL = None
_VECTORIZER = None
_METADATA = None


def _load_artifacts():
    """Lazy-load model/vectorizer once and cache them (used by the Streamlit app too)."""
    global _MODEL, _VECTORIZER, _METADATA
    if _MODEL is None:
        model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

        for p in [model_path, vec_path, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing {p}. Run `python src/train.py` first to train and save "
                    f"the model, vectorizer, and metadata."
                )

        _MODEL = joblib.load(model_path)
        _VECTORIZER = joblib.load(vec_path)
        _METADATA = load_json(meta_path)
        logger.info(f"Loaded model: {_METADATA['best_model']}")
    return _MODEL, _VECTORIZER, _METADATA


def predict_news(text: str) -> dict:
    """
    Predict REAL/FAKE for a single piece of news text (headline, article,
    or both concatenated).

    Returns a dict:
        label            : "FAKE" or "REAL" (mapped from the training label)
        confidence       : float in [0, 100]
        is_calibrated     : True if `confidence` is a genuine probability
                            (predict_proba), False if it's a decision-score
                            based approximation (e.g. for LinearSVC, which
                            has no predict_proba by default).
        model_used        : human-readable model name
    """
    model, vectorizer, metadata = _load_artifacts()

    cleaned = clean_text(text)
    if cleaned == "":
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "is_calibrated": False,
            "model_used": metadata["best_model"],
            "note": "Input text was empty after cleaning; cannot classify.",
        }

    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    is_calibrated = hasattr(model, "predict_proba")
    if is_calibrated:
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        confidence = float(np.max(proba)) * 100
    else:
        # LinearSVC etc: use decision_function as a relative confidence
        # score, NOT a calibrated probability. We squash it with a
        # sigmoid purely for a 0-100 display range -- this is explicitly
        # NOT a true probability and the UI must say so.
        score = model.decision_function(X)
        score = score[0] if hasattr(score, "__len__") else score
        confidence = float(1 / (1 + np.exp(-abs(score)))) * 100

    label_str = str(pred).upper()
    # Normalize common label encodings to FAKE/REAL for display
    if label_str in ("1", "FAKE", "TRUE_FAKE"):
        display_label = "FAKE"
    elif label_str in ("0", "REAL", "TRUE_REAL"):
        display_label = "REAL"
    else:
        display_label = label_str  # dataset uses its own string labels already

    return {
        "label": display_label,
        "confidence": round(confidence, 2),
        "is_calibrated": is_calibrated,
        "model_used": metadata["best_model"],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "news text here"')
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    result = predict_news(input_text)
    print(f"Prediction: {result['label']}")
    conf_label = "Confidence" if result["is_calibrated"] else "Confidence (decision-score based, not a calibrated probability)"
    print(f"{conf_label}: {result['confidence']}%")
    print(f"Model: {result['model_used']}")
