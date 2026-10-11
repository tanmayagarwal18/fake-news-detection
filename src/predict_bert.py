"""
predict_bert.py
----------------
Loads the fine-tuned BERT/DistilBERT artifacts produced by
notebooks/Fake_News_BERT.ipynb and exposes:

    predict_news_bert(text) -> {"label", "confidence", "is_calibrated", "model_used"}

Mirrors predict.py / predict_bilstm.py's interface.

Requires: torch, transformers
"""

import os
import sys
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import MODELS_DIR, load_json, get_logger

logger = get_logger("predict_bert")

# ============================================================
# BERT ARTIFACT DIRECTORY
# Your actual structure is:
# models/
# └── bert_artifacts/
#     ├── model/
#     ├── tokenizer/
#     └── metadata.json
# ============================================================
BERT_DIR = os.path.join(MODELS_DIR, "bert_artifacts")

URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<.*?>", re.DOTALL)

_MODEL = None
_TOKENIZER = None
_METADATA = None
_DEVICE = None


def minimal_clean(text: str) -> str:
    """Same minimal cleaning used in the BERT training notebook."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)

    return text.strip()


def _load_artifacts():
    global _MODEL, _TOKENIZER, _METADATA, _DEVICE

    if _MODEL is None:

        model_dir = os.path.join(BERT_DIR, "model")
        tok_dir = os.path.join(BERT_DIR, "tokenizer")
        meta_path = os.path.join(BERT_DIR, "metadata.json")

        # Check that all BERT artifacts exist
        for p in [model_dir, tok_dir, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing {p}. Fine-tune the model and place the "
                    f"BERT artifacts inside models/bert_artifacts/."
                )

        # Lazy imports
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and trained model
        _TOKENIZER = AutoTokenizer.from_pretrained(tok_dir)

        _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)

        _MODEL.to(_DEVICE)
        _MODEL.eval()

        # Load metadata
        _METADATA = load_json(meta_path)

        logger.info(
            f"Loaded {_METADATA['model_type']} on {_DEVICE} "
            f"(max_len={_METADATA['max_len']})"
        )

    return _MODEL, _TOKENIZER, _METADATA


def predict_news_bert(text: str) -> dict:
    """
    Predict REAL/FAKE for a single piece of news text using
    the fine-tuned BERT/DistilBERT model.

    confidence is the softmax probability of the predicted class.

    is_calibrated is False because softmax probability is not
    automatically a calibrated probability.
    """

    import torch

    model, tokenizer, metadata = _load_artifacts()

    cleaned = minimal_clean(text)

    # Handle empty input
    if cleaned == "":
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "is_calibrated": False,
            "model_used": metadata["model_type"],
            "note": "Input text was empty after cleaning; cannot classify.",
        }

    # ============================================================
    # TOKENIZATION
    # Uses exactly the same max length as training
    # ============================================================
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=metadata["max_len"],
        return_tensors="pt",
    ).to(_DEVICE)

    # ============================================================
    # MODEL PREDICTION
    # ============================================================
    with torch.no_grad():

        logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_class = int(probs.argmax())

    confidence = float(probs[pred_class]) * 100

    # ============================================================
    # LABEL MAPPING
    #
    # During training:
    #
    # POS_LABEL = "FAKE"
    #
    # Therefore:
    # class 0 -> REAL
    # class 1 -> FAKE
    #
    # This is determined from metadata rather than hard-coding
    # the labels.
    # ============================================================
    pos_label = metadata["pos_label"]
    classes = metadata["classes"]

    neg_label = next(
        (c for c in classes if c != pos_label),
        classes[0]
    )

    if pred_class == 1:
        display_label = str(pos_label).upper()
    else:
        display_label = str(neg_label).upper()

    return {
        "label": display_label,
        "confidence": round(confidence, 2),
        "is_calibrated": False,
        "model_used": metadata["model_type"],
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python src/predict_bert.py "news text here"')
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])

    result = predict_news_bert(input_text)

    print(f"Prediction: {result['label']}")
    print(
        f"Confidence "
        f"(model probability estimate, not guaranteed calibrated): "
        f"{result['confidence']}%"
    )
    print(f"Model: {result['model_used']}")