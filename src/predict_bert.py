
"""
predict_bert.py
----------------
Loads fine-tuned BERT/DistilBERT artifacts from Hugging Face
and predicts whether a news article is FAKE or REAL.

Required Hugging Face repository files:
    config.json
    model.safetensors
    metadata.json
    tokenizer.json
    tokenizer_config.json

Repository:
    tanmayagarwal2005/fake-news-distilbert

Usage:
    python src/predict_bert.py "Some headline or article text"

Requires:
    torch, transformers, huggingface_hub
"""

import os
import sys
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_json, get_logger

logger = get_logger("predict_bert")

# ---------------------------------------------------------
# Hugging Face repository
# ---------------------------------------------------------
HF_REPO_ID = "tanmayagarwal2005/fake-news-distilbert"

# ---------------------------------------------------------
# Text cleaning patterns
# ---------------------------------------------------------
URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<.*?>", re.DOTALL)

# ---------------------------------------------------------
# Cached model objects
# ---------------------------------------------------------
_MODEL = None
_TOKENIZER = None
_METADATA = None
_DEVICE = None


def minimal_clean(text: str) -> str:
    """Apply the same minimal cleaning used during training."""

    if not isinstance(text, str) or not text.strip():
        return ""

    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)

    return text.strip()


def _load_artifacts():
    """Download and load the model, tokenizer, and metadata once."""

    global _MODEL, _TOKENIZER, _METADATA, _DEVICE

    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _METADATA

    # Lazy imports
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    from huggingface_hub import hf_hub_download

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Downloading DistilBERT artifacts from {HF_REPO_ID}"
    )

    # Download metadata from the repository root
    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="metadata.json",
    )

    # Read metadata
    metadata = load_json(metadata_path)

    # Load model and tokenizer directly from Hugging Face.
    # The repository contains config.json, model.safetensors,
    # tokenizer.json, and tokenizer_config.json at its root.
    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID,
        trust_remote_code=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO_ID,
        trust_remote_code=False,
    )

    model.to(_DEVICE)
    model.eval()

    # Validate metadata required for prediction
    required_keys = {"max_len", "pos_label", "classes", "model_type"}
    missing_keys = required_keys - set(metadata.keys())

    if missing_keys:
        raise ValueError(
            f"DistilBERT metadata is missing keys: {missing_keys}"
        )

    _MODEL = model
    _TOKENIZER = tokenizer
    _METADATA = metadata

    logger.info(
        f"Loaded {_METADATA['model_type']} on {_DEVICE} "
        f"(max_len={_METADATA['max_len']})"
    )

    return _MODEL, _TOKENIZER, _METADATA


def predict_news_bert(text: str) -> dict:
    """
    Predict REAL/FAKE for one news article.

    Confidence is the softmax probability of the predicted class.
    It is not necessarily a calibrated probability.
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
            "note": (
                "Input text was empty after cleaning; "
                "cannot classify."
            ),
        }

    # -----------------------------------------------------
    # Tokenization
    # -----------------------------------------------------
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=metadata["max_len"],
        return_tensors="pt",
    )

    # IMPORTANT:
    # DistilBERT's sequence-classification forward method
    # does not accept token_type_ids.
    inputs.pop("token_type_ids", None)

    # Move tensors to the selected device
    inputs = {
        key: value.to(_DEVICE)
        for key, value in inputs.items()
    }

    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class]) * 100

    # -----------------------------------------------------
    # Label mapping from training metadata
    # -----------------------------------------------------
    pos_label = metadata["pos_label"]
    classes = metadata["classes"]

    neg_label = next(
        (c for c in classes if c != pos_label),
        classes[0],
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


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python src/predict_bert.py "news text here"'
        )
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    result = predict_news_bert(input_text)

    print(f"Prediction: {result['label']}")
    print(
        "Confidence (model probability estimate, "
        f"not guaranteed calibrated): {result['confidence']}%"
    )
    print(f"Model: {result['model_used']}")
