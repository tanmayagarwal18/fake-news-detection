
"""
predict_bert.py
----------------
Loads fine-tuned BERT/DistilBERT artifacts from Hugging Face
and predicts whether a news article is FAKE or REAL.

Hugging Face repository:
    tanmayagarwal2005/fake-news-distilbert

Required repository files:
    config.json
    model.safetensors
    metadata.json
    tokenizer.json
    tokenizer_config.json

Requires:
    torch, transformers, huggingface_hub
"""

import os
import sys
import re
import inspect

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
    """Apply the minimal cleaning used during training."""

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

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Loading DistilBERT artifacts from {HF_REPO_ID}"
    )

    # Load metadata from the Hugging Face repository.
    # This repository stores its files at the root.
    from huggingface_hub import hf_hub_download

    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="metadata.json",
    )

    metadata = load_json(metadata_path)

    # Load tokenizer and model from the repository.
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

    # Validate the metadata required for prediction.
    required_keys = {
        "max_len",
        "pos_label",
        "classes",
        "model_type",
    }
    missing_keys = required_keys - set(metadata.keys())

    if missing_keys:
        raise ValueError(
            f"DistilBERT metadata is missing keys: {missing_keys}"
        )

    _TOKENIZER = tokenizer
    _MODEL = model
    _METADATA = metadata

    logger.info(
        f"Loaded {_METADATA['model_type']} on {_DEVICE} "
        f"(max_len={_METADATA['max_len']})"
    )

    return _MODEL, _TOKENIZER, _METADATA


def predict_news_bert(text: str) -> dict:
    """
    Predict REAL/FAKE for a single news article.

    Confidence is the softmax probability of the predicted class.
    It is not necessarily a calibrated probability.
    """

    import torch

    model, tokenizer, metadata = _load_artifacts()

    cleaned = minimal_clean(text)

    # Handle empty input.
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

    # Explicitly remove token_type_ids.
    # DistilBERT does not accept this argument.
    inputs.pop("token_type_ids", None)

    # Defensive filtering: pass only arguments accepted by
    # this model's forward() method.
    accepted_args = inspect.signature(
        model.forward
    ).parameters

    inputs = {
        key: value
        for key, value in inputs.items()
        if key in accepted_args
    }

    # Ensure input_ids are available.
    if "input_ids" not in inputs:
        raise RuntimeError(
            "Tokenizer did not produce input_ids. "
            "Check the tokenizer files in the Hugging Face repository."
        )

    # Move tensors to the selected device.
    inputs = {
        key: value.to(_DEVICE)
        for key, value in inputs.items()
    }

    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(
            logits,
            dim=-1,
        )[0].cpu().numpy()

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class]) * 100

    # -----------------------------------------------------
    # Label mapping from metadata
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
