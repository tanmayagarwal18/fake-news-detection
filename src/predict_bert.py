
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
# DEBUG: Confirm that this exact version is running.
# ---------------------------------------------------------
print("DEBUG: UPDATED predict_bert.py IS RUNNING — VERSION 2")

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

    if (
        _MODEL is not None
        and _TOKENIZER is not None
        and _METADATA is not None
    ):
        return _MODEL, _TOKENIZER, _METADATA

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    from huggingface_hub import hf_hub_download

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Loading DistilBERT artifacts from %s",
        HF_REPO_ID,
    )

    # -----------------------------------------------------
    # Load metadata from the Hugging Face repository.
    # Files are stored at the repository root.
    # -----------------------------------------------------
    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="metadata.json",
        repo_type="model",
    )

    metadata = load_json(metadata_path)

    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata.json must contain a JSON object."
        )

    # -----------------------------------------------------
    # Load tokenizer and model.
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Validate required metadata.
    # -----------------------------------------------------
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

    if not isinstance(metadata["classes"], list) or not metadata["classes"]:
        raise ValueError(
            "metadata['classes'] must be a non-empty list."
        )

    if metadata["pos_label"] not in metadata["classes"]:
        raise ValueError(
            "metadata['pos_label'] must be present in metadata['classes']."
        )

    _TOKENIZER = tokenizer
    _MODEL = model
    _METADATA = metadata

    logger.info(
        "Loaded %s on %s (max_len=%s)",
        _METADATA["model_type"],
        _DEVICE,
        _METADATA["max_len"],
    )

    return _MODEL, _TOKENIZER, _METADATA


def predict_news_bert(text: str) -> dict:
    """
    Predict REAL/FAKE for a single news article.

    Confidence is the softmax probability of the predicted class,
    expressed as a percentage. It is not necessarily calibrated.
    """

    import torch

    model, tokenizer, metadata = _load_artifacts()

    cleaned = minimal_clean(text)

    # -----------------------------------------------------
    # Handle empty input.
    # -----------------------------------------------------
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
        max_length=int(metadata["max_len"]),
        return_tensors="pt",
    )

    # Explicitly remove token_type_ids before the model call.
    inputs.pop("token_type_ids", None)

    # -----------------------------------------------------
    # DEBUG: Inspect the actual inputs and model signature.
    # -----------------------------------------------------
    forward_parameters = inspect.signature(
        model.forward
    ).parameters

    print("DEBUG: Model call inputs:", list(inputs.keys()))
    print(
        "DEBUG: Model forward parameters:",
        list(forward_parameters.keys()),
    )

    # Only pass arguments explicitly accepted by forward().
    # This provides additional protection against unexpected
    # tokenizer outputs being passed into the model.
    accepted_args = set(forward_parameters.keys())

    inputs = {
        key: value
        for key, value in inputs.items()
        if key in accepted_args
    }

    # Ensure required input IDs are available.
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
    # DEBUG: Confirm the final arguments before inference.
    # -----------------------------------------------------
    print(
        "DEBUG: Final model arguments:",
        list(inputs.keys()),
    )

    if "token_type_ids" in inputs:
        raise RuntimeError(
            "token_type_ids is still present before the model call."
        )

    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0].cpu().tolist()

    pred_class = int(max(range(len(probs)), key=lambda i: probs[i]))
    confidence = float(probs[pred_class]) * 100.0

    # -----------------------------------------------------
    # Label mapping from metadata
    # -----------------------------------------------------
    pos_label = metadata["pos_label"]
    classes = metadata["classes"]

    if pred_class >= len(classes):
        raise ValueError(
            f"Predicted class index {pred_class} exceeds "
            f"the metadata classes list: {classes}"
        )

    neg_label = next(
        (c for c in classes if c != pos_label),
        classes[0],
    )

    # Preserve the original binary-class mapping.
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

    try:
        result = predict_news_bert(input_text)

        print(f"Prediction: {result['label']}")
        print(
            "Confidence (model probability estimate, "
            f"not guaranteed calibrated): {result['confidence']}%"
        )
        print(f"Model: {result['model_used']}")

    except Exception as exc:
        logger.exception("Command-line prediction failed")
        print(f"ERROR: {exc}")
        sys.exit(1)
