
"""
predict_bert.py
----------------
Loads the fine-tuned BERT/DistilBERT artifacts produced by
notebooks/Fake_News_BERT.ipynb and exposes:

    predict_news_bert(text) -> {"label", "confidence", "is_calibrated", "model_used"}

Mirrors predict.py / predict_bilstm.py's interface.

Requires: torch, transformers, huggingface_hub
"""

import os
import sys
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import MODELS_DIR, load_json, get_logger

logger = get_logger("predict_bert")

# ============================================================
# HUGGING FACE REPOSITORY
# ============================================================
HF_REPO_ID = "tanmayagarwal2005/fake-news-distilbert"

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

        # Lazy imports
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and trained model from Hugging Face
        _TOKENIZER = AutoTokenizer.from_pretrained(HF_REPO_ID)

        _MODEL = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO_ID
        )

        _MODEL.to(_DEVICE)
        _MODEL.eval()

        # Download and load metadata from Hugging Face
        meta_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="metadata.json",
            repo_type="model",
        )

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
)

# Remove token_type_ids before calling DistilBERT.
inputs.pop("token_type_ids", None)

# Keep only inputs supported by the model's forward method.
import inspect

accepted_args = inspect.signature(model.forward).parameters

inputs = {
    key: value
    for key, value in inputs.items()
    if key in accepted_args
}

# Move inputs to the selected device.
inputs = {
    key: value.to(_DEVICE)
    for key, value in inputs.items()
}



    # ============================================================
    # MODEL PREDICTION
    # ============================================================
    
    # MODEL PREDICTION
    # Explicitly pass only the inputs DistilBERT needs.
    # Never forward the tokenizer's complete output dictionary.

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        ).logits

        probs = torch.softmax(
            logits,
            dim=-1,
        )[0].cpu().numpy()


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
