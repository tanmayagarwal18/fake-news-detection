
"""
predict_bert.py
----------------
Loads fine-tuned DistilBERT artifacts from Hugging Face
and predicts whether news is FAKE or REAL.
"""

import os
import sys
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_json, get_logger

logger = get_logger("predict_bert")

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
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        try:
            # The repository contains the files at its root.
            # from_pretrained accepts a Hugging Face repo ID directly.
            _TOKENIZER = AutoTokenizer.from_pretrained(
                HF_REPO_ID
            )

            _MODEL = AutoModelForSequenceClassification.from_pretrained(
                HF_REPO_ID,
                use_safetensors=True,
            )

            from huggingface_hub import hf_hub_download

            metadata_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="metadata.json",
            )

            _METADATA = load_json(metadata_path)

        except Exception as exc:
            # Keep the original exception available in Streamlit logs.
            logger.exception("Failed to load DistilBERT artifacts")
            raise RuntimeError(
                "Could not load DistilBERT artifacts from "
                f"Hugging Face repository '{HF_REPO_ID}'. "
                "Check repository visibility, file names, "
                "network access, and dependencies."
            ) from exc

        required_keys = {"model_type", "max_len", "pos_label", "classes"}
        missing_keys = required_keys - set(_METADATA.keys())

        if missing_keys:
            raise ValueError(
                f"DistilBERT metadata is missing keys: {missing_keys}"
            )

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        _MODEL.to(_DEVICE)
        _MODEL.eval()

        logger.info(
            f"Loaded {_METADATA['model_type']} on {_DEVICE} "
            f"(max_len={_METADATA['max_len']})"
        )

    return _MODEL, _TOKENIZER, _METADATA


def predict_news_bert(text: str) -> dict:
    """
    Predict REAL/FAKE using the fine-tuned DistilBERT model.

    Confidence is the softmax probability of the predicted class;
    it is not guaranteed to be calibrated.
    """

    cleaned = minimal_clean(text)

    if cleaned == "":
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "is_calibrated": False,
            "model_used": "DistilBERT",
            "note": "Input text was empty after cleaning; cannot classify.",
        }

    import torch

    model, tokenizer, metadata = _load_artifacts()

    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=metadata["max_len"],
        return_tensors="pt",
    ).to(_DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class]) * 100

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict_bert.py "news text here"')
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    result = predict_news_bert(input_text)

    print(f"Prediction: {result['label']}")
    print(
        "Confidence (model probability estimate, "
        f"not guaranteed calibrated): {result['confidence']}%"
    )
    print(f"Model: {result['model_used']}")
