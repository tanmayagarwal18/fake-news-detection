"""
utils.py
--------
Small shared helper functions used across preprocessing, training,
evaluation and prediction scripts. Keeping these in one place avoids
duplicating path/logging logic in every stage.
"""

import os
import json
import logging

# ---------------------------------------------------------------------
# Project paths (edit ROOT_DIR only if you move the project folder)
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def ensure_dirs():
    """Create all expected project folders if they don't already exist."""
    for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent, readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj: dict, path: str):
    """Save a dict as pretty-printed JSON (used for metrics/comparison tables)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
