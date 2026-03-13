"""
Shared constants and data-loading utilities used across the search modules.

Path constants are derived from the location of this file so the project can
be run from any working directory without adjustment.
"""

import json
from pathlib import Path

# BM25 hyperparameters
# k1 controls term-frequency saturation: higher values reduce saturation so
# that each additional occurrence of a term contributes more to the score.
BM25_K1 = 1.5

# b controls document-length normalisation: 1.0 = full normalisation,
# 0.0 = no normalisation.  0.75 is the widely-used default.
BM25_B = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOVIES_FILE = DATA_DIR / "data.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"

# Directory where pre-built indexes and embeddings are cached between runs.
CACHE_PATH = PROJECT_ROOT / "cache"


def load_movies() -> list[dict]:
    """
    Load the movie corpus from the JSON data file.

    Reads ``data/data.json`` (relative to the project root) and returns the
    value of its top-level ``"movies"`` key as a list of dicts.  Each dict is
    expected to contain at least ``id``, ``title``, and ``description`` fields.

    Returns:
        List of movie dicts representing the full corpus.

    Raises:
        FileNotFoundError: If ``data/data.json`` does not exist.
        KeyError: If the JSON file does not contain a ``"movies"`` key.
    """
    with open(MOVIES_FILE, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    """
    Load the stopword list from the plain-text stopwords file.

    Reads ``data/stopwords.txt`` (relative to the project root) where each
    line contains one stopword.  Stopwords are stripped of surrounding
    whitespace before being returned.  These are used during tokenisation to
    filter out common words that carry little discriminative value (e.g. "the",
    "and", "is").

    Returns:
        List of stopword strings.

    Raises:
        FileNotFoundError: If ``data/stopwords.txt`` does not exist.
    """
    with open(STOPWORDS_FILE, "r") as f:
        stopwords = [line.strip() for line in f.readlines()]
    return stopwords
