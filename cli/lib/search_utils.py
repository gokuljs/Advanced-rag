import json
from pathlib import Path

BM25_K1 = 1.5
BM25_B = 0.75 # this is a hyperparameter that controls the bias towards longer documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOVIES_FILE = DATA_DIR / "data.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"
CACHE_PATH = PROJECT_ROOT / "cache"


def load_movies():
    with open(MOVIES_FILE, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_FILE, "r") as f:
        stopwords = [line.strip() for line in f.readlines()]
    return stopwords
