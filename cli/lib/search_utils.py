import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOVIES_FILE = DATA_DIR / "data.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"


def load_movies():
    with open(MOVIES_FILE, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_FILE, "r") as f:
        stopwords = [line.strip() for line in f.readlines()]
    return stopwords
