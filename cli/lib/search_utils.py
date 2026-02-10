import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "data.json"


def load_movies():
    with open(DATA_DIR, "r") as f:
        data = json.load(f)
    return data["movies"]
