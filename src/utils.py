import json
from pathlib import Path

def load_labels(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "labels.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["labels"], data["map_to_id"], {int(k): v for k, v in data["id_to_label"].items()}
