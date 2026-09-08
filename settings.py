# settings.py

import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "settings.json"

SYSTEM = "You are SwAIm, an AI designed to solve the problem of forgetting what land looks like."
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
RUNS = "runs"
DATA_PATH = "data.jsonl"


def load(path=PATH):
    with open(path) as file:
        return json.load(file)


for key, value in load().items():
    globals()[key] = value
