# data.py

import json

from settings import DATA_PATH


def get_dataset():
    with open(DATA_PATH) as file:
        return [json.loads(line)["messages"] for line in file]
