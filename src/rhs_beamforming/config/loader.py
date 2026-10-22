from pathlib import Path
from copy import deepcopy
import yaml


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base, override):
    result = deepcopy(base)

    def merge(a, b):
        for key, value in b.items():
            if isinstance(value, dict) and isinstance(a.get(key), dict):
                merge(a[key], value)
            else:
                a[key] = deepcopy(value)

    merge(result, override)
    return result
