"""Small helpers to load the data and the list of labels."""
import json

import pandas as pd

from .config import DATA_DIR, MODELS_DIR, OOS


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def make_labels(train: pd.DataFrame) -> list[str]:
    """Sorted list of the 151 labels (150 intents + oos). Saved so all models use the same order."""
    labels = sorted(train.intent.unique())
    (MODELS_DIR / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    domains = json.loads((DATA_DIR / "domains.json").read_text(encoding="utf-8"))
    intent_to_domain = {i: d for d, intents in domains.items() for i in intents}
    intent_to_domain[OOS] = OOS
    (MODELS_DIR / "intent_domain.json").write_text(json.dumps(intent_to_domain), encoding="utf-8")
    return labels


def load_labels() -> list[str]:
    return json.loads((MODELS_DIR / "labels.json").read_text(encoding="utf-8"))


def load_intent_domain() -> dict:
    return json.loads((MODELS_DIR / "intent_domain.json").read_text(encoding="utf-8"))
