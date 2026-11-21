"""Choose a confidence threshold for each model on the VALIDATION set (not the test set).

If the model's best probability is below the threshold, the message is treated as
out-of-scope and goes to a human agent. For each model we try thresholds 0.00-0.98
and keep the one with the best average of in-scope accuracy and out-of-scope recall.
It saves models/thresholds.json.

How to run:  python -m src.thresholds
"""
import json

import numpy as np

from .config import MODELS_DIR, OOS
from .data import load_labels, load_split
from .models import MODEL_NAMES, load_model
from .preprocess import SpellCorrector


def score(probs, gold, labels, t):
    best = probs.argmax(1)
    conf = probs.max(1)
    pred = np.where(conf >= t, np.array(labels)[best], OOS)
    ins = gold != OOS
    acc_in = (pred[ins] == gold[ins]).mean()
    rec_oos = (pred[~ins] == OOS).mean()
    return (acc_in + rec_oos) / 2, acc_in, rec_oos


def main():
    val = load_split("val")
    labels = load_labels()
    speller = SpellCorrector.load(MODELS_DIR / "speller.json")
    texts = [speller.correct(t)[0] for t in val.text]
    gold = val.intent.values
    out = {}
    for key in MODEL_NAMES:
        model = load_model(key)
        probs = np.stack([model.predict_proba(t) for t in texts])
        best = max((score(probs, gold, labels, t) + (t,) for t in np.arange(0, 0.99, 0.02)),
                   key=lambda x: x[0])
        out[key] = round(float(best[3]), 2)
        print(f"{MODEL_NAMES[key]:42} threshold {out[key]:.2f} | val in-scope acc {best[1]:.4f} "
              f"| val oos recall {best[2]:.4f}")
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
