"""Download the CLINC150 dataset and save the train / validation / test files.

CLINC150 (Larson et al., 2019): 150 intents in 10 domains + out-of-scope (oos) messages.
We use the official splits:
- train: 15,000 in-scope + 100 oos messages
- val:    3,000 in-scope + 100 oos messages
- test:   4,500 in-scope + 1,000 oos messages  -> the SAME test set for all models (R3)

It saves data/train.csv, data/val.csv, data/test.csv (columns: text, intent)
and data/domains.json (which intent belongs to which domain / team).

How to run:  python -m src.prepare_data
"""
import json
import urllib.request

import pandas as pd

from .config import DATA_DIR, DATA_URL, DOMAINS_URL


def download(url, path):
    if not path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    data = download(DATA_URL, DATA_DIR / "data_full.json")
    download(DOMAINS_URL, DATA_DIR / "domains.json")
    for split in ["train", "val", "test"]:
        rows = data[split] + data[f"oos_{split}"]
        df = pd.DataFrame(rows, columns=["text", "intent"])
        df.to_csv(DATA_DIR / f"{split}.csv", index=False)
        n_oos = (df.intent == "oos").sum()
        print(f"{split:5}: {len(df):6} messages | in-scope {len(df) - n_oos} | out-of-scope {n_oos} "
              f"| intents {df.intent.nunique()}")


if __name__ == "__main__":
    main()
