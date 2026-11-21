"""Test ALL models on the SAME test set (R3) and make the results table (R4).

Test set: CLINC150 official test = 4,500 in-scope + 1,000 out-of-scope messages.
For each model it measures:
- in-scope accuracy: correct intent for in-scope messages
- out-of-scope recall: share of out-of-scope messages sent to a human agent
- routing accuracy: correct team (human agent for out-of-scope), all 5,500 messages
- size of the model files (MB)
- average time per message on CPU (cleaning + model, one message at a time, after 10 warm-up runs)

It saves results/results.md, results/results.csv and results/errors_student_int8.csv
How to run:  python -m src.benchmark --runs 200
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from .config import HUMAN_AGENT, OOS, RESULTS_DIR, SEED, TEAMS
from .data import load_split
from .hub import download
from .models import MODEL_NAMES, model_size_mb
from .router import Router

ORDER = ["tfidf", "ffn", "teacher_fp32", "teacher_int8", "student_fp32", "student_int8"]


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:  # get the CPU name on Linux
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":")[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        pass
    return (f"{cpu} | {os.cpu_count()} logical cores | torch threads = {torch.get_num_threads()} | "
            f"{platform.system()} {platform.release()} | CPU only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200, help="messages used to measure speed (>=100)")
    ap.add_argument("--models", nargs="*", default=ORDER)
    args = ap.parse_args()

    download()  # download the models from Hugging Face if needed
    torch.set_grad_enabled(False)
    test = load_split("test")
    gold = test.intent.values
    ins = gold != OOS
    lat_msgs = test.sample(n=min(args.runs, len(test)), random_state=SEED).text.tolist()
    router = Router(args.models)
    gold_team = [HUMAN_AGENT if g == OOS else TEAMS[router.domain[g]] for g in gold]

    rows = []
    for key in args.models:
        print(f"=== {MODEL_NAMES[key]} ===")
        res = [router.predict(t, key) for t in test.text]
        pred = np.array([r["intent"] for r in res])
        teams = [r["team"] for r in res]

        for t in lat_msgs[:10]:            # warm up
            router.predict(t, key)
        times = [router.predict(t, key)["latency_ms"] for t in lat_msgs]

        rows.append({
            "Model": MODEL_NAMES[key],
            "In-scope accuracy": round(float((pred[ins] == gold[ins]).mean()), 4),
            "OOS recall": round(float((pred[~ins] == OOS).mean()), 4),
            "Routing accuracy": round(float(np.mean([a == b for a, b in zip(teams, gold_team)])), 4),
            "Size (MB)": round(model_size_mb(key), 2),
            "Latency mean (ms)": round(float(np.mean(times)), 2),
            "Latency p95 (ms)": round(float(np.percentile(times, 95)), 2),
            "Threshold": router.thresholds.get(key, 0.0),
        })
        print(rows[-1])
        if key == "student_int8":   # save the wrong predictions (for the limitations slide)
            err = test.assign(pred=pred, confidence=[round(r["confidence"], 3) for r in res])
            err[err.pred != err.intent].to_csv(RESULTS_DIR / "errors_student_int8.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    md = (f"# Results on the CLINC150 test set (n = {len(test)}: {ins.sum()} in-scope, "
          f"{(~ins).sum()} out-of-scope)\n\n"
          f"Latency: cleaning + model, mean over {len(lat_msgs)} single messages after 10 warm-up runs.\n\n"
          f"Hardware: {hardware()}\n\n" + df.to_markdown(index=False) + "\n")
    (RESULTS_DIR / "results.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
